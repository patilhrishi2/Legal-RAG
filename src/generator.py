# src/generator.py
# ─────────────────────────────────────────────────────────────
# V4 GENERATION PIPELINE
#
# What this file does:
#   1. Takes retrieved chunks from retriever.py (V3 output)
#   2. Assembles them into a numbered context block
#   3. Builds a grounded prompt with strict citation instructions
#   4. Calls Gemini 3.5 Flash for generation
#   5. Returns structured response with answer + sources used
#
# Key principle: the LLM is ONLY allowed to use the retrieved
# chunks as its source of truth. It cannot draw on its own
# training knowledge. This is what "grounded generation" means
# and it is the primary defence against hallucination in RAG.
# ─────────────────────────────────────────────────────────────

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

GENERATION_MODEL = "gemini-3.5-flash"

# Maximum chunks to include in context.
# More chunks = more context = better coverage but:
#   (a) longer prompts cost more tokens
#   (b) LLMs lose focus on very long contexts ("lost in the middle" problem)
# 5 is the standard starting point for RAG generation.
MAX_CONTEXT_CHUNKS = 5


# ── System prompt ─────────────────────────────────────────────
# This is sent once as the system instruction.
# It establishes the model's role and hard constraints.
SYSTEM_PROMPT = """You are a legal research assistant specialising in Supreme Court of India judgments.

Your job is to answer legal questions based EXCLUSIVELY on the case excerpts provided to you.

STRICT RULES — follow these without exception:
1. Use ONLY the provided case excerpts to construct your answer. Do not use your own legal knowledge or training data.
2. Every factual claim you make MUST be followed by a citation in this format: (Case Name, Citation)
3. If the provided excerpts do not contain enough information to answer the question, say exactly: "The retrieved documents do not contain sufficient information to answer this question." Do not guess or infer beyond what is explicitly stated.
4. If different excerpts contain contradictory positions on the same legal point, explicitly flag this: "Note: The sources contain conflicting positions on this point."
5. Do not fabricate case names, citations, dates, or legal principles.
6. Keep your answer focused and structured. Do not pad with generic legal commentary.

Your answer must follow this exact structure:

ANSWER:
[Your grounded answer with inline citations]

KEY LEGAL PRINCIPLES:
[Bullet points of the core legal principles found in the excerpts, each with a citation]

SOURCES USED:
[Numbered list of the cases you actually cited, with their citations]

CONFIDENCE:
[One of: HIGH (answer clearly supported by excerpts) / MEDIUM (partially supported) / LOW (inferred from limited context)]
"""


# ── Build numbered context block ──────────────────────────────
def build_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Format retrieved chunks into a numbered context block for the LLM.

    WHY NUMBERING?
    The LLM needs a stable reference system to cite specific chunks.
    "As stated in excerpt [3]..." is more reliable than asking the model
    to memorise and reproduce case names correctly mid-generation.

    Returns:
        context_text : the formatted string sent to the LLM
        sources      : list of source dicts (for the API response)
    """
    context_parts = []
    sources       = []

    for i, chunk in enumerate(chunks[:MAX_CONTEXT_CHUNKS], 1):
        case_name  = chunk.get("case_name",  "Unknown Case")
        citation   = chunk.get("citation",   "Unknown Citation")
        year       = chunk.get("year",       "Unknown Year")
        domain     = chunk.get("legal_domain", "Unknown Domain")
        chunk_num  = chunk.get("chunk_num",  -1)
        text       = chunk.get("text",       "")

        context_parts.append(
            f"[EXCERPT {i}]\n"
            f"Case      : {case_name}\n"
            f"Citation  : {citation}\n"
            f"Year      : {year}\n"
            f"Domain    : {domain}\n"
            f"Text      :\n{text}\n"
        )

        sources.append({
            "excerpt_num" : i,
            "case_name"   : case_name,
            "citation"    : citation,
            "year"        : year,
            "chunk_num"   : chunk_num,
            "score"       : chunk.get("score", 0),
            "in_both"     : chunk.get("in_both", False),
        })

    context_text = "\n" + ("─" * 60) + "\n"
    context_text += "\n\n".join(context_parts)
    context_text += "\n" + ("─" * 60)

    return context_text, sources


# ── Build user prompt ─────────────────────────────────────────
def build_prompt(query: str, context: str) -> str:
    """
    Assemble the full user-turn prompt.

    Structure:
      1. The retrieved excerpts (context)
      2. The user's question
      3. Reminder of citation format

    WHY PUT CONTEXT BEFORE QUESTION?
    Research shows LLMs attend better to context placed before
    the question than after. With long contexts, placing the
    question last also makes the instruction immediately visible
    right before the model starts generating.
    """
    return f"""Below are excerpts from Supreme Court of India judgments retrieved for your question.
Use ONLY these excerpts to answer. Do not use any other knowledge.

RETRIEVED EXCERPTS:
{context}

QUESTION:
{query}

Remember: cite every claim as (Case Name, Citation). If the answer is not in the excerpts, say so explicitly.
"""


# ── Parse generated response ──────────────────────────────────
def parse_response(raw_text: str) -> dict:
    sections = {
        "answer"              : "",
        "key_legal_principles": "",
        "sources_used"        : "",
        "confidence"          : "UNKNOWN",
        "full_response"       : raw_text,
    }

    current_section = None
    lines           = raw_text.split("\n")
    buffer          = []

    # More flexible matching — check if line CONTAINS the header
    # rather than startswith, handles slight variations
    section_map = {
        "ANSWER"              : "answer",
        "KEY LEGAL PRINCIPLES": "key_legal_principles",
        "SOURCES USED"        : "sources_used",
        "CONFIDENCE"          : "confidence",
    }

    for line in lines:
        stripped = line.strip().upper()

        matched = False
        for header, key in section_map.items():
            if stripped.startswith(header):
                if current_section:
                    sections[current_section] = "\n".join(buffer).strip()
                current_section = key
                buffer = []
                # grab anything after the header and colon on the same line
                original = line.strip()
                colon_idx = original.find(":")
                if colon_idx != -1:
                    inline = original[colon_idx + 1:].strip()
                    if inline:
                        buffer.append(inline)
                matched = True
                break

        if not matched and current_section:
            buffer.append(line)

    if current_section and buffer:
        sections[current_section] = "\n".join(buffer).strip()

    confidence_raw = sections["confidence"].upper()
    for level in ["HIGH", "MEDIUM", "LOW"]:
        if level in confidence_raw:
            sections["confidence"] = level
            break

    return sections


# ── Main generation function ──────────────────────────────────
def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Full generation pipeline — call this from main.py.

    Takes:
        query  : the user's original question
        chunks : list of retrieved chunk dicts from retriever.search()

    Returns a dict with:
        answer               : grounded natural language answer
        key_legal_principles : bullet points of principles found
        sources_used         : text list of cited cases
        confidence           : HIGH / MEDIUM / LOW
        sources              : structured list of chunks used as context
        full_response        : raw LLM output (for debugging)
        error                : set if generation failed
    """
    if not chunks:
        return {
            "answer"               : "No relevant documents were retrieved for this query.",
            "key_legal_principles" : "",
            "sources_used"         : "",
            "confidence"           : "LOW",
            "sources"              : [],
            "full_response"        : "",
            "error"                : None,
        }

    # Build context and prompt
    context, sources = build_context(chunks)
    prompt           = build_prompt(query, context)

    try:
        response = gemini_client.models.generate_content(
            model    = GENERATION_MODEL,
            contents = [
                types.Content(
                    role  = "user",
                    parts = [types.Part(text=prompt)]
                )
            ],
            config = types.GenerateContentConfig(
                system_instruction = SYSTEM_PROMPT,
                temperature        = 0.1,   # low but not zero
                                            # zero = too rigid, may refuse to synthesise
                                            # 0.1 = slight flexibility for natural phrasing
                                            # while keeping answers grounded
                max_output_tokens  = 3000,  # enough for a detailed legal answer
            )
        )

        raw_text = response.text
        parsed   = parse_response(raw_text)

        return {
            "answer"               : parsed["answer"],
            "key_legal_principles" : parsed["key_legal_principles"],
            "sources_used"         : parsed["sources_used"],
            "confidence"           : parsed["confidence"],
            "sources"              : sources,
            "full_response"        : parsed["full_response"],
            "error"                : None,
        }

    except Exception as e:
        return {
            "answer"               : "",
            "key_legal_principles" : "",
            "sources_used"         : "",
            "confidence"           : "LOW",
            "sources"              : sources,
            "full_response"        : "",
            "error"                : str(e),
        }


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from retriever import search

    test_queries = [
        # Test 1 — answer clearly in the index
        "What are the constitutional limits on preventive detention?",

        # Test 2 — answer NOT in the index (hallucination test)
        # Your index only has 1950 cases — this should trigger the refusal
        "What did the Supreme Court rule about right to privacy in 2017?",
    ]

    for query in test_queries:
        print(f"\n{'═'*65}")
        print(f"QUERY: {query}")
        print(f"{'═'*65}")

        chunks = search(query, top_k=5)
        result = generate_answer(query, chunks)

        if result["error"]:
            print(f"ERROR: {result['error']}")
            continue

        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nKEY LEGAL PRINCIPLES:\n{result['key_legal_principles']}")
        print(f"\nSOURCES USED:\n{result['sources_used']}")
        print(f"\nCONFIDENCE: {result['confidence']}")
        print(f"\nCHUNKS USED AS CONTEXT:")
        for s in result["sources"]:
            both = "both pipelines" if s["in_both"] else "one pipeline"
            print(f"  [{s['excerpt_num']}] {s['case_name']} | {s['citation']} | chunk #{s['chunk_num']} | score={s['score']:.5f} | {both}")