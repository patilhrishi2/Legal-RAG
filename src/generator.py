# src/generator.py
# V5 — context assembly now uses structured extraction when available

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

GENERATION_MODEL = "gemini-3.5-flash"
MAX_CONTEXT_CHUNKS = 5

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


# ── Build context from raw chunks (V4 behaviour) ──────────────
def build_raw_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Format retrieved chunks as numbered excerpts.
    Used when extraction is not requested (generate=true, extract=false).
    Identical to V4 build_context().
    """
    context_parts = []
    sources       = []

    for i, chunk in enumerate(chunks[:MAX_CONTEXT_CHUNKS], 1):
        case_name = chunk.get("case_name",    "Unknown Case")
        citation  = chunk.get("citation",     "Unknown Citation")
        year      = chunk.get("year",         "Unknown Year")
        domain    = chunk.get("legal_domain", "Unknown Domain")
        chunk_num = chunk.get("chunk_num",    -1)
        text      = chunk.get("text",         "")

        context_parts.append(
            f"[EXCERPT {i}]\n"
            f"Case      : {case_name}\n"
            f"Citation  : {citation}\n"
            f"Year      : {year}\n"
            f"Domain    : {domain}\n"
            f"Text      :\n{text}\n"
        )

        sources.append({
            "excerpt_num": i,
            "case_name"  : case_name,
            "citation"   : citation,
            "year"       : year,
            "chunk_num"  : chunk_num,
            "score"      : chunk.get("score",   0),
            "in_both"    : chunk.get("in_both", False),
        })

    context_text  = "\n" + ("─" * 60) + "\n"
    context_text += "\n\n".join(context_parts)
    context_text += "\n" + ("─" * 60)

    return context_text, sources


# ── Build context from structured extraction (V5 behaviour) ───
def build_structured_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Format enriched chunks (with extraction dicts) into a structured
    context block that separates argument types.

    WHY THIS IS BETTER THAN RAW CONTEXT:
    Raw context gives the LLM 400 words of mixed content per chunk.
    Structured context gives it labelled sections — petitioner argued X,
    court held Y, principle established Z. The model can then answer
    argument-specific queries precisely rather than scanning raw text.

    Only includes non-empty extraction fields to keep context tight.
    """
    context_parts = []
    sources       = []

    for i, chunk in enumerate(chunks[:MAX_CONTEXT_CHUNKS], 1):
        case_name = chunk.get("case_name",    "Unknown Case")
        citation  = chunk.get("citation",     "Unknown Citation")
        year      = chunk.get("year",         "Unknown Year")
        chunk_num = chunk.get("chunk_num",    -1)
        ext       = chunk.get("extraction",   {})

        # Build structured section — only include non-empty fields
        section = (
            f"[EXCERPT {i}]\n"
            f"Case     : {case_name}\n"
            f"Citation : {citation}\n"
            f"Year     : {year}\n"
        )

        field_labels = [
            ("petitioner_arguments", "Petitioner argued"),
            ("respondent_arguments", "Respondent argued"),
            ("court_reasoning",      "Court reasoning"),
            ("legal_principles",     "Legal principles"),
            ("decision",             "Decision/Holding"),
            ("key_facts",            "Key facts"),
        ]

        has_extraction = False
        for field, label in field_labels:
            val = ext.get(field, "").strip()
            if val:
                section += f"{label}: {val}\n"
                has_extraction = True

        # If extraction is empty for this chunk, fall back to raw text
        if not has_extraction:
            section += f"Text:\n{chunk.get('text', '')}\n"

        context_parts.append(section)
        sources.append({
            "excerpt_num"          : i,
            "case_name"            : case_name,
            "citation"             : citation,
            "year"                 : year,
            "chunk_num"            : chunk_num,
            "score"                : chunk.get("score",   0),
            "in_both"              : chunk.get("in_both", False),
            "extraction_available" : has_extraction,
        })

    context_text  = "\n" + ("─" * 60) + "\n"
    context_text += "\n\n".join(context_parts)
    context_text += "\n" + ("─" * 60)

    return context_text, sources


# ── Build user prompt ─────────────────────────────────────────
def build_prompt(query: str, context: str, structured: bool = False) -> str:
    mode_note = (
        "The excerpts below have been pre-analysed and broken into "
        "labelled components (petitioner arguments, court reasoning, etc.). "
        "Use these labels to give a precise, argument-aware answer.\n\n"
        if structured else ""
    )
    return (
        f"{mode_note}"
        f"Below are excerpts from Supreme Court of India judgments retrieved for your question.\n"
        f"Use ONLY these excerpts to answer. Do not use any other knowledge.\n\n"
        f"RETRIEVED EXCERPTS:\n{context}\n\n"
        f"QUESTION:\n{query}\n\n"
        f"Remember: cite every claim as (Case Name, Citation). "
        f"If the answer is not in the excerpts, say so explicitly."
    )


# ── Parse response (unchanged from V4) ───────────────────────
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

    section_map = {
        "ANSWER"              : "answer",
        "KEY LEGAL PRINCIPLES": "key_legal_principles",
        "SOURCES USED"        : "sources_used",
        "CONFIDENCE"          : "confidence",
    }

    for line in lines:
        stripped = line.strip().upper()
        matched  = False

        for header, key in section_map.items():
            if stripped.startswith(header):
                if current_section:
                    sections[current_section] = "\n".join(buffer).strip()
                current_section = key
                buffer          = []
                original        = line.strip()
                colon_idx       = original.find(":")
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

        # Fallback: if confidence still UNKNOWN, infer from answer content
    if sections["confidence"] == "UNKNOWN":
        answer_lower = sections["answer"].lower()
        if "do not contain sufficient information" in answer_lower:
            sections["confidence"] = "LOW"
        elif sections["answer"] and len(sections["answer"]) > 200:
            sections["confidence"] = "MEDIUM"
        else:
            sections["confidence"] = "LOW"

    return sections



# ── Main generation function ──────────────────────────────────
def generate_answer(query: str, chunks: list[dict],
                    use_extraction: bool = False) -> dict:
    """
    Generate a grounded answer from retrieved chunks.

    use_extraction=False  →  V4 behaviour, raw chunk text as context
    use_extraction=True   →  V5 behaviour, structured extraction as context
                             (chunks must already have "extraction" key
                              added by extractor.extract_chunks())
    """
    if not chunks:
        return {
            "answer"               : "No relevant documents were retrieved.",
            "key_legal_principles" : "",
            "sources_used"         : "",
            "confidence"           : "LOW",
            "sources"              : [],
            "full_response"        : "",
            "error"                : None,
        }

    # Choose context builder based on whether extraction was run
    if use_extraction:
        context, sources = build_structured_context(chunks)
    else:
        context, sources = build_raw_context(chunks)

    prompt = build_prompt(query, context, structured=use_extraction)

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
                temperature        = 0.1,
                max_output_tokens  = 3000,
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
            "extraction_used"      : use_extraction,
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
            "extraction_used"      : use_extraction,
        }


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from retriever import search
    from extractor import extract_chunks

    query  = "What arguments did the petitioner make about fundamental rights and preventive detention?"
    chunks = search(query, top_k=3)

    print(f"\n{'═'*65}")
    print("MODE: V4 (raw chunks, no extraction)")
    print(f"{'═'*65}")
    result = generate_answer(query, chunks, use_extraction=False)
    print(f"\nANSWER:\n{result['answer']}")
    print(f"\nCONFIDENCE: {result['confidence']}")

    print(f"\n{'═'*65}")
    print("MODE: V5 (structured extraction)")
    print(f"{'═'*65}")
    enriched = extract_chunks(chunks)
    result   = generate_answer(query, enriched, use_extraction=True)
    print(f"\nANSWER:\n{result['answer']}")
    print(f"\nKEY LEGAL PRINCIPLES:\n{result['key_legal_principles']}")
    print(f"\nCONFIDENCE: {result['confidence']}")