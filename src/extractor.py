# src/extractor.py
# ─────────────────────────────────────────────────────────────
# V5 LEGAL ARGUMENT EXTRACTION
#
# What this file does:
#   Takes retrieved chunks (already found by retriever.py) and
#   sends each one to Groq to identify the specific legal
#   components within it:
#     - Petitioner arguments
#     - Respondent arguments
#     - Court reasoning
#     - Legal principles applied
#     - Decision/holding
#     - Key facts
#
# WHY AT QUERY TIME (not index time)?
#   Extraction at index time would require re-ingesting all
#   documents and storing extracted components in ChromaDB.
#   Query-time extraction lets us iterate on the extraction
#   prompt without touching the index. The cost is paid per
#   search, not per document.
#
# WHY GROQ (not Gemini)?
#   Groq's LPU hardware makes Llama 3.3 70B respond in ~1s
#   per chunk. We extract up to 5 chunks per query = ~5s total.
#   Gemini would work too but is slower for this structured
#   extraction task.
# ─────────────────────────────────────────────────────────────

import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "qwen/qwen3.6-27b"

SYSTEM_PROMPT = """You are a legal analyst specialising in Supreme Court of India judgments.

Your job is to analyse a chunk of text from a legal judgment and identify the specific legal components present in it.

Return ONLY a valid JSON object. No markdown, no explanation, no code fences.

Rules:
- Extract only what is explicitly present in the text. Do not infer or fabricate.
- If a component is not present in this chunk, use an empty string "".
- Keep each field concise — one to three sentences maximum.
- All values must be strings only.

The six components to identify:

1. petitioner_arguments  : Claims or arguments made BY the petitioner/appellant in favour of their position.
2. respondent_arguments  : Claims or arguments made BY the respondent/state against the petitioner.
3. court_reasoning       : The judges' own analysis, interpretation, and reasoning process.
4. legal_principles      : Specific legal rules, tests, or doctrines the court stated or applied.
5. decision              : What the court actually held or decided in this passage, if present.
6. key_facts             : Factual background relevant to the legal question being decided.

IMPORTANT: A single chunk rarely contains all six components. Most chunks will have two or three.
Do not force-fill empty components — use "" if genuinely absent.
"""

USER_PROMPT_TEMPLATE = """Analyse this excerpt from a Supreme Court of India judgment and extract the legal components present.

Case    : {case_name}
Citation: {citation}
Year    : {year}

Text:
{text}

Return a JSON object with exactly these six fields:
{{
  "petitioner_arguments" : "",
  "respondent_arguments" : "",
  "court_reasoning"      : "",
  "legal_principles"     : "",
  "decision"             : "",
  "key_facts"            : ""
}}
"""


# ── Empty extraction result ───────────────────────────────────
def empty_extraction() -> dict:
    """
    Fallback when extraction fails or returns unparseable output.
    Ensures every chunk always has a complete extraction dict
    with consistent keys — prevents KeyError downstream.
    """
    return {
        "petitioner_arguments": "",
        "respondent_arguments": "",
        "court_reasoning"     : "",
        "legal_principles"    : "",
        "decision"            : "",
        "key_facts"           : "",
    }


# ── Parse JSON from Groq response ────────────────────────────
def parse_extraction(raw_text: str) -> dict:
    """
    Same defensive JSON parsing as metadata.py.
    Strip markdown fences, try json.loads, fall back to regex.
    """
    text = raw_text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*",     "", text)
    text = re.sub(r"\s*```$",     "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    return {}


# ── Extract components from one chunk ────────────────────────
def extract_chunk(chunk: dict) -> dict:
    """
    Send one retrieved chunk to Groq and get back
    a structured dict of legal components.

    Input:  chunk dict from retriever.search()
    Output: extraction dict with six labelled fields
            merged back into the original chunk dict
    """
    prompt = USER_PROMPT_TEMPLATE.format(
        case_name = chunk.get("case_name", "Unknown"),
        citation  = chunk.get("citation",  "Unknown"),
        year      = chunk.get("year",      "Unknown"),
        text      = chunk.get("text",      ""),
    )

    try:
        response = groq_client.chat.completions.create(
            model       = MODEL,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature = 0.0,     # deterministic — extraction not generation
            max_tokens  = 600,     # six short fields — 600 is generous
            reasoning_effort = "none",
        )

        raw_text   = response.choices[0].message.content
        extraction = parse_extraction(raw_text)

        if not extraction:
            return empty_extraction()

        # Ensure all six keys exist even if model omitted some
        result = empty_extraction()
        for key in result:
            if key in extraction and isinstance(extraction[key], str):
                result[key] = extraction[key].strip()

        return result

    except Exception as e:
        print(f"  ⚠️  Extraction failed for chunk {chunk.get('chunk_num', '?')}: {e}")
        return empty_extraction()


# ── Extract components from multiple chunks ───────────────────
def extract_chunks(chunks: list[dict]) -> list[dict]:
    """
    Run extraction on a list of retrieved chunks.
    Returns the same list with each chunk enriched by an
    "extraction" key containing the six labelled components.

    WHY ENRICH RATHER THAN REPLACE?
    The original chunk text and metadata are still needed —
    generator.py uses the raw text for context assembly and
    the metadata for citations. Extraction adds a new key
    rather than replacing anything, so downstream code
    can use whichever representation it needs.
    """
    enriched = []

    for i, chunk in enumerate(chunks):
        print(f"  ⏳ Extracting chunk {i+1}/{len(chunks)} "
              f"(chunk #{chunk.get('chunk_num', '?')})...")

        extraction = extract_chunk(chunk)

        # Merge extraction into chunk dict under "extraction" key
        enriched_chunk = dict(chunk)
        enriched_chunk["extraction"] = extraction
        enriched.append(enriched_chunk)

    return enriched


# ── Summarise what was found across all chunks ────────────────
def summarise_extractions(chunks: list[dict]) -> dict:
    """
    Aggregate extracted components across all chunks into
    a single summary dict. Used by generator.py to build
    a richer context block.

    For each component type, collects all non-empty values
    and joins them. This gives the generator a consolidated
    view of all petitioner arguments found, all court
    reasoning found, etc. — across the full retrieved set.
    """
    summary = {
        "petitioner_arguments": [],
        "respondent_arguments": [],
        "court_reasoning"     : [],
        "legal_principles"    : [],
        "decisions"           : [],
        "key_facts"           : [],
    }

    for chunk in chunks:
        ext = chunk.get("extraction", {})
        src = f"({chunk.get('case_name','?')}, {chunk.get('citation','?')})"

        def add(key, summary_key):
            val = ext.get(key, "").strip()
            if val:
                summary[summary_key].append(f"{val} {src}")

        add("petitioner_arguments", "petitioner_arguments")
        add("respondent_arguments", "respondent_arguments")
        add("court_reasoning",      "court_reasoning")
        add("legal_principles",     "legal_principles")
        add("decision",             "decisions")
        add("key_facts",            "key_facts")

    # Join each list into a single string
    return {k: "\n\n".join(v) for k, v in summary.items()}


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from retriever import search

    query  = "What arguments were made about preventive detention and fundamental rights?"
    chunks = search(query, top_k=3)

    print(f"\nQuery: {query}")
    print(f"Retrieved {len(chunks)} chunks\n")

    enriched = extract_chunks(chunks)

    for chunk in enriched:
        ext = chunk["extraction"]
        print(f"\n{'═'*60}")
        print(f"Chunk #{chunk['chunk_num']} | {chunk['case_name']}")
        print(f"Score: {chunk['score']:.5f} | in_both: {chunk['in_both']}")
        print(f"{'─'*60}")

        for field, label in [
            ("petitioner_arguments", "PETITIONER ARGS"),
            ("respondent_arguments", "RESPONDENT ARGS"),
            ("court_reasoning",      "COURT REASONING"),
            ("legal_principles",     "LEGAL PRINCIPLES"),
            ("decision",             "DECISION"),
            ("key_facts",            "KEY FACTS"),
        ]:
            val = ext.get(field, "")
            if val:
                print(f"\n{label}:\n  {val}")

    print(f"\n{'═'*60}")
    print("AGGREGATED SUMMARY ACROSS ALL CHUNKS:")
    print(f"{'─'*60}")
    summary = summarise_extractions(enriched)
    for key, val in summary.items():
        if val:
            print(f"\n{key.upper()}:\n{val}")