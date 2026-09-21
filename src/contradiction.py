# src/contradiction.py
# ─────────────────────────────────────────────────────────────
# V7 CONTRADICTION DETECTION
#
# What this file does:
#   Takes retrieved chunks (already re-ranked if rerank=true)
#   and asks Groq to identify whether any of them reach
#   contradictory conclusions on the same legal question.
#
# WHY THIS IS DIFFERENT FROM EXTRACTION (V5):
#   Extraction (extractor.py) labels components WITHIN a chunk.
#   Contradiction detection compares ACROSS chunks from different
#   cases. It looks at what multiple cases held and identifies
#   where they diverge on the same legal point.
#
# WHY GROQ RATHER THAN A RULE-BASED APPROACH?
#   Legal contradictions are semantic, not syntactic. Two cases
#   can contradict each other without sharing a single keyword.
#   A.K. Gopalan said Article 22 is a complete code. Maneka Gandhi
#   said Article 21 applies alongside Article 22. These are opposite
#   conclusions that no keyword rule could detect.
#   An LLM reading both can identify the logical conflict.
#
# IMPORTANT LIMITATION:
#   This works well when contradicting cases are BOTH in the
#   retrieved top-k. If only one side of a contradiction is
#   retrieved, this module cannot detect it. This is a fundamental
#   retrieval limitation — the system can only reason about what
#   it retrieved. With more cases in the index and better retrieval,
#   this improves automatically.
# ─────────────────────────────────────────────────────────────

import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "qwen/qwen3.8-27b"

SYSTEM_PROMPT = """You are a legal analyst specialising in Supreme Court of India judgments.

Your job is to analyse a set of case excerpts and identify genuine legal contradictions between them.

A genuine legal contradiction exists when:
- Two or more cases address the SAME legal question
- AND reach OPPOSITE or SIGNIFICANTLY DIFFERENT conclusions
- OR when a later case explicitly overrules, distinguishes, or limits an earlier case

Do NOT flag as contradictions:
- Cases that address different legal questions (even if from the same domain)
- Cases that reach similar conclusions through different reasoning
- Cases where one is simply more specific than another

Return ONLY a valid JSON object. No markdown, no explanation, no code fences.
"""

USER_PROMPT_TEMPLATE = """Analyse these Supreme Court of India case excerpts for legal contradictions.

EXCERPTS:
{excerpts}

Return a JSON object with exactly these fields:
{{
  "contradictions_found": true or false,
  "contradiction_count": 0,
  "contradictions": [
    {{
      "legal_question": "The specific legal question both cases address",
      "case_1": "Case name and citation",
      "case_1_position": "What case 1 held on this question",
      "case_2": "Case name and citation",
      "case_2_position": "What case 2 held on this question",
      "significance": "Why this contradiction matters for legal research",
      "overruled": true or false
    }}
  ],
  "analysis_note": "Brief note on the overall relationship between these cases"
}}

If no contradictions found, return contradictions_found=false, contradiction_count=0, contradictions=[], and a brief analysis_note.
"""


# ── Format chunks for contradiction analysis ──────────────────
def build_excerpts(chunks: list[dict]) -> str:
    """
    Format retrieved chunks as a numbered excerpt list.
    Groups chunks by case so the model sees each case's
    position clearly before comparing across cases.

    Only sends case name, citation, outcome, legal principle,
    and a short text excerpt — not the full chunk text.
    This keeps the prompt tight and focused on holdings,
    not on procedural detail.
    """
    # Group by case first
    by_case = {}
    for chunk in chunks:
        src = chunk.get("source_file", "unknown")
        if src not in by_case:
            by_case[src] = {
                "case_name"      : chunk.get("case_name",       "Unknown"),
                "citation"       : chunk.get("citation",        "Unknown"),
                "year"           : chunk.get("year",            0),
                "outcome"        : chunk.get("outcome",         "Unknown"),
                "legal_principle": chunk.get("legal_principle", "Unknown"),
                "excerpts"       : [],
            }
        # Add a short excerpt (first 300 chars of chunk text)
        text = chunk.get("text", "").strip()[:300]
        if text and text not in by_case[src]["excerpts"]:
            by_case[src]["excerpts"].append(text)

    # Format grouped output
    lines = []
    for i, (src, case) in enumerate(by_case.items(), 1):
        lines.append(f"CASE {i}: {case['case_name']} ({case['citation']}, {case['year']})")
        lines.append(f"  Outcome        : {case['outcome']}")
        lines.append(f"  Legal principle: {case['legal_principle']}")
        lines.append(f"  Key excerpts   :")
        for j, exc in enumerate(case["excerpts"][:3], 1):   # max 3 excerpts per case
            lines.append(f"    [{j}] {exc}...")
        lines.append("")

    return "\n".join(lines)


# ── Parse Groq response ───────────────────────────────────────
def parse_contradiction_response(raw_text: str) -> dict:
    """
    Defensive JSON parsing — same pattern as metadata.py.
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


# ── Empty result ──────────────────────────────────────────────
def no_contradiction(note: str = "") -> dict:
    """
    Safe fallback when detection fails or finds nothing.
    """
    return {
        "contradictions_found": False,
        "contradiction_count" : 0,
        "contradictions"      : [],
        "analysis_note"       : note or "No contradictions detected.",
        "error"               : None,
    }


# ── Check if chunks are from multiple cases ───────────────────
def has_multiple_cases(chunks: list[dict]) -> bool:
    """
    Contradiction detection only makes sense when chunks
    come from more than one case. If all chunks are from
    the same document, skip the Groq call entirely.
    """
    sources = set(c.get("source_file", "") for c in chunks)
    return len(sources) > 1


# ── Main detection function ───────────────────────────────────
def detect_contradictions(query: str, chunks: list[dict]) -> dict:
    """
    Main function — call this from main.py.

    Takes the query and retrieved chunks.
    Returns a structured contradiction report.

    Short-circuits if all chunks are from the same case
    (no cross-case comparison possible) to save a Groq call.
    """
    if not chunks:
        return no_contradiction("No chunks provided.")

    if not has_multiple_cases(chunks):
        return no_contradiction(
            "All retrieved chunks are from the same case — "
            "no cross-case contradiction possible. "
            "Add more cases to the index to enable contradiction detection."
        )

    excerpts = build_excerpts(chunks)
    prompt   = USER_PROMPT_TEMPLATE.format(excerpts=excerpts)

    try:
        response = groq_client.chat.completions.create(
            model            = MODEL,
            messages         = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature      = 0.0,
            max_tokens       = 1000,
            reasoning_effort = "none",
        )

        raw_text = response.choices[0].message.content
        parsed   = parse_contradiction_response(raw_text)

        if not parsed:
            return no_contradiction("Could not parse contradiction analysis.")

        # Ensure required fields exist
        return {
            "contradictions_found": bool(parsed.get("contradictions_found", False)),
            "contradiction_count" : int(parsed.get("contradiction_count",   0)),
            "contradictions"      : parsed.get("contradictions",            []),
            "analysis_note"       : parsed.get("analysis_note",             ""),
            "error"               : None,
        }

    except Exception as e:
        return {
            "contradictions_found": False,
            "contradiction_count" : 0,
            "contradictions"      : [],
            "analysis_note"       : "",
            "error"               : str(e),
        }


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from retriever import search

    # This query should retrieve chunks from both Gopalan (1950)
    # and Maneka Gandhi (1978) — which directly contradict each other
    # on the interpretation of Article 21 and personal liberty
    test_queries = [
        "Does Article 21 apply to preventive detention?",
        "What is the scope of personal liberty under the Constitution?",
        "contract breach merchant obligation",   # should find no contradiction
    ]

    for query in test_queries:
        print(f"\n{'═'*65}")
        print(f"Query: {query}")
        print(f"{'─'*65}")

        chunks = search(query, top_k=5, rerank=True)

        # Show which cases were retrieved
        cases = set(c.get("case_name", "?") for c in chunks)
        print(f"Cases retrieved: {', '.join(cases)}")

        result = detect_contradictions(query, chunks)

        print(f"Contradictions found: {result['contradictions_found']}")
        print(f"Count: {result['contradiction_count']}")

        if result["contradictions_found"]:
            for i, c in enumerate(result["contradictions"], 1):
                print(f"\n  Contradiction {i}:")
                print(f"    Question   : {c.get('legal_question', '')}")
                print(f"    Case 1     : {c.get('case_1', '')}")
                print(f"    Position 1 : {c.get('case_1_position', '')}")
                print(f"    Case 2     : {c.get('case_2', '')}")
                print(f"    Position 2 : {c.get('case_2_position', '')}")
                print(f"    Overruled  : {c.get('overruled', False)}")
                print(f"    Significance: {c.get('significance', '')}")

        print(f"\nAnalysis note: {result['analysis_note']}")
        if result["error"]:
            print(f"Error: {result['error']}")