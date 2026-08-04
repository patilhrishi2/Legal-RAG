# src/metadata.py

import os
import re
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL             = "llama-3.3-70b-versatile"
HEADER_WORD_COUNT = 5000

SYSTEM_PROMPT = """You are a legal metadata extractor for Supreme Court of India judgments.
Return ONLY a valid JSON object. No markdown, no explanation, no code fences.
Use "Unknown" for unknown strings, 0 for unknown integers.
All values must be strings or integers only — no lists, no arrays, no nested objects.
Join multiple items with ", ".

BENCH TYPE CLASSIFICATION — follow this decision tree exactly:

Step 1: Count the number of judges in the Bench field.
Step 2: Check if the case involves a constitutional question.
        A constitutional question means: validity of a law under the Constitution,
        interpretation of Fundamental Rights (Part III), or interpretation of
        constitutional provisions. Look for words like "constitutional validity",
        "void under the Constitution", "infringes fundamental rights", "Article 32".
Step 3: Apply the rule:
        - 1 judge                          → "Single Bench"
        - 2 judges                         → "Division Bench"
        - 3 or 4 judges                    → "Full Bench"
        - 5+ judges AND constitutional question → "Constitution Bench"
        - 5+ judges AND NO constitutional question → "Full Bench"

EXAMPLES:
  Bench: 6 judges, case challenges Preventive Detention Act under Article 22 → "Constitution Bench"
  Bench: 5 judges, case is about agency commission in a property sale → "Full Bench"
  Bench: 3 judges, criminal appeal → "Full Bench"
  Bench: 2 judges, contract dispute → "Division Bench"
"""

USER_PROMPT_TEMPLATE = """Extract metadata from this Supreme Court of India judgment.

Return a JSON object with EXACTLY these fields and no others:
{{
  "case_name"       : "Full case name",
  "citation"        : "Primary citation e.g. AIR 1950 SC 27",
  "year"            : 1950,
  "bench_size"      : 6,
  "bench_type"      : "Apply the decision tree from system prompt — Constitution Bench / Full Bench / Division Bench / Single Bench",
  "legal_domain"    : "Pick ONE: Constitutional Law, Criminal Law, Contract Law, Property Law, Family Law, Tax Law, Labour Law, Administrative Law, Civil Law",
  "key_provisions"  : "Comma-separated e.g. Article 21, Article 22, IPC Section 302",
  "outcome"         : "Petition dismissed / Appeal allowed / Petition allowed / Appeal dismissed / Partially allowed",
  "legal_principle" : "Core legal principle in one sentence",
  "petitioner_type" : "Individual / State / Company / Government / NGO / Unknown"
}}

Judgment text:
{text}
"""


HEADER_WORD_COUNT = 5000   # increase from 3000

def extract_header(full_text: str) -> str:
    """
    Extract the most useful portion of the judgment for metadata extraction.

    Problem with Indian Kanoon PDFs: the document header contains a large
    CITATOR INFO block (hundreds of case references like 'F 1951 SC 157')
    that consumes most of the word budget before the actual case content.

    Fix: find where the citator block ends by looking for ACT: or HEADNOTE:
    or JUDGMENT: markers, then take text from that point forward.
    This ensures Groq sees the constitutional question, not just citations.
    """
    # Markers that signal the end of the citator block
    # and the start of meaningful case content
    content_markers = [
        "ACT:", "HEADNOTE:", "JUDGMENT:", "HEAD NOTE:",
        "FACTS:", "HELD:", "The petitioner", "The appellant",
        "This is a petition", "This appeal"
    ]

    for marker in content_markers:
        idx = full_text.find(marker)
        if idx != -1:
            # Take from 500 chars before the marker (to catch any preamble)
            # through HEADER_WORD_COUNT words from that point
            start     = max(0, idx - 500)
            remainder = full_text[start:]
            words     = remainder.split()
            return " ".join(words[:HEADER_WORD_COUNT])

    # Fallback: no marker found, just take first HEADER_WORD_COUNT words
    words = full_text.split()
    return " ".join(words[:HEADER_WORD_COUNT])


def parse_json_from_response(text: str) -> dict:
    text = text.strip()
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

    print("  ⚠️  Could not parse metadata JSON — using fallback")
    return {}


def get_fallback_metadata(source_file: str) -> dict:
    return {
        "case_name"      : "Unknown",
        "citation"       : "Unknown",
        "year"           : 0,
        "bench_size"     : 0,
        "bench_type"     : "Unknown",
        "legal_domain"   : "Unknown",
        "key_provisions" : "Unknown",
        "outcome"        : "Unknown",
        "legal_principle": "Unknown",
        "petitioner_type": "Unknown",
        "source_file"    : source_file,
    }


def validate_and_clean(raw: dict, source_file: str) -> dict:
    fallback = get_fallback_metadata(source_file)

    def safe_str(key):
        val = raw.get(key, fallback[key])
        return str(val).strip()[:200] if val else fallback[key]

    def safe_int(key):
        try:
            return int(raw.get(key, fallback[key]))
        except (ValueError, TypeError):
            return fallback[key]

    return {
        "case_name"      : safe_str("case_name"),
        "citation"       : safe_str("citation"),
        "year"           : safe_int("year"),
        "bench_size"     : safe_int("bench_size"),
        "bench_type"     : safe_str("bench_type"),
        "legal_domain"   : safe_str("legal_domain"),
        "key_provisions" : safe_str("key_provisions"),
        "outcome"        : safe_str("outcome"),
        "legal_principle": safe_str("legal_principle"),
        "petitioner_type": safe_str("petitioner_type"),
        "source_file"    : source_file,
    }


def extract_metadata(full_text: str, source_file: str) -> dict:
    header = extract_header(full_text)
    prompt = USER_PROMPT_TEMPLATE.format(text=header)

    try:
        response = groq_client.chat.completions.create(
            model      = MODEL,
            messages   = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            temperature = 0.0,
            max_tokens  = 512,
        )
        raw_text = response.choices[0].message.content
        raw_dict = parse_json_from_response(raw_text)

        if not raw_dict:
            return get_fallback_metadata(source_file)

        return validate_and_clean(raw_dict, source_file)

    except Exception as e:
        print(f"  ⚠️  Metadata extraction failed: {e}")
        return get_fallback_metadata(source_file)


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    A.K. Gopalan vs The State Of Madras Union Of India on 19 May, 1950
    Equivalent citations: 1950 AIR 27, 1950 SCR 88
    Bench: Kania, H.J. (CJ), Fazal Ali, Saiyid, Patanjali Sastri, M.,
           Mahajan, Mehr Chand, Das, Sudhi Ranjan, Mukherjea, B.K.
    ACT: Constitution of India - Articles 13, 19, 21, 22, 32;
         Preventive Detention Act, 1950
    HEADNOTE: The petitioner challenged the constitutional validity of the
    Preventive Detention Act contending it violated Articles 19, 21 and 22.
    """
    result = extract_metadata(sample, "A_K_Gopalan_test.PDF")
    print("\nExtracted metadata:")
    for k, v in result.items():
        print(f"  {k:<18} : {v}")