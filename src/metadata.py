# src/metadata.py

import os
import re
import json
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MODEL             = "qwen/qwen3.8-27b"
HEADER_WORD_COUNT = 3000


# ── Groq client rotation ──────────────────────────────────────
def _load_groq_clients():
    clients = []
    i = 1
    while True:
        key = os.getenv(f"GROQ_API_KEY_{i}")
        if not key:
            break
        clients.append(Groq(api_key=key))
        i += 1
    if not clients:
        fallback = os.getenv("GROQ_API_KEY")
        if fallback:
            clients.append(Groq(api_key=fallback))
    if not clients:
        raise RuntimeError("No Groq API keys found in .env")
    print(f"  Loaded {len(clients)} Groq key(s)")
    return clients

_groq_clients = _load_groq_clients()
_groq_key_idx = [0]


def _get_groq_client():
    return _groq_clients[_groq_key_idx[0]]


def _rotate_groq_key() -> bool:
    next_idx = _groq_key_idx[0] + 1
    if next_idx >= len(_groq_clients):
        return False
    _groq_key_idx[0] = next_idx
    print(f"  🔑 Switching to Groq key {next_idx + 1}/{len(_groq_clients)}")
    return True


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


# ── Header extraction ─────────────────────────────────────────
def extract_header(full_text: str) -> str:
    content_markers = [
        "ACT:", "HEADNOTE:", "JUDGMENT:", "HEAD NOTE:",
        "FACTS:", "HELD:", "The petitioner", "The appellant",
        "This is a petition", "This appeal"
    ]
    for marker in content_markers:
        idx = full_text.find(marker)
        if idx != -1:
            start     = max(0, idx - 500)
            remainder = full_text[start:]
            words     = remainder.split()
            return " ".join(words[:HEADER_WORD_COUNT])

    words = full_text.split()
    return " ".join(words[:HEADER_WORD_COUNT])


# ── JSON parsing ──────────────────────────────────────────────
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
    return {}


# ── Validation ────────────────────────────────────────────────
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


# ── Standard extraction (silent fallback) ────────────────────
def extract_metadata(full_text: str, source_file: str) -> dict:
    """
    Soft extraction — catches all errors and returns fallback.
    Used by patch scripts and non-critical contexts.
    """
    try:
        return extract_metadata_strict(full_text, source_file)
    except Exception as e:
        print(f"  ⚠️  Metadata extraction failed: {e}")
        return get_fallback_metadata(source_file)


# ── Strict extraction (raises on API errors) ──────────────────
def extract_metadata_strict(full_text: str, source_file: str) -> dict:
    """
    Hard extraction — raises on API rate limits and quota errors.
    Used by ingest.py so the pipeline stops rather than
    silently storing Unknown metadata.
    """
    header = extract_header(full_text)
    prompt = USER_PROMPT_TEMPLATE.format(text=header)

    while True:
        try:
            response = _get_groq_client().chat.completions.create(
                model            = MODEL,
                messages         = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature      = 0.0,
                max_tokens       = 512,
                reasoning_effort = "none",
            )
            raw_text = response.choices[0].message.content
            raw_dict = parse_json_from_response(raw_text)

            if not raw_dict:
                raise ValueError("Could not parse metadata JSON from response")

            return validate_and_clean(raw_dict, source_file)

        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower() or "RESOURCE_EXHAUSTED" in err:
                if any(x in err.lower() for x in ["day", "1000", "rpd", "daily", "quota"]):
                    # Daily limit hit — try rotating to next key
                    if _rotate_groq_key():
                        continue
                    else:
                        raise RuntimeError(
                            f"All Groq keys exhausted for today: {err}"
                        )
                else:
                    # TPM limit — wait and retry same key
                    print(f"\n  ⏳ Groq TPM limit — waiting 30s...")
                    time.sleep(30)
                    continue
            else:
                raise


# ── Test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    A.K. Gopalan vs The State Of Madras Union Of India on 19 May, 1950
    Equivalent citations: 1950 AIR 27, 1950 SCR 88
    Bench: Kania, H.J. (CJ), Fazal Ali, Saiyid, Patanjali Sastri, M.,
           Mahajan, Mehr Chand, Das, Sudhi Ranjan, Mukherjea, B.K.
    ACT: Constitution of India - Articles 13, 19, 21, 22, 32;
         Preventive Detention Act, 1950 - Sections 3, 7, 12, 14
    HEADNOTE:
    The petitioner, a communist leader, was detained under the Preventive
    Detention Act, 1950. He challenged the constitutional validity of the
    Act contending that it violated Articles 13, 19, 21 and 22 of the
    Constitution of India...
    """
    print("Testing metadata extraction...")
    result = extract_metadata(sample, "A_K_Gopalan_test.PDF")
    print("Extracted metadata:")
    for key, val in result.items():
        print(f"  {key:<18} : {val}")