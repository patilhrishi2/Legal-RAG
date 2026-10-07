# src/metadata.py

import os
import re
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────
HEADER_WORD_COUNT = 3000

# Google AI Studio silently maps model strings — use the one that works.
# Confirmed working via debug output: gemini-3.5-flash-lite
GEMMA_MODEL = "gemini-3.5-flash-lite"

# Rate limits per key: 15 RPM, 500 RPD, 250K TPM
# With 10 keys: 5,000 files/day capacity
RPD_LIMIT_METADATA = 490   # stop at 490/500 to leave headroom per key

# Debug logging — set False once confirmed working at scale
DEBUG_LOG  = False
DEBUG_FILE = "storage/metadata_debug.json"


# ── Key management ────────────────────────────────────────────
def _load_gemini_keys() -> list[str]:
    keys = []
    i = 1
    while True:
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if not key:
            break
        key = key.split("#")[0].strip().replace("\xa0", "")
        if key:
            keys.append(key)
        i += 1
    if not keys:
        fallback = os.getenv("GEMINI_API_KEY", "").strip()
        if fallback:
            keys.append(fallback)
    if not keys:
        raise RuntimeError("No Gemini API keys found in .env")
    return keys


_gemini_keys      = _load_gemini_keys()
_meta_key_idx     = [0]
_meta_key_counter = [0]   # RPD calls on current key


def _get_api_key() -> str:
    return _gemini_keys[_meta_key_idx[0]]


def _rotate_metadata_key() -> bool:
    """Rotate to next key. Returns True if a new key is available."""
    next_idx = _meta_key_idx[0] + 1
    if next_idx >= len(_gemini_keys):
        return False
    _meta_key_idx[0]     = next_idx
    _meta_key_counter[0] = 0
    print(f"\n  🔑 Metadata: switching to Gemini key "
          f"{next_idx + 1}/{len(_gemini_keys)}")
    return True


# ── Prompts ───────────────────────────────────────────────────
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
    if not text:
        return {}
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


# ── Debug logging ─────────────────────────────────────────────
def _log_debug(source_file: str, raw_response: dict,
               raw_text: str, parsed: dict):
    if not DEBUG_LOG:
        return
    os.makedirs(os.path.dirname(DEBUG_FILE), exist_ok=True)
    existing = []
    if os.path.exists(DEBUG_FILE):
        try:
            with open(DEBUG_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.append({
        "source_file"     : source_file,
        "raw_api_response": raw_response,
        "extracted_text"  : raw_text,
        "parsed_result"   : parsed,
    })
    with open(DEBUG_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


# ── Fallback ──────────────────────────────────────────────────
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


# ── Validation ────────────────────────────────────────────────
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


# ── REST API call ─────────────────────────────────────────────
def _call_metadata_api(prompt: str) -> tuple[dict, str]:
    """
    Call via REST API directly.
    Returns (raw_response_dict, extracted_text).
    """
    api_key = _get_api_key()
    url     = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMMA_MODEL}:generateContent?key={api_key}"
    )

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role" : "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature"    : 0.0,
            "maxOutputTokens": 512,
        }
    }

    response = requests.post(
        url,
        headers = {"Content-Type": "application/json"},
        json    = payload,
        timeout = 60,
    )

    raw_dict = response.json()

    if response.status_code != 200:
        err     = raw_dict.get("error", {})
        status  = err.get("status",  "")
        message = err.get("message", str(raw_dict))
        code    = str(response.status_code)
        raise RuntimeError(f"API {code} [{status}]: {message}")

    try:
        text = raw_dict["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        text = None

    return raw_dict, text


# ── Strict extraction ─────────────────────────────────────────
def extract_metadata_strict(full_text: str, source_file: str) -> dict:
    """
    Hard extraction — raises RuntimeError on daily quota exhaustion.
    Used by ingest.py so the pipeline stops cleanly rather than
    silently storing Unknown metadata.
    """
    header = extract_header(full_text)
    prompt = USER_PROMPT_TEMPLATE.format(text=header)

    while True:
        # Check RPD for current key before making call
        if _meta_key_counter[0] >= RPD_LIMIT_METADATA:
            if _rotate_metadata_key():
                pass   # counter already reset in rotate
            else:
                raise RuntimeError(
                    "All Gemini keys exhausted for metadata extraction today."
                )

        try:
            raw_api, raw_text = _call_metadata_api(prompt)
            _meta_key_counter[0] += 1

            _log_debug(source_file, raw_api, raw_text or "", {})

            raw_dict = parse_json_from_response(raw_text or "")

            _log_debug(source_file, raw_api, raw_text or "", raw_dict)

            if not raw_dict:
                raise ValueError(
                    f"Empty parse result. Raw text: {repr(raw_text)}"
                )

            return validate_and_clean(raw_dict, source_file)

        except RuntimeError as e:
            err = str(e)
            # Check for daily quota in the error message
            if any(x in err.lower() for x in
                   ["429", "resource_exhausted", "quota",
                    "day", "daily", "per day"]):
                if _rotate_metadata_key():
                    continue
                raise RuntimeError(
                    f"All Gemini metadata keys exhausted: {err}"
                )
            raise

        except Exception as e:
            err = str(e)
            if any(x in err.lower() for x in
                   ["429", "resource_exhausted", "quota"]):
                if "day" in err.lower() or "daily" in err.lower():
                    if _rotate_metadata_key():
                        continue
                    raise RuntimeError(
                        f"All Gemini metadata keys exhausted: {err}"
                    )
                else:
                    print(f"\n  ⏳ Metadata rate limit — waiting 30s...")
                    time.sleep(30)
                    continue
            raise


# ── Soft extraction ───────────────────────────────────────────
def extract_metadata(full_text: str, source_file: str) -> dict:
    """
    Soft extraction — catches all errors, returns Unknown fallback.
    Use extract_metadata_strict for ingestion.
    """
    try:
        return extract_metadata_strict(full_text, source_file)
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
         Preventive Detention Act, 1950 - Sections 3, 7, 12, 14
    HEADNOTE:
    The petitioner, a communist leader, was detained under the Preventive
    Detention Act, 1950. He challenged the constitutional validity of the
    Act contending that it violated Articles 13, 19, 21 and 22 of the
    Constitution of India. The Supreme Court dismissed the petition.
    """

    print(f"Testing with {GEMMA_MODEL} via REST...\n")
    result = extract_metadata(sample, "A_K_Gopalan_test.PDF")
    print("Extracted metadata:")
    for key, val in result.items():
        print(f"  {key:<18} : {val}")
    if DEBUG_LOG:
        print(f"\nDebug saved to: {DEBUG_FILE}")