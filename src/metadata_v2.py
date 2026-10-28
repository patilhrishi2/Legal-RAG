# src/metadata_v2.py  (new; drop-in for metadata.py)
import os
import re
import json
import time
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────
HEADER_WORD_COUNT  = 3000
TAIL_WORD_COUNT    = 700
GEMMA_MODEL        = "gemini-3.5-flash-lite"
RPD_LIMIT_METADATA = 490
MIN_CALL_GAP       = 4.2      # seconds between calls (15 RPM limit)
TPM_WAIT_SECS      = 35
TPM_MAX_RETRIES    = 3
USAGE_FILE         = "storage_v2/meta_usage.json"

DOMAINS = ["Constitutional Law", "Criminal Law", "Contract Law", "Property Law",
           "Family Law", "Tax Law", "Labour Law", "Administrative Law", "Civil Law"]

DEBUG_LOG  = False
DEBUG_FILE = "storage_v2/metadata_debug.json"


# ── Daily usage persistence ───────────────────────────────────
def _quota_day() -> str:
    """Google daily quota resets at midnight Pacific time."""
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/Los_Angeles"))
    except Exception:
        now = datetime.now(timezone(timedelta(hours=-7)))
    return now.strftime("%Y-%m-%d")


def _load_usage() -> dict:
    try:
        with open(USAGE_FILE, "r") as f:
            data = json.load(f)
        if data.get("day") == _quota_day():
            return data
    except Exception:
        pass
    return {"day": _quota_day(), "counts": {}}


def _save_usage():
    os.makedirs(os.path.dirname(USAGE_FILE), exist_ok=True)
    with open(USAGE_FILE, "w") as f:
        json.dump(_usage, f)


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


_gemini_keys = _load_gemini_keys()
_usage       = _load_usage()
_last_call   = [0.0]


def _used(idx: int) -> int:
    return _usage["counts"].get(str(idx), 0)


def _bump(idx: int):
    _usage["counts"][str(idx)] = _used(idx) + 1
    _save_usage()


def _mark_exhausted(idx: int):
    _usage["counts"][str(idx)] = RPD_LIMIT_METADATA
    _save_usage()


# start on the first key that still has quota today
_meta_key_idx = [0]
for _i in range(len(_gemini_keys)):
    if _used(_i) < RPD_LIMIT_METADATA:
        _meta_key_idx[0] = _i
        break
else:
    _meta_key_idx[0] = len(_gemini_keys) - 1


def _get_api_key() -> str:
    return _gemini_keys[_meta_key_idx[0]]


def _rotate_metadata_key() -> bool:
    """Move to the next key with quota left. Returns False if none."""
    for nxt in range(_meta_key_idx[0] + 1, len(_gemini_keys)):
        if _used(nxt) < RPD_LIMIT_METADATA:
            _meta_key_idx[0] = nxt
            print(f"\n  🔑 Metadata: switching to Gemini key {nxt + 1}/{len(_gemini_keys)}")
            return True
    return False


def is_daily_quota_error(err: str) -> bool:
    e = err.lower().replace(" ", "")
    return any(x in e for x in ["perday", "daily", "requestsperday"]) and \
           "perminute" not in e


# ── Prompts ───────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a legal metadata extractor for Supreme Court of India judgments.
Return ONLY a valid JSON object. No markdown, no explanation, no code fences.
Use "Unknown" for unknown strings, 0 for unknown integers.
All values must be strings or integers only — no lists, no arrays, no nested objects.
Join multiple items with ", ".
The text you receive is the beginning of the judgment followed by its final part
(marked "[... END OF JUDGMENT ...]"). Take "outcome" from the final part.

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
  Bench: 5 judges, case challenges a State law under Article 19 → "Constitution Bench"
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
  "bench_size"      : 5,
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

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "case_name":       {"type": "STRING"},
        "citation":        {"type": "STRING"},
        "year":            {"type": "INTEGER"},
        "bench_size":      {"type": "INTEGER"},
        "bench_type":      {"type": "STRING"},
        "legal_domain":    {"type": "STRING", "enum": DOMAINS},
        "key_provisions":  {"type": "STRING"},
        "outcome":         {"type": "STRING"},
        "legal_principle": {"type": "STRING"},
        "petitioner_type": {"type": "STRING"},
    },
    "required": ["case_name", "citation", "year", "bench_size", "bench_type",
                 "legal_domain", "key_provisions", "outcome",
                 "legal_principle", "petitioner_type"],
}


# ── Header + tail extraction ──────────────────────────────────
def extract_header(full_text: str) -> str:
    content_markers = [
        "ACT:", "HEADNOTE:", "JUDGMENT:", "HEAD NOTE:",
        "FACTS:", "HELD:", "The petitioner", "The appellant",
        "This is a petition", "This appeal"
    ]
    all_words = full_text.split()
    head = " ".join(all_words[:HEADER_WORD_COUNT])
    for marker in content_markers:
        idx = full_text.find(marker)
        if idx != -1:
            start = max(0, idx - 500)
            head  = " ".join(full_text[start:].split()[:HEADER_WORD_COUNT])
            break
    # always include the case title / bench lines from the very top
    top = " ".join(all_words[:120])
    if top not in head:
        head = top + " ... " + head
    # add the end of the judgment so the outcome is visible
    if len(all_words) > HEADER_WORD_COUNT + TAIL_WORD_COUNT:
        head += "\n[... END OF JUDGMENT ...]\n" + " ".join(all_words[-TAIL_WORD_COUNT:])
    return head


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
def _log_debug(source_file: str, raw_response: dict, raw_text: str, parsed: dict):
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
    existing.append({"source_file": source_file, "raw_api_response": raw_response,
                     "extracted_text": raw_text, "parsed_result": parsed})
    with open(DEBUG_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


# ── Fallback ──────────────────────────────────────────────────
def get_fallback_metadata(source_file: str) -> dict:
    return {
        "case_name": "Unknown", "citation": "Unknown", "year": 0, "bench_size": 0,
        "bench_type": "Unknown", "legal_domain": "Unknown", "key_provisions": "Unknown",
        "outcome": "Unknown", "legal_principle": "Unknown", "petitioner_type": "Unknown",
        "source_file": source_file,
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

    out = {
        "case_name": safe_str("case_name"), "citation": safe_str("citation"),
        "year": safe_int("year"), "bench_size": safe_int("bench_size"),
        "bench_type": safe_str("bench_type"), "legal_domain": safe_str("legal_domain"),
        "key_provisions": safe_str("key_provisions"), "outcome": safe_str("outcome"),
        "legal_principle": safe_str("legal_principle"),
        "petitioner_type": safe_str("petitioner_type"), "source_file": source_file,
    }
    # normalise domain to the allowed list (case-insensitive)
    match = next((d for d in DOMAINS if d.lower() == out["legal_domain"].lower()), None)
    out["legal_domain"] = match or "Unknown"
    return out


# ── REST API call ─────────────────────────────────────────────
def _call_metadata_api(prompt: str) -> tuple[dict, str]:
    # keep at least MIN_CALL_GAP seconds between calls (15 RPM limit)
    wait = MIN_CALL_GAP - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMMA_MODEL}:generateContent")
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 1024,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        },
    }
    response = requests.post(
        url,
        headers={"Content-Type": "application/json",
                 "x-goog-api-key": _get_api_key()},
        json=payload, timeout=60,
    )
    try:
        raw_dict = response.json()
    except Exception:
        raw_dict = {"error": {"message": response.text[:300]}}

    if response.status_code != 200:
        err = raw_dict.get("error", {})
        raise RuntimeError(f"API {response.status_code} [{err.get('status', '')}]: "
                           f"{err.get('message', str(raw_dict))}")
    try:
        text = raw_dict["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        text = None
    return raw_dict, text


# ── Strict extraction ─────────────────────────────────────────
def extract_metadata_strict(full_text: str, source_file: str) -> dict:
    """
    Raises RuntimeError only when every key is out of daily quota.
    Raises ValueError for a bad/garbage result (caller counts it as a failure).
    429 per-minute -> wait and retry; 429 per-day -> rotate key.
    """
    prompt = USER_PROMPT_TEMPLATE.format(text=extract_header(full_text))
    tpm_retries = 0
    net_retries = 0

    while True:
        if _used(_meta_key_idx[0]) >= RPD_LIMIT_METADATA:
            if not _rotate_metadata_key():
                raise RuntimeError("All Gemini keys exhausted for metadata extraction today.")
            tpm_retries = 0

        try:
            raw_api, raw_text = _call_metadata_api(prompt)
            _bump(_meta_key_idx[0])
            net_retries = 0

            raw_dict = parse_json_from_response(raw_text or "")
            _log_debug(source_file, raw_api, raw_text or "", raw_dict)
            if not raw_dict:
                raise ValueError(f"Empty parse result. Raw text: {repr(raw_text)}")

            meta = validate_and_clean(raw_dict, source_file)
            if meta["case_name"] == "Unknown" or meta["legal_domain"] == "Unknown":
                raise ValueError(f"Low-quality metadata (case_name/legal_domain unknown) "
                                 f"for {source_file}")
            return meta

        except RuntimeError as e:
            err = str(e)
            low = err.lower()
            if "403" in err or "permission_denied" in low:
                print(f"\n  ⛔ Metadata key {_meta_key_idx[0]+1}: 403, skipping key")
                _mark_exhausted(_meta_key_idx[0])
                continue
            if "429" in err or "resource_exhausted" in low:
                if is_daily_quota_error(err):
                    print(f"\n  ⚠️  Metadata key {_meta_key_idx[0]+1}: daily quota exhausted")
                    _mark_exhausted(_meta_key_idx[0])
                    continue
                tpm_retries += 1
                if tpm_retries <= TPM_MAX_RETRIES:
                    print(f"\n  ⏳ Metadata rate limit (retry {tpm_retries}/{TPM_MAX_RETRIES}) "
                          f"— waiting {TPM_WAIT_SECS}s...")
                    time.sleep(TPM_WAIT_SECS)
                    continue
                print(f"\n  ⚠️  Still rate limited after {TPM_MAX_RETRIES} retries → next key")
                if _rotate_metadata_key():
                    tpm_retries = 0
                    continue
                raise RuntimeError(f"All Gemini metadata keys rate limited: {err[:200]}")
            if any(c in err for c in ("API 500", "API 502", "API 503", "API 504")):
                net_retries += 1
                if net_retries <= 3:
                    time.sleep(5 * net_retries)
                    continue
            raise

        except requests.RequestException as e:
            net_retries += 1
            if net_retries <= 3:
                print(f"\n  ⚠️  Network error, retrying ({net_retries}/3)")
                time.sleep(5 * net_retries)
                continue
            raise RuntimeError(f"Network failure: {type(e).__name__}")


# ── Soft extraction ───────────────────────────────────────────
def extract_metadata(full_text: str, source_file: str) -> dict:
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
    for key, val in result.items():
        print(f"  {key:<18} : {val}")