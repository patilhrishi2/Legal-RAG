# src/metadata.py
# ─────────────────────────────────────────────────────────────
# METADATA EXTRACTION
# Sends the first ~3000 words of a legal judgment to
# Groq (Llama 3.3 70B) and gets back structured metadata as JSON.
#
# Called once per document during ingestion — NOT per chunk.
# The same metadata dict is attached to every chunk of that document.
# ─────────────────────────────────────────────────────────────

import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"

# How many words from the start of the document to send.
# The header, bench details, citations, and opening summary
# of every SC judgment appear in the first 2000–3000 words.
# No need to send the full document — saves tokens and is faster.
HEADER_WORD_COUNT = 3000


SYSTEM_PROMPT = """You are a legal metadata extractor specializing in Supreme Court of India judgments.

Your job is to extract structured metadata from the provided text and return it as a single valid JSON object.

Rules:
- Return ONLY the JSON object. No explanation, no markdown, no code fences.
- If a field cannot be determined from the text, use "Unknown" for strings and 0 for integers.
- All values must be strings or integers — no lists, no nested objects, no arrays.
- For key_provisions, join multiple items with ", " (comma space).
- For legal_domain, pick the single most relevant domain.
- Keep all string values concise (under 200 characters).
"""

USER_PROMPT_TEMPLATE = """Extract metadata from this Supreme Court of India judgment header.

Return a JSON object with exactly these fields:
{{
  "case_name"       : "Full case name e.g. A.K. Gopalan vs State of Madras",
  "citation"        : "Primary citation e.g. AIR 1950 SC 27 or 1950 SCR 88",
  "year"            : 1950,
  "bench_size"      : 5,
  "bench_type"      : "Constitution Bench or Division Bench or Full Bench or Single Bench",
  "legal_domain"    : "e.g. Constitutional Law, Criminal Law, Contract Law, Property Law, Family Law, Tax Law, Labour Law, Administrative Law",
  "key_provisions"  : "Comma-separated statutes and articles e.g. Article 21, Article 22, IPC Section 302",
  "outcome"         : "Petition dismissed or Appeal allowed or Petition allowed or Appeal dismissed or Partially allowed",
  "legal_principle" : "The core legal principle established or applied in one sentence",
  "petitioner_type" : "Individual or State or Company or Government or NGO or Unknown"
}}

Judgment text:
{text}
"""


def extract_header(full_text: str) -> str:
    """
    Take only the first HEADER_WORD_COUNT words of the document.
    This contains everything needed for metadata extraction.
    Sending the full 95,000-word judgment would waste tokens and hit context limits.
    """
    words = full_text.split()
    header_words = words[:HEADER_WORD_COUNT]
    return " ".join(header_words)


def parse_json_from_response(text: str) -> dict:
    """
    Safely parse JSON from the LLM response.

    Even with clear instructions, LLMs occasionally wrap JSON in
    markdown fences (```json ... ```) or add a preamble sentence.
    This function handles those cases gracefully.
    """
    # Strip markdown code fences if present
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*",     "", text)
    text = re.sub(r"\s*```$",     "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # If still failing, try to find a JSON object anywhere in the response
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # If all parsing fails, return a safe fallback
    # Ingestion continues — better to have a chunk with Unknown metadata than to crash
    print("  ⚠️  Could not parse metadata JSON — using fallback values")
    return {}


def get_fallback_metadata(source_file: str) -> dict:
    """
    Fallback metadata when extraction fails or returns empty.
    Ensures every chunk always has a complete metadata dict.
    ChromaDB requires consistent keys across all documents in a collection.
    """
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
    """
    Enforce correct types and fill missing fields.

    ChromaDB will reject the entire .add() call if any metadata value
    is the wrong type (e.g. year as a string instead of int).
    This function makes sure that never happens.
    """
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
    """
    Main function — call this from ingest.py.

    Takes the full document text and filename.
    Returns a clean, validated metadata dict ready for ChromaDB.
    """
    header = extract_header(full_text)

    prompt = USER_PROMPT_TEMPLATE.format(text=header)

    try:
        response = groq_client.chat.completions.create(
            model       = MODEL,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            temperature = 0.0,    # deterministic — we want consistent JSON, not creativity
            max_tokens  = 512,    # metadata JSON is small; no need for more
        )

        raw_text = response.choices[0].message.content
        raw_dict = parse_json_from_response(raw_text)

        if not raw_dict:
            return get_fallback_metadata(source_file)

        cleaned = validate_and_clean(raw_dict, source_file)

        return cleaned

    except Exception as e:
        print(f"  ⚠️  Metadata extraction failed: {e}")
        return get_fallback_metadata(source_file)


# ── Quick test — run this file directly ──────────────────────
if __name__ == "__main__":
    sample = """
    A.K. Gopalan vs The State Of Madras Union Of India on 19 May, 1950
    Equivalent citations: 1950 AIR 27, 1950 SCR 88
    Author: Kania
    Bench: Kania, H.J. (CJ), Fazal Ali, Saiyid, Patanjali Sastri, M.,
           Mahajan, Mehr Chand, Das, Sudhi Ranjan, Mukherjea, B.K.
    PETITIONER: A.K. GOPALAN
    RESPONDENT: THE STATE OF MADRAS. UNION OF INDIA (Intervener)
    DATE OF JUDGMENT: 19/05/1950
    ACT: Constitution of India - Articles 13, 19, 21, 22, 32;
         Preventive Detention Act, 1950 - Sections 3, 7, 12, 14
    HEADNOTE:
    The petitioner, a communist leader, was detained under the Preventive
    Detention Act, 1950. He challenged the constitutional validity of the
    Act contending that it violated Articles 13, 19, 21 and 22 of the
    Constitution of India...
    """

    print("Testing metadata extraction...\n")
    result = extract_metadata(sample, "A_K_Gopalan_test.PDF")

    print("Extracted metadata:")
    for key, val in result.items():
        print(f"  {key:<18} : {val}")