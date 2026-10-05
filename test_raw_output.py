# test_raw_output.py
# Run from project root: python test_raw_output.py

import os
import json
import time
from groq import Groq
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

groq_client   = Groq(api_key=os.getenv("GROQ_API_KEY_1") or os.getenv("GROQ_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY_1") or os.getenv("GEMINI_API_KEY"))

QWEN_MODEL  = "qwen/qwen3.8-27b"
GEMMA_MODEL = "gemma-4-26b-a4b-it"

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

TEST_TEXT = """
A.K. Gopalan vs The State Of Madras Union Of India on 19 May, 1950
Equivalent citations: 1950 AIR 27, 1950 SCR 88
Bench: Kania, H.J. (CJ), Fazal Ali, Saiyid, Patanjali Sastri, M.,
       Mahajan, Mehr Chand, Das, Sudhi Ranjan, Mukherjea, B.K.
PETITIONER: A.K. GOPALAN
RESPONDENT: THE STATE OF MADRAS. UNION OF INDIA
DATE OF JUDGMENT: 19/05/1950
ACT: Constitution of India - Articles 13, 19, 21, 22, 32;
     Preventive Detention Act, 1950 - Sections 3, 7, 12, 14
HEADNOTE:
The petitioner, a communist leader, was detained under the Preventive
Detention Act, 1950. He challenged the constitutional validity of the
Act contending that it violated Articles 13, 19, 21 and 22 of the
Constitution of India. The Supreme Court dismissed the petition holding
that the Preventive Detention Act was constitutionally valid.
"""

prompt = USER_PROMPT_TEMPLATE.format(text=TEST_TEXT)

# ── Call Qwen ─────────────────────────────────────────────────
print("Calling Qwen...", flush=True)
try:
    qwen_response = groq_client.chat.completions.create(
        model            = QWEN_MODEL,
        messages         = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature      = 0.0,
        max_tokens       = 512,
        reasoning_effort = "none",
    )
    qwen_raw = qwen_response.choices[0].message.content
    qwen_err = None
except Exception as e:
    qwen_raw = None
    qwen_err = str(e)

time.sleep(3)

# ── Call Gemma ────────────────────────────────────────────────
print("Calling Gemma...", flush=True)

gemma_raw = None
gemma_err = None

try:
    gemma_response = gemini_client.models.generate_content(
        model    = GEMMA_MODEL,
        contents = f"{SYSTEM_PROMPT}\n\n{prompt}",
        config   = types.GenerateContentConfig(
            temperature        = 0.0,
            max_output_tokens  = 2048,
            response_mime_type = "application/json",
        )
    )
    
    if gemma_response.text:
        gemma_raw = gemma_response.text
    elif gemma_response.candidates and gemma_response.candidates[0].content.parts:
        gemma_raw = gemma_response.candidates[0].content.parts[0].text
    else:
        gemma_err = f"Empty response or blocked: {gemma_response.candidates}"

except Exception as e:
    gemma_err = f"{type(e).__name__}: {str(e)}"

# Save and print results safely
# print(f"\nGemma raw:\n{gemma_raw or gemma_err}")

# ── Save to file ──────────────────────────────────────────────
output = {
    "test_case"   : "A.K. Gopalan vs State of Madras (1950)",
    "qwen": {
        "model"     : QWEN_MODEL,
        "raw_output": qwen_raw,
        "error"     : qwen_err,
    },
    "gemma": {
        "model"     : GEMMA_MODEL,
        "raw_output": gemma_raw,
        "error"     : gemma_err,
    },
}

with open("test_raw_output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("Saved to test_raw_output.json")
print(f"\nQwen raw:\n{qwen_raw or qwen_err}")
print(f"\nGemma raw:\n{gemma_raw or gemma_err}")