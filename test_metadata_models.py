# test_metadata_models.py
# Temporary comparison script — Qwen vs Gemma 4 for metadata extraction
# Run from project root: python test_metadata_models.py

import os
import re
import json
import time
from groq import Groq
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

groq_client   = Groq(api_key=os.getenv("GROQ_API_KEY_1") or os.getenv("GROQ_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY_1") or os.getenv("GEMINI_API_KEY"))

QWEN_MODEL   = "qwen/qwen3.8-27b"
GEMMA_MODEL  = "gemma-4-26b"   # check and update if needed

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

# ── Test cases ────────────────────────────────────────────────
# Each has a header excerpt and the EXPECTED correct values
TEST_CASES = [
    {
        "name"    : "A.K. Gopalan — Constitution Bench (constitutional challenge)",
        "expected": {
            "bench_type"  : "Constitution Bench",
            "bench_size"  : 6,
            "legal_domain": "Constitutional Law",
            "outcome"     : "Petition dismissed",
        },
        "text": """
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
""",
    },
    {
        "name"    : "Abdulla Ahmed — Full Bench (non-constitutional, 5 judges)",
        "expected": {
            "bench_type"  : "Full Bench",
            "legal_domain": "Contract Law",
            "outcome"     : "Appeal dismissed",
        },
        "text": """
Abdulla Ahmed vs Animendra Kissen Mitter on 14 March, 1950
Equivalent citations: 1950 AIR 15, 1950 SCR 30
Bench: Kania, H.J. (CJ), Fazal Ali, Saiyid, Patanjali Sastri, M.,
       Mahajan, Mehr Chand, Mukherjea, B.K.
PETITIONER: ABDULLA AHMED
RESPONDENT: ANIMENDRA KISSEN MITTER
DATE OF JUDGMENT: 14/03/1950
ACT: Transfer of Property Act; Contract Act
HEADNOTE:
This appeal arises from a dispute over commission claimed by an estate
agent for services rendered in connection with the sale of immovable
property. The respondent had appointed the appellant as agent to find
a purchaser for his property. The Supreme Court dismissed the appeal
holding the agent was not entitled to commission as the sale did not
complete on the terms originally proposed.
""",
    },
    {
        "name"    : "Maneka Gandhi — Constitution Bench (Article 21 passport)",
        "expected": {
            "bench_type"  : "Constitution Bench",
            "legal_domain": "Constitutional Law",
            "outcome"     : "Petition allowed",
        },
        "text": """
Maneka Gandhi vs Union Of India on 25 January, 1978
Equivalent citations: AIR 1978 SC 597, 1978 SCR 621
Bench: Beg, M. Hameedullah (CJ), Chandrachud, Y.V., Bhagwati, P.N.,
       Krishnaiyer, V.R., Untwalia, N.L., Fazalali, Syed Murtaza,
       Kailasam, P.S.
PETITIONER: MANEKA GANDHI
RESPONDENT: UNION OF INDIA
DATE OF JUDGMENT: 25/01/1978
ACT: Constitution of India - Articles 14, 19, 21; Passports Act 1967
HEADNOTE:
The petitioner's passport was impounded by the Government of India
without giving reasons. She challenged this as violating her fundamental
rights under Articles 14, 19 and 21. The Court allowed the petition
holding that Article 21 must be read with Articles 14 and 19 and the
procedure established by law must be just, fair and reasonable.
""",
    },
    {
        "name"    : "A.M. Mair — Division Bench (contract, 2-3 judges)",
        "expected": {
            "bench_type"  : "Division Bench",
            "legal_domain": "Contract Law",
        },
        "text": """
A.M. Mair And Co vs Gordhandass Sagarmull on 30 November, 1950
Equivalent citations: 1950 AIR 355, 1951 SCR 441
Bench: Fazal Ali, Saiyid, Patanjali Sastri, M.
PETITIONER: A.M. MAIR AND CO
RESPONDENT: GORDHANDASS SAGARMULL
DATE OF JUDGMENT: 30/11/1950
ACT: Contract Act; Sale of Goods Act
HEADNOTE:
This case concerns a dispute between jute merchants regarding the
delivery and payment terms of a contract for sale of jute goods.
The appellant contended that the respondent failed to accept delivery.
The court examined the contractual obligations of both parties.
""",
    },
    {
        "name"    : "Criminal appeal — Full Bench (3 judges, IPC)",
        "expected": {
            "bench_type"  : "Full Bench",
            "legal_domain": "Criminal Law",
        },
        "text": """
State of Bihar vs Shiva Bhikshuk Mishra on 15 February, 1970
Equivalent citations: AIR 1970 SC 1033
Bench: Bachawat, R.S., Ramaswami, V., Dua, I.D.
PETITIONER: STATE OF BIHAR
RESPONDENT: SHIVA BHIKSHUK MISHRA
DATE OF JUDGMENT: 15/02/1970
ACT: Indian Penal Code - Sections 302, 34
HEADNOTE:
Appeal by the State against acquittal in a murder case. The respondent
was charged under Section 302 read with Section 34 IPC for the alleged
murder of one Ram Prasad. The trial court had acquitted the respondent
on the grounds of insufficient evidence.
""",
    },
]


# ── JSON parsing (same defensive parser as metadata.py) ───────
def parse_json(text: str) -> dict:
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


# ── Call Qwen via Groq ────────────────────────────────────────
def call_qwen(text: str) -> tuple[dict, str]:
    prompt = USER_PROMPT_TEMPLATE.format(text=text)
    try:
        response = groq_client.chat.completions.create(
            model            = QWEN_MODEL,
            messages         = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature      = 0.0,
            max_tokens       = 512,
            reasoning_effort = "none",
        )
        raw  = response.choices[0].message.content
        data = parse_json(raw)
        return data, raw
    except Exception as e:
        return {}, str(e)


# ── Call Gemma via Gemini client ──────────────────────────────
def call_gemma(text: str) -> tuple[dict, str]:
    prompt = USER_PROMPT_TEMPLATE.format(text=text)
    try:
        response = gemini_client.models.generate_content(
            model    = GEMMA_MODEL,
            contents = prompt,
            config   = types.GenerateContentConfig(
                system_instruction = SYSTEM_PROMPT,
                temperature        = 0.0,
                max_output_tokens  = 512,
            )
        )
        raw  = response.text
        data = parse_json(raw)
        return data, raw
    except Exception as e:
        return {}, str(e)


# ── Scoring ───────────────────────────────────────────────────
def score_result(result: dict, expected: dict) -> tuple[int, int, list[str]]:
    """Returns (correct_fields, total_expected_fields, list_of_mismatches)."""
    correct    = 0
    mismatches = []
    for key, exp_val in expected.items():
        got_val = result.get(key, "MISSING")
        if str(got_val).strip().lower() == str(exp_val).strip().lower():
            correct += 1
        else:
            mismatches.append(f"  {key}: expected '{exp_val}' got '{got_val}'")
    return correct, len(expected), mismatches


def check_json_validity(result: dict, raw: str) -> list[str]:
    """Check for common JSON compliance issues."""
    issues = []
    if not result:
        issues.append("FAILED TO PARSE JSON")
        return issues
    required = ["case_name","citation","year","bench_size","bench_type",
                "legal_domain","key_provisions","outcome","legal_principle","petitioner_type"]
    missing = [k for k in required if k not in result]
    if missing:
        issues.append(f"Missing fields: {missing}")
    for k, v in result.items():
        if isinstance(v, list):
            issues.append(f"Field '{k}' is a list — should be string")
        if isinstance(v, dict):
            issues.append(f"Field '{k}' is a dict — should be string or int")
    if "```" in raw:
        issues.append("Raw response contained markdown fences")
    return issues


# ── Main comparison ───────────────────────────────────────────
def run_comparison():
    print("\n" + "═"*70)
    print("  METADATA MODEL COMPARISON — Qwen 3.8 27B vs Gemma 4")
    print("  " + QWEN_MODEL + " vs " + GEMMA_MODEL)
    print("═"*70)

    qwen_total_correct  = 0
    gemma_total_correct = 0
    qwen_total_fields   = 0
    gemma_total_fields  = 0
    qwen_json_issues    = 0
    gemma_json_issues   = 0

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n{'─'*70}")
        print(f"Test {i}: {tc['name']}")
        print(f"{'─'*70}")

        # Qwen
        print("  Running Qwen...", end=" ", flush=True)
        q_result, q_raw = call_qwen(tc["text"])
        print("done")
        time.sleep(2)   # respect TPM

        # Gemma
        print("  Running Gemma...", end=" ", flush=True)
        g_result, g_raw = call_gemma(tc["text"])
        print("done")
        time.sleep(2)

        # Score
        q_correct, q_total, q_mismatches = score_result(q_result, tc["expected"])
        g_correct, g_total, g_mismatches = score_result(g_result, tc["expected"])

        q_issues = check_json_validity(q_result, q_raw)
        g_issues = check_json_validity(g_result, g_raw)

        qwen_total_correct  += q_correct
        qwen_total_fields   += q_total
        gemma_total_correct += g_correct
        gemma_total_fields  += g_total
        if q_issues: qwen_json_issues  += 1
        if g_issues: gemma_json_issues += 1

        # Print comparison
        print(f"\n  {'Field':<20} {'Qwen':<30} {'Gemma':<30}")
        print(f"  {'─'*20} {'─'*30} {'─'*30}")

        all_fields = ["case_name","citation","year","bench_size","bench_type",
                      "legal_domain","outcome","legal_principle","petitioner_type"]
        for field in all_fields:
            qv = str(q_result.get(field, "MISSING"))[:28]
            gv = str(g_result.get(field, "MISSING"))[:28]
            # Mark expected fields
            exp = tc["expected"].get(field)
            q_mark = "✓" if exp and str(q_result.get(field,"")).strip().lower() == str(exp).strip().lower() else (" " if not exp else "✗")
            g_mark = "✓" if exp and str(g_result.get(field,"")).strip().lower() == str(exp).strip().lower() else (" " if not exp else "✗")
            print(f"  {field:<20} {q_mark} {qv:<28} {g_mark} {gv}")

        print(f"\n  Score: Qwen {q_correct}/{q_total}  |  Gemma {g_correct}/{g_total}")

        if q_mismatches:
            print(f"  Qwen mismatches:")
            for m in q_mismatches: print(f"    ✗{m}")
        if g_mismatches:
            print(f"  Gemma mismatches:")
            for m in g_mismatches: print(f"    ✗{m}")

        if q_issues:
            print(f"  Qwen JSON issues: {q_issues}")
        if g_issues:
            print(f"  Gemma JSON issues: {g_issues}")

    # Overall summary
    print(f"\n{'═'*70}")
    print(f"  SUMMARY")
    print(f"{'─'*70}")
    print(f"  Accuracy    — Qwen: {qwen_total_correct}/{qwen_total_fields}  "
          f"| Gemma: {gemma_total_correct}/{gemma_total_fields}")
    print(f"  JSON issues — Qwen: {qwen_json_issues}/{len(TEST_CASES)} tests  "
          f"| Gemma: {gemma_json_issues}/{len(TEST_CASES)} tests")

    winner = "Qwen" if qwen_total_correct >= gemma_total_correct else "Gemma"
    if qwen_total_correct == gemma_total_correct:
        winner = "Tie"
    print(f"  Verdict     — {winner} wins on accuracy")
    if gemma_json_issues > qwen_json_issues:
        print(f"  ⚠️  Gemma had more JSON compliance issues — prompt hardening needed before switching")
    elif gemma_json_issues < qwen_json_issues:
        print(f"  Gemma had fewer JSON issues — good sign")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    run_comparison()