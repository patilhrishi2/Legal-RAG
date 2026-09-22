# src/strategist.py
# ─────────────────────────────────────────────────────────────
# V8 LEGAL STRATEGY INTELLIGENCE
#
# What this file does:
#   Takes everything already produced by V1-V7 pipeline stages
#   (retrieved chunks, extraction results, contradiction report,
#   generated answer) and synthesises a strategic assessment:
#
#     - Winning arguments historically used in similar cases
#     - Failure patterns — what arguments courts rejected
#     - Key decisive factors the court focused on
#     - Strength assessment of the user's position
#     - Recommended primary and fallback arguments
#     - Risk factors to anticipate
#
# WHY THIS IS DIFFERENT FROM GENERATION (V4):
#   Generator answers "what has the court held?"
#   Strategist answers "given my situation, what should I argue?"
#   Generator is descriptive. Strategist is prescriptive.
#
# WHY GEMINI RATHER THAN GROQ HERE?
#   Strategy synthesis requires longer, more nuanced reasoning
#   than structured JSON extraction. Gemini 3.5 Flash handles
#   complex multi-document synthesis better than Qwen for this
#   open-ended task. Groq is used for structured extraction
#   (metadata, arguments, contradictions) where JSON precision
#   and speed matter. Gemini is used for synthesis tasks where
#   reasoning quality matters more.
# ─────────────────────────────────────────────────────────────

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

GENERATION_MODEL = "gemini-3.5-flash"

SYSTEM_PROMPT = """You are a senior legal strategist specialising in Supreme Court of India jurisprudence.

Your job is to analyse retrieved case precedents and produce a practical legal strategy assessment for the user's situation.

STRICT RULES:
1. Base your strategy ONLY on the provided case excerpts and analysis. Do not use your own legal knowledge.
2. Every strategic recommendation must be grounded in specific cases with citations.
3. Distinguish clearly between arguments that SUCCEEDED in court and arguments that FAILED.
4. If the precedents are unfavourable to the user's position, say so honestly — do not manufacture optimism.
5. If contradictions exist in the precedents, factor them into your risk assessment explicitly.
6. Be specific and actionable — avoid generic legal commentary.

Your output must follow this exact structure:

SITUATION SUMMARY:
[Brief restatement of the user's legal situation in one or two sentences]

STRENGTH ASSESSMENT:
[One of: STRONG / MODERATE / WEAK — with a one-line reason]

WINNING ARGUMENTS:
[Arguments that have succeeded in similar cases, each with the case citation that supports it]

FAILURE PATTERNS:
[Arguments courts have rejected in similar cases — what NOT to lead with]

KEY DECISIVE FACTORS:
[The specific factors courts focused on when deciding similar cases]

RECOMMENDED STRATEGY:
[Primary argument to lead with, and a fallback argument if the primary fails]

RISK FACTORS:
[What could weaken the user's position — unfavourable precedents, contradictions, missing facts]

CONFIDENCE:
[HIGH / MEDIUM / LOW — based on how directly the retrieved cases match the situation]
"""


def build_strategy_context(
    query          : str,
    chunks         : list[dict],
    contradiction_report : dict = None,
    generated_answer     : str  = None,
) -> str:
    """
    Assemble everything V1-V7 produced into a single context block
    for the strategist. Unlike generator.py which focuses on what
    courts held, this context emphasises outcomes and patterns.

    We include:
      - The user's situation (query)
      - Each case's outcome and legal principle (from metadata)
      - Key excerpts showing court reasoning
      - Contradiction report if present
      - Generated answer if present (as a starting point)
    """
    lines = []
    lines.append(f"USER'S LEGAL SITUATION:\n{query}\n")

    # Group chunks by case for cleaner presentation
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
                "bench_type"     : chunk.get("bench_type",      "Unknown"),
                "bench_size"     : chunk.get("bench_size",      0),
                "excerpts"       : [],
                "extraction"     : {},
            }
        # Collect extraction if available
        if "extraction" in chunk and chunk["extraction"]:
            ext = chunk["extraction"]
            case = by_case[src]
            for field in ["court_reasoning", "legal_principles",
                          "decision", "petitioner_arguments", "respondent_arguments"]:
                val = ext.get(field, "").strip()
                if val and field not in case["extraction"]:
                    case["extraction"][field] = val

        text = chunk.get("text", "").strip()[:250]
        if text and text not in by_case[src]["excerpts"]:
            by_case[src]["excerpts"].append(text)

    lines.append("RELEVANT PRECEDENTS:\n")
    for i, (src, case) in enumerate(by_case.items(), 1):
        lines.append(f"Case {i}: {case['case_name']}")
        lines.append(f"  Citation   : {case['citation']} ({case['year']})")
        lines.append(f"  Bench      : {case['bench_type']} ({case['bench_size']} judges)")
        lines.append(f"  Outcome    : {case['outcome']}")
        lines.append(f"  Principle  : {case['legal_principle']}")

        ext = case["extraction"]
        if ext.get("court_reasoning"):
            lines.append(f"  Court reasoning: {ext['court_reasoning'][:300]}")
        if ext.get("decision"):
            lines.append(f"  Decision   : {ext['decision'][:200]}")
        if ext.get("petitioner_arguments"):
            lines.append(f"  Petitioner argued: {ext['petitioner_arguments'][:200]}")

        if case["excerpts"]:
            lines.append(f"  Key excerpt: {case['excerpts'][0]}...")
        lines.append("")

    # Add contradiction report if present
    if contradiction_report and contradiction_report.get("contradictions_found"):
        lines.append("CONTRADICTIONS IN PRECEDENTS:")
        for c in contradiction_report.get("contradictions", []):
            lines.append(
                f"  - On '{c.get('legal_question', '')}': "
                f"{c.get('case_1', '')} held '{c.get('case_1_position', '')}' "
                f"BUT {c.get('case_2', '')} held '{c.get('case_2_position', '')}'. "
                f"Overruled: {c.get('overruled', False)}."
            )
        lines.append("")

    # Add generated answer as context if available
    if generated_answer and generated_answer.strip():
        lines.append("PRIOR ANALYSIS OF PRECEDENTS:")
        lines.append(generated_answer[:800])
        lines.append("")

    return "\n".join(lines)


def build_strategy_prompt(context: str) -> str:
    return (
        f"Analyse the following legal situation and precedents, "
        f"then produce a strategic assessment.\n\n"
        f"{context}\n\n"
        f"Produce the strategic assessment now, following the exact "
        f"structure specified. Be specific, cite cases, and be honest "
        f"about weaknesses."
    )


def parse_strategy_response(raw_text: str) -> dict:
    """
    Parse the structured strategy response into a dict.
    Same defensive section-parsing as generator.py.
    """
    sections = {
        "situation_summary"   : "",
        "strength_assessment" : "",
        "winning_arguments"   : "",
        "failure_patterns"    : "",
        "key_decisive_factors": "",
        "recommended_strategy": "",
        "risk_factors"        : "",
        "confidence"          : "UNKNOWN",
        "full_response"       : raw_text,
    }

    section_map = {
        "SITUATION SUMMARY"   : "situation_summary",
        "STRENGTH ASSESSMENT" : "strength_assessment",
        "WINNING ARGUMENTS"   : "winning_arguments",
        "FAILURE PATTERNS"    : "failure_patterns",
        "KEY DECISIVE FACTORS": "key_decisive_factors",
        "RECOMMENDED STRATEGY": "recommended_strategy",
        "RISK FACTORS"        : "risk_factors",
        "CONFIDENCE"          : "confidence",
    }

    current_section = None
    buffer          = []

    for line in raw_text.split("\n"):
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

    # Clean up confidence
    conf_raw = sections["confidence"].upper()
    for level in ["HIGH", "MEDIUM", "LOW"]:
        if level in conf_raw:
            sections["confidence"] = level
            break

    # Fallback strength assessment
    if not sections["strength_assessment"]:
        sections["strength_assessment"] = "MODERATE"

    return sections


def generate_strategy(
    query                : str,
    chunks               : list[dict],
    contradiction_report : dict = None,
    generated_answer     : str  = None,
) -> dict:
    """
    Main function — call this from main.py.

    Takes the user's query and everything V1-V7 produced.
    Returns a structured strategic assessment.

    Parameters:
        query                : user's legal situation / question
        chunks               : retrieved (and optionally re-ranked,
                               extracted) chunks from retriever
        contradiction_report : output of detect_contradictions() or None
        generated_answer     : output of generate_answer()["answer"] or None
    """
    if not chunks:
        return {
            "situation_summary"   : query,
            "strength_assessment" : "UNKNOWN",
            "winning_arguments"   : "",
            "failure_patterns"    : "",
            "key_decisive_factors": "",
            "recommended_strategy": "No relevant cases retrieved.",
            "risk_factors"        : "",
            "confidence"          : "LOW",
            "full_response"       : "",
            "error"               : None,
        }

    context = build_strategy_context(
        query, chunks, contradiction_report, generated_answer
    )
    prompt  = build_strategy_prompt(context)

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
                temperature        = 0.2,   # slightly higher than generator —
                                            # strategy synthesis benefits from
                                            # a little more reasoning flexibility
                max_output_tokens  = 5000,
            )
        )

        raw_text = response.text
        parsed   = parse_strategy_response(raw_text)

        return {
            "situation_summary"   : parsed["situation_summary"],
            "strength_assessment" : parsed["strength_assessment"],
            "winning_arguments"   : parsed["winning_arguments"],
            "failure_patterns"    : parsed["failure_patterns"],
            "key_decisive_factors": parsed["key_decisive_factors"],
            "recommended_strategy": parsed["recommended_strategy"],
            "risk_factors"        : parsed["risk_factors"],
            "confidence"          : parsed["confidence"],
            "full_response"       : parsed["full_response"],
            "error"               : None,
        }

    except Exception as e:
        return {
            "situation_summary"   : "",
            "strength_assessment" : "UNKNOWN",
            "winning_arguments"   : "",
            "failure_patterns"    : "",
            "key_decisive_factors": "",
            "recommended_strategy": "",
            "risk_factors"        : "",
            "confidence"          : "LOW",
            "full_response"       : "",
            "error"               : str(e),
        }


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from retriever      import search
    from extractor      import extract_chunks
    from contradiction  import detect_contradictions
    from generator      import generate_answer

    situations = [
        "My client was detained under the Preventive Detention Act "
        "without being informed of the grounds for detention. "
        "The detention has lasted four months without advisory board review. "
        "What is the strongest constitutional challenge?",

        "A merchant entered a contract for sale of goods but the buyer "
        "refused to complete the purchase after the seller had already "
        "arranged delivery. What remedies are available?",
    ]

    for situation in situations:
        print(f"\n{'═'*65}")
        print(f"SITUATION:\n{situation}")
        print(f"{'─'*65}")

        chunks       = search(situation, top_k=5, rerank=True)
        enriched     = extract_chunks(chunks)
        contradicts  = detect_contradictions(situation, enriched)
        gen_result   = generate_answer(situation, enriched,
                                       use_extraction=True,
                                       contradiction_report=contradicts)
        strategy     = generate_strategy(
            situation,
            enriched,
            contradiction_report = contradicts,
            generated_answer     = gen_result["answer"],
        )

        if strategy["error"]:
            print(f"ERROR: {strategy['error']}")
            continue

        print(f"\nSTRENGTH: {strategy['strength_assessment']}")
        print(f"\nSITUATION SUMMARY:\n{strategy['situation_summary']}")
        print(f"\nWINNING ARGUMENTS:\n{strategy['winning_arguments']}")
        print(f"\nFAILURE PATTERNS:\n{strategy['failure_patterns']}")
        print(f"\nKEY DECISIVE FACTORS:\n{strategy['key_decisive_factors']}")
        print(f"\nRECOMMENDED STRATEGY:\n{strategy['recommended_strategy']}")
        print(f"\nRISK FACTORS:\n{strategy['risk_factors']}")
        print(f"\nCONFIDENCE: {strategy['confidence']}")