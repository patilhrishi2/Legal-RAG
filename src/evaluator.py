# src/evaluator.py
# ─────────────────────────────────────────────────────────────
# V9 EVALUATION FRAMEWORK
#
# Three types of evaluation:
#
# 1. RETRIEVAL EVALUATION
#    Given a query and known relevant cases, measures:
#    - Precision@K : of top-K results, what fraction are relevant?
#    - Recall@K    : of all relevant cases, what fraction retrieved?
#    - MRR         : Mean Reciprocal Rank — how high did the first
#                    relevant result appear?
#
# 2. FAITHFULNESS EVALUATION
#    Given a generated answer and source chunks, measures:
#    - Faithfulness score : are all claims in the answer grounded
#                           in the retrieved chunks?
#    - Hallucination flag : did the model fabricate anything?
#    Uses Gemini as judge — an LLM evaluating another LLM's output.
#
# 3. STRATEGY EVALUATION
#    Given a strategy and source chunks, measures:
#    - Grounding score : is each strategic recommendation backed
#                        by a specific retrieved case?
#    - Fabrication flag : did the strategist invent precedents?
#
# WHY LLM-AS-JUDGE?
#   Traditional NLP metrics (BLEU, ROUGE) measure token overlap,
#   not semantic faithfulness. A generated answer that paraphrases
#   a chunk correctly scores low on ROUGE but is perfectly faithful.
#   An LLM judge reads both the answer and the source and reasons
#   about whether the claims are supported — much closer to how a
#   human legal reviewer would evaluate the output.
# ─────────────────────────────────────────────────────────────

import os
import json
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
JUDGE_MODEL   = "gemini-3.5-flash"


# ═══════════════════════════════════════════════════════════════
# PART 1 — RETRIEVAL EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_retrieval(
    retrieved_chunks  : list[dict],
    relevant_cases    : list[str],
    k                 : int = 5,
) -> dict:
    """
    Measure retrieval quality given known relevant cases.

    Parameters:
        retrieved_chunks : list of chunk dicts from retriever.search()
        relevant_cases   : list of source_file strings that are known
                           relevant for this query
                           e.g. ["A_K_Gopalan_vs_...PDF"]
        k                : cutoff for Precision@K and Recall@K

    Returns dict with:
        precision_at_k   : fraction of top-k that are relevant
        recall_at_k      : fraction of relevant cases retrieved in top-k
        mrr              : mean reciprocal rank of first relevant result
        retrieved_cases  : unique cases found in top-k results
        relevant_found   : which relevant cases were retrieved
        relevant_missed  : which relevant cases were NOT retrieved
    """
    if not retrieved_chunks or not relevant_cases:
        return {
            "precision_at_k" : 0.0,
            "recall_at_k"    : 0.0,
            "mrr"            : 0.0,
            "k"              : k,
            "retrieved_cases": [],
            "relevant_found" : [],
            "relevant_missed": relevant_cases,
            "error"          : "No chunks or relevant cases provided",
        }

    top_k_chunks = retrieved_chunks[:k]

    # Get unique source files in top-k
    retrieved_sources = []
    seen = set()
    for chunk in top_k_chunks:
        src = chunk.get("source_file", "")
        if src and src not in seen:
            retrieved_sources.append(src)
            seen.add(src)

    # Precision@K — what fraction of retrieved are relevant?
    relevant_set   = set(relevant_cases)
    relevant_found = [s for s in retrieved_sources if s in relevant_set]
    relevant_missed = [s for s in relevant_cases if s not in set(retrieved_sources)]

    precision_at_k = len(relevant_found) / k if k > 0 else 0.0

    # Recall@K — what fraction of relevant cases did we retrieve?
    recall_at_k = (
        len(relevant_found) / len(relevant_cases)
        if relevant_cases else 0.0
    )

    # MRR — reciprocal rank of first relevant result
    mrr = 0.0
    for rank, chunk in enumerate(top_k_chunks, 1):
        src = chunk.get("source_file", "")
        if src in relevant_set:
            mrr = 1.0 / rank
            break

    return {
        "precision_at_k" : round(precision_at_k, 4),
        "recall_at_k"    : round(recall_at_k,    4),
        "mrr"            : round(mrr,             4),
        "k"              : k,
        "retrieved_cases": retrieved_sources,
        "relevant_found" : relevant_found,
        "relevant_missed": relevant_missed,
        "error"          : None,
    }


# ═══════════════════════════════════════════════════════════════
# PART 2 — FAITHFULNESS EVALUATION (LLM-as-judge)
# ═══════════════════════════════════════════════════════════════

FAITHFULNESS_SYSTEM = """You are an expert legal evaluator assessing whether a generated answer is faithful to its source documents.

Your job: check every factual claim in the generated answer against the provided source chunks.

A claim is FAITHFUL if it is explicitly stated or directly implied by the source chunks.
A claim is HALLUCINATED if it is not present in the source chunks, even if it might be generally true in law.

Return ONLY a valid JSON object. No markdown, no explanation, no code fences.
"""

FAITHFULNESS_PROMPT = """Evaluate the faithfulness of this generated answer against the source chunks.

SOURCE CHUNKS:
{source_text}

GENERATED ANSWER:
{answer}

Return a JSON object with exactly these fields:
{{
  "faithfulness_score": 0.0 to 1.0 (1.0 = fully faithful, 0.0 = fully hallucinated),
  "hallucination_detected": true or false,
  "faithful_claims": ["list of claims that ARE supported by sources"],
  "hallucinated_claims": ["list of claims NOT found in sources"],
  "reasoning": "Brief explanation of the evaluation"
}}
"""

def evaluate_faithfulness(
    generated_answer : str,
    chunks           : list[dict],
) -> dict:
    """
    Use Gemini as a judge to evaluate whether the generated answer
    is faithful to the retrieved source chunks.

    WHY THIS METRIC MATTERS:
    A RAG system can retrieve perfectly relevant chunks but still
    hallucinate in the generated answer — adding facts from training
    data that aren't in the retrieved context. Faithfulness
    specifically measures this: not "is the answer correct?" but
    "is the answer grounded in what was retrieved?"
    """
    if not generated_answer or not chunks:
        return {
            "faithfulness_score"    : 0.0,
            "hallucination_detected": True,
            "faithful_claims"       : [],
            "hallucinated_claims"   : [],
            "reasoning"             : "No answer or chunks provided",
            "error"                 : None,
        }

    # Build source text from chunks — just the text, no metadata noise
    source_parts = []
    for i, chunk in enumerate(chunks[:5], 1):
        source_parts.append(
            f"[Source {i} — {chunk.get('case_name', 'Unknown')} "
            f"({chunk.get('citation', '?')})]\n"
            f"{chunk.get('text', '')[:500]}"
        )
    source_text = "\n\n".join(source_parts)

    prompt = FAITHFULNESS_PROMPT.format(
        source_text = source_text,
        answer      = generated_answer[:2000],  # cap to stay under token limit
    )

    try:
        response = gemini_client.models.generate_content(
            model    = JUDGE_MODEL,
            contents = [
                types.Content(
                    role  = "user",
                    parts = [types.Part(text=prompt)]
                )
            ],
            config = types.GenerateContentConfig(
                system_instruction = FAITHFULNESS_SYSTEM,
                temperature        = 0.0,
                max_output_tokens  = 1000,
            )
        )

        raw  = response.text.strip()
        raw  = re.sub(r"^```json\s*", "", raw)
        raw  = re.sub(r"^```\s*",     "", raw)
        raw  = re.sub(r"\s*```$",     "", raw)
        data = json.loads(raw.strip())

        return {
            "faithfulness_score"    : float(data.get("faithfulness_score",     0.0)),
            "hallucination_detected": bool(data.get("hallucination_detected",  True)),
            "faithful_claims"       : data.get("faithful_claims",              []),
            "hallucinated_claims"   : data.get("hallucinated_claims",          []),
            "reasoning"             : data.get("reasoning",                    ""),
            "error"                 : None,
        }

    except Exception as e:
        return {
            "faithfulness_score"    : 0.0,
            "hallucination_detected": True,
            "faithful_claims"       : [],
            "hallucinated_claims"   : [],
            "reasoning"             : "",
            "error"                 : str(e),
        }


# ═══════════════════════════════════════════════════════════════
# PART 3 — STRATEGY GROUNDING EVALUATION
# ═══════════════════════════════════════════════════════════════

STRATEGY_SYSTEM = """You are an expert legal evaluator assessing whether a legal strategy is grounded in the provided case precedents.

A strategic recommendation is GROUNDED if it is directly supported by a specific case in the source chunks.
A strategic recommendation is FABRICATED if it references cases or legal principles not present in the sources.

Return ONLY a valid JSON object. No markdown, no explanation, no code fences.
"""

STRATEGY_PROMPT = """Evaluate whether this legal strategy is grounded in the provided source cases.

SOURCE CASES:
{source_text}

LEGAL STRATEGY:
Winning arguments: {winning_arguments}
Recommended strategy: {recommended_strategy}
Risk factors: {risk_factors}

Return a JSON object with exactly these fields:
{{
  "grounding_score": 0.0 to 1.0 (1.0 = fully grounded, 0.0 = fully fabricated),
  "fabrication_detected": true or false,
  "grounded_recommendations": ["list of recommendations backed by sources"],
  "ungrounded_recommendations": ["list of recommendations NOT in sources"],
  "reasoning": "Brief explanation"
}}
"""

def evaluate_strategy(
    strategy : dict,
    chunks   : list[dict],
) -> dict:
    """
    Evaluate whether the strategy's recommendations are grounded
    in the retrieved source cases.
    """
    if not strategy or not chunks:
        return {
            "grounding_score"              : 0.0,
            "fabrication_detected"         : True,
            "grounded_recommendations"     : [],
            "ungrounded_recommendations"   : [],
            "reasoning"                    : "No strategy or chunks provided",
            "error"                        : None,
        }

    source_parts = []
    for i, chunk in enumerate(chunks[:5], 1):
        source_parts.append(
            f"[Case {i}: {chunk.get('case_name', 'Unknown')} "
            f"({chunk.get('citation', '?')}, {chunk.get('year', '?')})\n"
            f"Outcome: {chunk.get('outcome', '?')}\n"
            f"Excerpt: {chunk.get('text', '')[:300]}]"
        )
    source_text = "\n\n".join(source_parts)

    prompt = STRATEGY_PROMPT.format(
        source_text          = source_text,
        winning_arguments    = strategy.get("winning_arguments",    "")[:800],
        recommended_strategy = strategy.get("recommended_strategy", "")[:500],
        risk_factors         = strategy.get("risk_factors",         "")[:500],
    )

    try:
        response = gemini_client.models.generate_content(
            model    = JUDGE_MODEL,
            contents = [
                types.Content(
                    role  = "user",
                    parts = [types.Part(text=prompt)]
                )
            ],
            config = types.GenerateContentConfig(
                system_instruction = STRATEGY_SYSTEM,
                temperature        = 0.0,
                max_output_tokens  = 800,
            )
        )

        raw  = response.text.strip()
        raw  = re.sub(r"^```json\s*", "", raw)
        raw  = re.sub(r"^```\s*",     "", raw)
        raw  = re.sub(r"\s*```$",     "", raw)
        data = json.loads(raw.strip())

        return {
            "grounding_score"             : float(data.get("grounding_score",            0.0)),
            "fabrication_detected"        : bool(data.get("fabrication_detected",        True)),
            "grounded_recommendations"    : data.get("grounded_recommendations",         []),
            "ungrounded_recommendations"  : data.get("ungrounded_recommendations",       []),
            "reasoning"                   : data.get("reasoning",                        ""),
            "error"                       : None,
        }

    except Exception as e:
        return {
            "grounding_score"             : 0.0,
            "fabrication_detected"        : True,
            "grounded_recommendations"    : [],
            "ungrounded_recommendations"  : [],
            "reasoning"                   : "",
            "error"                       : str(e),
        }


# ═══════════════════════════════════════════════════════════════
# PART 4 — FULL PIPELINE EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_pipeline(
    query            : str,
    relevant_cases   : list[str],
    chunks           : list[dict],
    generated_answer : str  = None,
    strategy         : dict = None,
    k                : int  = 5,
) -> dict:
    """
    Run all three evaluations and return a unified report.

    Parameters:
        query           : the search query used
        relevant_cases  : ground truth — which source_files are relevant
        chunks          : retrieved chunks (already re-ranked if applicable)
        generated_answer: the answer from generator.py (optional)
        strategy        : the strategy from strategist.py (optional)
        k               : cutoff for retrieval metrics

    Returns a unified evaluation report with all three scores.
    """
    report = {
        "query"    : query,
        "k"        : k,
        "retrieval": None,
        "faithfulness": None,
        "strategy_grounding": None,
        "summary"  : {},
    }

    # 1. Retrieval evaluation (always runs)
    report["retrieval"] = evaluate_retrieval(chunks, relevant_cases, k)

    # 2. Faithfulness evaluation (only if answer provided)
    if generated_answer:
        report["faithfulness"] = evaluate_faithfulness(generated_answer, chunks)

    # 3. Strategy grounding (only if strategy provided)
    if strategy:
        report["strategy_grounding"] = evaluate_strategy(strategy, chunks)

    # Build summary
    ret = report["retrieval"]
    fai = report["faithfulness"]
    str_ = report["strategy_grounding"]

    report["summary"] = {
        "precision_at_k"     : ret["precision_at_k"],
        "recall_at_k"        : ret["recall_at_k"],
        "mrr"                : ret["mrr"],
        "faithfulness_score" : fai["faithfulness_score"] if fai else None,
        "hallucination"      : fai["hallucination_detected"] if fai else None,
        "strategy_grounding" : str_["grounding_score"] if str_ else None,
        "fabrication"        : str_["fabrication_detected"] if str_ else None,
        "overall_health"     : _overall_health(ret, fai, str_),
    }

    return report


def _overall_health(ret: dict, fai: dict, str_: dict) -> str:
    """
    Simple overall system health signal based on all three scores.
    GREEN  = retrieval good + no hallucination + no fabrication
    YELLOW = retrieval ok OR minor issues
    RED    = poor retrieval OR hallucination detected OR fabrication
    """
    issues = []

    if ret["precision_at_k"] < 0.4:
        issues.append("low retrieval precision")
    if ret["recall_at_k"] < 0.5:
        issues.append("low retrieval recall")
    if fai and fai["hallucination_detected"]:
        issues.append("hallucination detected")
    if fai and fai["faithfulness_score"] < 0.7:
        issues.append("low faithfulness")
    if str_ and str_["fabrication_detected"]:
        issues.append("strategy fabrication detected")

    if not issues:
        return "GREEN"
    if len(issues) <= 1:
        return f"YELLOW — {issues[0]}"
    return f"RED — {', '.join(issues)}"


# ═══════════════════════════════════════════════════════════════
# TEST SUITE — built-in ground truth for your current index
# ═══════════════════════════════════════════════════════════════

# Ground truth: for each query, which source_files are relevant?
# These are hand-labelled — the correct answer for your current index.
TEST_CASES = [
    {
        "query"          : "What are the constitutional limits on preventive detention?",
        "relevant_cases" : [
            "A_K_Gopalan_vs_The_State_Of_Madras_Union_Of_India__on_19_May_1950_1.PDF",
            "Maneka_Gandhi_vs_Union_Of_India_on_25_January_1978_1.PDF",
        ],
        "description"    : "Should retrieve both Gopalan and Maneka Gandhi",
    },
    {
        "query"          : "Does Article 21 apply to preventive detention?",
        "relevant_cases" : [
            "A_K_Gopalan_vs_The_State_Of_Madras_Union_Of_India__on_19_May_1950_1.PDF",
            "Maneka_Gandhi_vs_Union_Of_India_on_25_January_1978_1.PDF",
        ],
        "description"    : "Classic Gopalan/Maneka Gandhi contradiction query",
    },
    {
        "query"          : "commission agent contract property sale",
        "relevant_cases" : [
            "Abdulla_Ahmed_vs_Animendra_Kissen_Mitter_on_14_March_1950_1.PDF",
        ],
        "description"    : "Should retrieve Abdulla Ahmed — contract law",
    },
    {
        "query"          : "jute merchant sale delivery obligation",
        "relevant_cases" : [
            "A_M_Mair_Co_vs_Gordhandass_Sagarmull_on_30_November_1950_1.PDF",
        ],
        "description"    : "Should retrieve A.M. Mair — jute merchant case",
    },
    {
        "query"          : "procedure established by law personal liberty Article 21",
        "relevant_cases" : [
            "A_K_Gopalan_vs_The_State_Of_Madras_Union_Of_India__on_19_May_1950_1.PDF",
            "Maneka_Gandhi_vs_Union_Of_India_on_25_January_1978_1.PDF",
        ],
        "description"    : "Maneka Gandhi is the landmark case on this — should rank highly",
    },
]


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from retriever import search
    from generator import generate_answer
    from strategist import generate_strategy

    print("\n🔬 Running V9 Evaluation Suite")
    print(f"   {len(TEST_CASES)} test cases\n")

    all_precision = []
    all_recall    = []
    all_mrr       = []
    all_faith     = []

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"{'─'*65}")
        print(f"Test {i}: {tc['description']}")
        print(f"Query  : {tc['query']}")

        # Retrieve with re-ranking
        chunks = search(tc["query"], top_k=5, rerank=True)

        # Generate answer for faithfulness check (only first 2 tests
        # to keep runtime reasonable)
        answer   = None
        strategy = None
        if i <= 2:
            gen    = generate_answer(tc["query"], chunks)
            answer = gen["answer"]
            strat  = generate_strategy(tc["query"], chunks,
                                       generated_answer=answer)
            strategy = strat

        # Run evaluation
        report = evaluate_pipeline(
            query            = tc["query"],
            relevant_cases   = tc["relevant_cases"],
            chunks           = chunks,
            generated_answer = answer,
            strategy         = strategy,
            k                = 5,
        )

        ret = report["retrieval"]
        print(f"\n  Retrieval:")
        print(f"    Precision@5 : {ret['precision_at_k']}")
        print(f"    Recall@5    : {ret['recall_at_k']}")
        print(f"    MRR         : {ret['mrr']}")
        print(f"    Found       : {ret['relevant_found']}")
        print(f"    Missed      : {ret['relevant_missed']}")

        if report["faithfulness"]:
            fai = report["faithfulness"]
            print(f"\n  Faithfulness:")
            print(f"    Score       : {fai['faithfulness_score']}")
            print(f"    Hallucinated: {fai['hallucination_detected']}")
            print(f"    Reasoning   : {fai['reasoning'][:200]}")
            all_faith.append(fai["faithfulness_score"])

        if report["strategy_grounding"]:
            sg = report["strategy_grounding"]
            print(f"\n  Strategy grounding:")
            print(f"    Score       : {sg['grounding_score']}")
            print(f"    Fabrication : {sg['fabrication_detected']}")

        print(f"\n  Overall health: {report['summary']['overall_health']}")

        all_precision.append(ret["precision_at_k"])
        all_recall.append(ret["recall_at_k"])
        all_mrr.append(ret["mrr"])

    # Aggregate metrics
    print(f"\n{'═'*65}")
    print("AGGREGATE METRICS")
    print(f"{'─'*65}")
    print(f"  Mean Precision@5  : {sum(all_precision)/len(all_precision):.4f}")
    print(f"  Mean Recall@5     : {sum(all_recall)/len(all_recall):.4f}")
    print(f"  Mean MRR          : {sum(all_mrr)/len(all_mrr):.4f}")
    if all_faith:
        print(f"  Mean Faithfulness : {sum(all_faith)/len(all_faith):.4f}")
    print(f"\n  Test cases run    : {len(TEST_CASES)}")
    print(f"  With faithfulness : {len(all_faith)}")