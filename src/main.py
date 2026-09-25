# src/main.py
# V10 — adds request logging, query history, health monitoring

import os
import sys
import time
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from retriever    import search, list_cases
from ingest       import run_ingestion, get_collection
from generator    import generate_answer
from extractor    import extract_chunks
from contradiction import detect_contradictions
from strategist   import generate_strategy
from evaluator    import evaluate_pipeline, TEST_CASES
from database     import init_db, log_request, get_history, get_stats

app = Flask(__name__)

# Initialise database on startup
init_db()


# ── GET /health ───────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """
    Comprehensive health check — checks all system components
    and returns usage statistics.

    More detailed than /status:
      /status : is ChromaDB reachable + chunk count
      /health : all components + BM25 index + DB stats + usage
    """
    health_report = {
        "version"   : "V10",
        "timestamp" : __import__("datetime").datetime.utcnow().isoformat(),
        "components": {},
        "statistics": {},
        "overall"   : "healthy",
    }

    issues = []

    # Check ChromaDB
    try:
        collection = get_collection()
        health_report["components"]["chromadb"] = {
            "status"      : "ok",
            "total_chunks": collection.count(),
        }
    except Exception as e:
        health_report["components"]["chromadb"] = {
            "status": "error",
            "error" : str(e),
        }
        issues.append("chromadb unreachable")

    # Check BM25 index
    bm25_path = "storage/bm25_index.pkl"
    if os.path.exists(bm25_path):
        size_kb = os.path.getsize(bm25_path) // 1024
        health_report["components"]["bm25_index"] = {
            "status" : "ok",
            "size_kb": size_kb,
        }
    else:
        health_report["components"]["bm25_index"] = {
            "status": "missing",
            "error" : "Run python src/ingest.py to rebuild",
        }
        issues.append("BM25 index missing")

    # Check log database
    db_path = "storage/logs.db"
    if os.path.exists(db_path):
        health_report["components"]["log_database"] = {
            "status" : "ok",
            "size_kb": os.path.getsize(db_path) // 1024,
        }
    else:
        health_report["components"]["log_database"] = {
            "status": "not initialised",
        }

    # Check re-ranker model cache
    cache_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2"
    )
    health_report["components"]["reranker_model"] = {
        "status": "cached" if os.path.exists(cache_dir) else "will download on first use",
    }

    # Usage statistics from database
    try:
        stats = get_stats()
        health_report["statistics"] = stats
    except Exception as e:
        health_report["statistics"] = {"error": str(e)}

    # List indexed cases
    try:
        cases = list_cases()
        health_report["indexed_cases"] = [
            {
                "case_name"   : c["case_name"],
                "citation"    : c["citation"],
                "year"        : c["year"],
                "legal_domain": c["legal_domain"],
            }
            for c in cases
        ]
    except Exception as e:
        health_report["indexed_cases"] = []

    if issues:
        health_report["overall"] = f"degraded — {', '.join(issues)}"

    return jsonify(health_report)


# ── GET /status ───────────────────────────────────────────────
@app.route("/status", methods=["GET"])
def status():
    try:
        collection = get_collection()
        return jsonify({
            "status"      : "ok",
            "total_chunks": collection.count(),
            "collection"  : "legal_cases",
            "version"     : "V10",
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── GET /history ──────────────────────────────────────────────
@app.route("/history", methods=["GET"])
def history():
    """
    Return paginated query history from the log database.

    Query params:
      ?limit=20          max results (default 20, max 100)
      ?offset=0          pagination offset
      ?query=detention   filter by query text (contains)
      ?errors_only=true  show only failed requests
    """
    try:
        limit        = int(request.args.get("limit",       20))
        offset       = int(request.args.get("offset",       0))
        query_filter = request.args.get("query",        None)
        errors_only  = request.args.get("errors_only", "false").lower() == "true"

        result = get_history(
            limit        = limit,
            offset       = offset,
            query_filter = query_filter,
            errors_only  = errors_only,
        )
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── GET /cases ────────────────────────────────────────────────
@app.route("/cases", methods=["GET"])
def cases():
    try:
        all_cases  = list_cases()
        domain     = request.args.get("legal_domain")
        year_from  = request.args.get("year_from",  type=int)
        year_to    = request.args.get("year_to",    type=int)
        bench_type = request.args.get("bench_type")

        if domain:
            all_cases = [c for c in all_cases if c["legal_domain"] == domain]
        if year_from:
            all_cases = [c for c in all_cases if c["year"] >= year_from]
        if year_to:
            all_cases = [c for c in all_cases if c["year"] <= year_to]
        if bench_type:
            all_cases = [c for c in all_cases if c["bench_type"] == bench_type]

        return jsonify({"count": len(all_cases), "cases": all_cases})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── POST /search ──────────────────────────────────────────────
@app.route("/search", methods=["POST"])
def search_cases():
    """
    Full search pipeline with logging.
    Every request — success or failure — is logged to SQLite.

    Request body:
    {
        "query"                : "...",
        "top_k"                : 5,
        "filters"              : {},
        "rerank"               : true,
        "extract"              : true,
        "detect_contradictions": true,
        "generate"             : true,
        "strategize"           : true
    }
    """
    start_time = time.time()   # start timing for response_time_ms

    data = request.get_json()

    if not data or "query" not in data:
        return jsonify({"error": "Request body must include a 'query' field"}), 400

    query         = data["query"].strip()
    top_k         = int(data.get("top_k",    5))
    filters       = data.get("filters",      {})
    rerank        = data.get("rerank",       False)
    extract       = data.get("extract",      False)
    do_contradict = data.get("detect_contradictions", False)
    generate      = data.get("generate",     False)
    strategize    = data.get("strategize",   False)

    if not query:
        return jsonify({"error": "'query' cannot be empty"}), 400
    if not 1 <= top_k <= 20:
        return jsonify({"error": "'top_k' must be between 1 and 20"}), 400

    retrieved_cases = []
    had_error       = False
    error_message   = ""

    try:
        # ── Step 1: Retrieval ─────────────────────────────────
        chunks = search(query, top_k=top_k, filters=filters, rerank=rerank)

        # Capture retrieved case names for logging
        seen = set()
        for c in chunks:
            src = c.get("source_file", "")
            if src and src not in seen:
                retrieved_cases.append(src)
                seen.add(src)

        # ── Step 2: Extraction ────────────────────────────────
        if extract:
            chunks = extract_chunks(chunks)

        # ── Step 3: Contradiction detection ───────────────────
        contradiction_result = None
        if do_contradict:
            try:
                contradiction_result = detect_contradictions(query, chunks)
            except Exception as e:
                contradiction_result = {
                    "contradictions_found": False,
                    "contradiction_count" : 0,
                    "contradictions"      : [],
                    "analysis_note"       : "",
                    "error"               : str(e),
                }

        # ── Step 4: Format chunks ─────────────────────────────
        formatted_chunks = []
        for i, r in enumerate(chunks, 1):
            chunk_entry = {
                "rank"           : i,
                "score"          : r["score"],
                "rrf_score"      : r.get("rrf_score"),
                "rerank_score"   : r.get("rerank_score"),
                "rerank_rank"    : r.get("rerank_rank"),
                "semantic_score" : r.get("semantic_score"),
                "bm25_score"     : r.get("bm25_score"),
                "in_both"        : r.get("in_both"),
                "chunk_num"      : r["chunk_num"],
                "text"           : r["text"],
                "case_name"      : r["case_name"],
                "citation"       : r["citation"],
                "year"           : r["year"],
                "bench_type"     : r["bench_type"],
                "bench_size"     : r["bench_size"],
                "legal_domain"   : r["legal_domain"],
                "key_provisions" : r["key_provisions"],
                "outcome"        : r["outcome"],
                "legal_principle": r["legal_principle"],
                "petitioner_type": r["petitioner_type"],
                "source_file"    : r["source_file"],
            }
            if extract and "extraction" in r:
                chunk_entry["extraction"] = r["extraction"]
            formatted_chunks.append(chunk_entry)

        # ── Step 5: Build response ────────────────────────────
        response = {
            "query"                  : query,
            "filters_applied"        : filters,
            "rerank_used"            : rerank,
            "extract_used"           : extract,
            "contradiction_detection": contradiction_result,
            "count"                  : len(formatted_chunks),
            "results"                : formatted_chunks,
            "generated"              : None,
            "strategy"               : None,
        }

        # ── Step 6: Generation ────────────────────────────────
        gen_result = None
        if generate:
            try:
                gen_result = generate_answer(
                    query,
                    chunks,
                    use_extraction       = extract,
                    contradiction_report = contradiction_result
                )
                response["generated"] = {
                    "answer"              : gen_result["answer"],
                    "contradictions"      : gen_result.get("contradictions", ""),
                    "key_legal_principles": gen_result["key_legal_principles"],
                    "sources_used"        : gen_result["sources_used"],
                    "confidence"          : gen_result["confidence"],
                    "sources"             : gen_result["sources"],
                    "extraction_used"     : gen_result.get("extraction_used", False),
                    "error"               : gen_result["error"],
                }
            except Exception as e:
                response["generated"] = {"answer": "", "error": str(e)}

        # ── Step 7: Strategy ──────────────────────────────────
        if strategize:
            try:
                strategy_result = generate_strategy(
                    query,
                    chunks,
                    contradiction_report = contradiction_result,
                    generated_answer     = gen_result["answer"]
                                           if gen_result else None,
                )
                response["strategy"] = {
                    "situation_summary"   : strategy_result["situation_summary"],
                    "strength_assessment" : strategy_result["strength_assessment"],
                    "winning_arguments"   : strategy_result["winning_arguments"],
                    "failure_patterns"    : strategy_result["failure_patterns"],
                    "key_decisive_factors": strategy_result["key_decisive_factors"],
                    "recommended_strategy": strategy_result["recommended_strategy"],
                    "risk_factors"        : strategy_result["risk_factors"],
                    "confidence"          : strategy_result["confidence"],
                    "error"               : strategy_result["error"],
                }
            except Exception as e:
                response["strategy"] = {
                    "situation_summary": "", "strength_assessment": "UNKNOWN",
                    "winning_arguments": "", "failure_patterns": "",
                    "key_decisive_factors": "", "recommended_strategy": "",
                    "risk_factors": "", "confidence": "LOW", "error": str(e),
                }

    except Exception as e:
        had_error     = True
        error_message = str(e)
        response      = {"error": error_message}

    finally:
        # ── Always log the request ────────────────────────────
        # Runs whether the request succeeded or failed
        response_time_ms = int((time.time() - start_time) * 1000)
        try:
            log_request(
                query            = query,
                top_k            = top_k,
                rerank_used      = rerank,
                extract_used     = extract,
                contradictions   = do_contradict,
                generate_used    = generate,
                strategize_used  = strategize,
                filters          = filters,
                retrieved_cases  = retrieved_cases,
                result_count     = len(formatted_chunks)
                                   if not had_error else 0,
                response_time_ms = response_time_ms,
                had_error        = had_error,
                error_message    = error_message,
            )
        except Exception:
            pass   # never let logging failure break the response

    if had_error:
        return jsonify(response), 500

    return jsonify(response)


# ── POST /evaluate ────────────────────────────────────────────
@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400

    if data.get("run_test_suite"):
        from retriever import search as retriever_search

        results       = []
        all_precision = []
        all_recall    = []
        all_mrr       = []

        for tc in TEST_CASES:
            chunks = retriever_search(tc["query"], top_k=5, rerank=True)
            report = evaluate_pipeline(
                query          = tc["query"],
                relevant_cases = tc["relevant_cases"],
                chunks         = chunks,
                k              = 5,
            )
            results.append({
                "description": tc["description"],
                "query"      : tc["query"],
                "summary"    : report["summary"],
                "retrieval"  : report["retrieval"],
            })
            all_precision.append(report["retrieval"]["precision_at_k"])
            all_recall.append(report["retrieval"]["recall_at_k"])
            all_mrr.append(report["retrieval"]["mrr"])

        return jsonify({
            "test_suite_results": results,
            "aggregate": {
                "mean_precision_at_5": round(
                    sum(all_precision)/len(all_precision), 4),
                "mean_recall_at_5"   : round(
                    sum(all_recall)/len(all_recall), 4),
                "mean_mrr"           : round(
                    sum(all_mrr)/len(all_mrr), 4),
                "tests_run"          : len(TEST_CASES),
            }
        })

    query          = data.get("query", "").strip()
    relevant_cases = data.get("relevant_cases", [])
    top_k          = int(data.get("top_k", 5))
    rerank         = data.get("rerank", True)
    eval_faith     = data.get("eval_faithfulness", False)
    eval_strat     = data.get("eval_strategy",     False)

    if not query:
        return jsonify({"error": "'query' is required"}), 400
    if not relevant_cases:
        return jsonify({"error": "'relevant_cases' list is required"}), 400

    from retriever  import search as retriever_search
    from generator  import generate_answer as gen_answer
    from strategist import generate_strategy as gen_strategy

    chunks = retriever_search(query, top_k=top_k, rerank=rerank)
    answer   = None
    strategy = None

    if eval_faith or eval_strat:
        gen    = gen_answer(query, chunks)
        answer = gen["answer"]

    if eval_strat:
        strat    = gen_strategy(query, chunks, generated_answer=answer)
        strategy = strat

    try:
        report = evaluate_pipeline(
            query            = query,
            relevant_cases   = relevant_cases,
            chunks           = chunks,
            generated_answer = answer,
            strategy         = strategy,
            k                = top_k,
        )
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── POST /ingest ──────────────────────────────────────────────
@app.route("/ingest", methods=["POST"])
def ingest():
    try:
        run_ingestion()
        collection = get_collection()
        return jsonify({
            "status"      : "ingestion complete",
            "total_chunks": collection.count()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🏛  Legal RAG — V10")
    print("   GET  http://localhost:5000/health")
    print("   GET  http://localhost:5000/status")
    print("   GET  http://localhost:5000/history")
    print("   GET  http://localhost:5000/cases")
    print("   POST http://localhost:5000/search")
    print("   POST http://localhost:5000/evaluate")
    print("   POST http://localhost:5000/ingest\n")
    app.run(debug=True, port=5000)