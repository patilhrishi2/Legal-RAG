# src/main_v2.py
import os, sys, time, datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

import chromadb
from retriever_v2  import search
from store         import connect
from generator     import generate_answer
from extractor     import extract_chunks
from contradiction import detect_contradictions
from strategist    import generate_strategy
from evaluator     import evaluate_pipeline, TEST_CASES
from database      import init_db, log_request, get_history, get_stats

CHROMA_PATH, COLLECTION = "storage_v2/chroma", "legal_cases_qwen"

app = Flask(__name__)
CORS(app)
init_db()

_col = {}
def get_collection():
    if "c" not in _col:
        _col["c"] = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(COLLECTION)
    return _col["c"]


def list_cases():
    conn = connect()
    try:
        rows = conn.execute("""SELECT case_name, citation, year, bench_type, bench_size,
                legal_domain, key_provisions, outcome, legal_principle, petitioner_type, source_file
                FROM cases ORDER BY year, case_name""").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.route("/health", methods=["GET"])
def health():
    rep = {"version": "V11.5", "timestamp": datetime.datetime.utcnow().isoformat(),
           "components": {}, "statistics": {}, "overall": "healthy"}
    issues = []
    try:
        rep["components"]["chromadb"] = {"status": "ok", "total_chunks": get_collection().count()}
    except Exception as e:
        rep["components"]["chromadb"] = {"status": "error", "error": str(e)}
        issues.append("chromadb unreachable")
    db = "storage_v2/legal.db"
    if os.path.exists(db):
        rep["components"]["keyword_index_fts5"] = {"status": "ok", "size_kb": os.path.getsize(db) // 1024}
    else:
        rep["components"]["keyword_index_fts5"] = {"status": "missing"}
        issues.append("legal.db missing")
    rep["components"]["log_database"] = {"status": "ok" if os.path.exists("storage/logs.db") else "not initialised"}
    cache = os.path.expanduser("~/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2")
    rep["components"]["reranker_model"] = {"status": "cached" if os.path.exists(cache) else "will download on first use"}
    try:
        rep["statistics"] = get_stats()
    except Exception as e:
        rep["statistics"] = {"error": str(e)}
    try:
        cs = list_cases()
        rep["case_count"] = len(cs)
        rep["indexed_cases"] = [{"case_name": c["case_name"], "citation": c["citation"],
                                 "year": c["year"], "legal_domain": c["legal_domain"]} for c in cs[-50:]]
    except Exception:
        rep["indexed_cases"] = []
    if issues:
        rep["overall"] = "degraded — " + ", ".join(issues)
    return jsonify(rep)


@app.route("/status", methods=["GET"])
def status():
    try:
        return jsonify({"status": "ok", "total_chunks": get_collection().count(),
                        "collection": COLLECTION, "version": "V11.5"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/history", methods=["GET"])
def history():
    try:
        return jsonify(get_history(
            limit=int(request.args.get("limit", 20)), offset=int(request.args.get("offset", 0)),
            query_filter=request.args.get("query"),
            errors_only=request.args.get("errors_only", "false").lower() == "true"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/cases", methods=["GET"])
def cases():
    try:
        allc = list_cases()
        d, yf, yt, b = (request.args.get("legal_domain"), request.args.get("year_from", type=int),
                        request.args.get("year_to", type=int), request.args.get("bench_type"))
        if d:  allc = [c for c in allc if c["legal_domain"] == d]
        if yf: allc = [c for c in allc if (c["year"] or 0) >= yf]
        if yt: allc = [c for c in allc if (c["year"] or 0) <= yt]
        if b:  allc = [c for c in allc if c["bench_type"] == b]
        return jsonify({"count": len(allc), "cases": allc})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/search", methods=["POST"])
def search_cases():
    start = time.time()
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Request body must include a 'query' field"}), 400

    query      = data["query"].strip()
    top_k      = int(data.get("top_k", 5))
    filters    = data.get("filters", {})
    rerank     = data.get("rerank", False)
    extract    = data.get("extract", False)
    do_contra  = data.get("detect_contradictions", False)
    generate   = data.get("generate", False)
    strategize = data.get("strategize", False)

    if not query:
        return jsonify({"error": "'query' cannot be empty"}), 400
    if not 1 <= top_k <= 20:
        return jsonify({"error": "'top_k' must be between 1 and 20"}), 400

    retrieved_cases, had_error, error_message = [], False, ""
    formatted_chunks = []
    try:
        chunks = search(query, top_k=top_k, filters=filters, rerank=rerank)
        seen = set()
        for c in chunks:
            s = c.get("source_file", "")
            if s and s not in seen:
                retrieved_cases.append(s); seen.add(s)

        if extract:
            chunks = extract_chunks(chunks)

        contradiction_result = None
        if do_contra:
            try:
                contradiction_result = detect_contradictions(query, chunks)
            except Exception as e:
                contradiction_result = {"contradictions_found": False, "contradiction_count": 0,
                                        "contradictions": [], "analysis_note": "", "error": str(e)}

        for i, r in enumerate(chunks, 1):
            entry = {"rank": i, "score": r["score"], "rrf_score": r.get("rrf_score"),
                     "rerank_score": r.get("rerank_score"), "rerank_rank": r.get("rerank_rank"),
                     "semantic_score": r.get("semantic_score"), "bm25_score": r.get("bm25_score"),
                     "in_both": r.get("in_both"), "chunk_num": r["chunk_num"], "text": r["text"],
                     "opinion": r.get("opinion", ""),
                     **{k: r[k] for k in ("case_name", "citation", "year", "bench_type", "bench_size",
                                          "legal_domain", "key_provisions", "outcome",
                                          "legal_principle", "petitioner_type", "source_file")}}
            if extract and "extraction" in r:
                entry["extraction"] = r["extraction"]
            formatted_chunks.append(entry)

        response = {"query": query, "filters_applied": filters, "rerank_used": rerank,
                    "extract_used": extract, "contradiction_detection": contradiction_result,
                    "count": len(formatted_chunks), "results": formatted_chunks,
                    "generated": None, "strategy": None}

        gen_result = None
        if generate:
            try:
                gen_result = generate_answer(query, chunks, use_extraction=extract,
                                             contradiction_report=contradiction_result)
                response["generated"] = {
                    "answer": gen_result["answer"],
                    "contradictions": gen_result.get("contradictions", ""),
                    "key_legal_principles": gen_result["key_legal_principles"],
                    "sources_used": gen_result["sources_used"],
                    "confidence": gen_result["confidence"],
                    "sources": gen_result["sources"],
                    "extraction_used": gen_result.get("extraction_used", False),
                    "error": gen_result["error"]}
            except Exception as e:
                response["generated"] = {"answer": "", "error": str(e)}

        if strategize:
            try:
                s = generate_strategy(query, chunks, contradiction_report=contradiction_result,
                                      generated_answer=gen_result["answer"] if gen_result else None)
                response["strategy"] = {k: s[k] for k in (
                    "situation_summary", "strength_assessment", "winning_arguments",
                    "failure_patterns", "key_decisive_factors", "recommended_strategy",
                    "risk_factors", "confidence", "error")}
            except Exception as e:
                response["strategy"] = {"situation_summary": "", "strength_assessment": "UNKNOWN",
                    "winning_arguments": "", "failure_patterns": "", "key_decisive_factors": "",
                    "recommended_strategy": "", "risk_factors": "", "confidence": "LOW", "error": str(e)}

    except Exception as e:
        had_error, error_message = True, str(e)
        response = {"error": error_message}

    finally:
        try:
            log_request(query=query, top_k=top_k, rerank_used=rerank, extract_used=extract,
                        contradictions=do_contra, generate_used=generate, strategize_used=strategize,
                        filters=filters, retrieved_cases=retrieved_cases,
                        result_count=0 if had_error else len(formatted_chunks),
                        response_time_ms=int((time.time() - start) * 1000),
                        had_error=had_error, error_message=error_message)
        except Exception:
            pass

    return (jsonify(response), 500) if had_error else jsonify(response)


@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400

    if data.get("run_test_suite"):
        res, P, R, M = [], [], [], []
        for tc in TEST_CASES:
            chunks = search(tc["query"], top_k=5, rerank=True)
            rep = evaluate_pipeline(query=tc["query"], relevant_cases=tc["relevant_cases"],
                                    chunks=chunks, k=5)
            res.append({"description": tc["description"], "query": tc["query"],
                        "summary": rep["summary"], "retrieval": rep["retrieval"]})
            P.append(rep["retrieval"]["precision_at_k"]); R.append(rep["retrieval"]["recall_at_k"])
            M.append(rep["retrieval"]["mrr"])
        n = len(TEST_CASES)
        return jsonify({"test_suite_results": res, "aggregate": {
            "mean_precision_at_5": round(sum(P)/n, 4), "mean_recall_at_5": round(sum(R)/n, 4),
            "mean_mrr": round(sum(M)/n, 4), "tests_run": n}})

    query = data.get("query", "").strip()
    relevant = data.get("relevant_cases", [])
    top_k = int(data.get("top_k", 5))
    if not query:    return jsonify({"error": "'query' is required"}), 400
    if not relevant: return jsonify({"error": "'relevant_cases' list is required"}), 400

    chunks = search(query, top_k=top_k, rerank=data.get("rerank", True))
    answer = strategy = None
    ef, es = data.get("eval_faithfulness", False), data.get("eval_strategy", False)
    if ef or es:
        answer = generate_answer(query, chunks)["answer"]
    if es:
        strategy = generate_strategy(query, chunks, generated_answer=answer)
    try:
        return jsonify(evaluate_pipeline(query=query, relevant_cases=relevant, chunks=chunks,
                                         generated_answer=answer, strategy=strategy, k=top_k))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n🏛  Legal RAG — V11.5 (Qwen3 + FTS5)  http://localhost:5000")
    app.run(debug=False, use_reloader=False, threaded=False, port=5000)