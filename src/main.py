# src/main.py
# V4 — adds optional generation to /search endpoint

import os
import sys
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from retriever  import search, list_cases
from ingest     import run_ingestion, get_collection
from generator  import generate_answer

app = Flask(__name__)


# ── GET /status ───────────────────────────────────────────────
@app.route("/status", methods=["GET"])
def status():
    try:
        collection = get_collection()
        return jsonify({
            "status"      : "ok",
            "total_chunks": collection.count(),
            "collection"  : "legal_cases",
            "version"     : "V4"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


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
    Hybrid search with optional generation.

    Request body:
    {
        "query"    : "What are the limits on preventive detention?",
        "top_k"    : 5,
        "filters"  : { "legal_domain": "Constitutional Law" },
        "generate" : true
    }

    "generate" is optional and defaults to false.
    When false  → behaviour identical to V3, raw chunks only.
    When true   → chunks passed to Gemini, structured answer returned.

    WHY OPTIONAL?
    Generation costs tokens and adds ~2-3 seconds of latency.
    Not every use case needs a generated answer — sometimes
    the user just wants to browse relevant chunks. Keeping
    generation opt-in preserves V3 behaviour by default and
    lets the frontend decide when to pay the latency cost.
    """
    data = request.get_json()

    if not data or "query" not in data:
        return jsonify({"error": "Request body must include a 'query' field"}), 400

    query    = data["query"].strip()
    top_k    = int(data.get("top_k", 5))
    filters  = data.get("filters", {})
    generate = data.get("generate", False)   # ← new in V4

    if not query:
        return jsonify({"error": "'query' cannot be empty"}), 400
    if not 1 <= top_k <= 20:
        return jsonify({"error": "'top_k' must be between 1 and 20"}), 400

    # ── Retrieval (identical to V3) ───────────────────────────
    try:
        chunks = search(query, top_k=top_k, filters=filters)
    except Exception as e:
        return jsonify({"error": f"Retrieval failed: {str(e)}"}), 500

    # ── Format retrieved chunks ───────────────────────────────
    formatted_chunks = []
    for i, r in enumerate(chunks, 1):
        formatted_chunks.append({
            "rank"            : i,
            "score"           : r["score"],
            "semantic_score"  : r["semantic_score"],
            "bm25_score"      : r["bm25_score"],
            "in_both"         : r["in_both"],
            "chunk_num"       : r["chunk_num"],
            "text"            : r["text"],
            "case_name"       : r["case_name"],
            "citation"        : r["citation"],
            "year"            : r["year"],
            "bench_type"      : r["bench_type"],
            "bench_size"      : r["bench_size"],
            "legal_domain"    : r["legal_domain"],
            "key_provisions"  : r["key_provisions"],
            "outcome"         : r["outcome"],
            "legal_principle" : r["legal_principle"],
            "petitioner_type" : r["petitioner_type"],
            "source_file"     : r["source_file"],
        })

    # ── Base response (V3 compatible) ─────────────────────────
    response = {
        "query"          : query,
        "filters_applied": filters,
        "count"          : len(formatted_chunks),
        "results"        : formatted_chunks,
        "generated"      : None,   # null when generate=false
    }

    # ── Generation (new in V4) ────────────────────────────────
    if generate:
        try:
            gen_result = generate_answer(query, chunks)
            response["generated"] = {
                "answer"               : gen_result["answer"],
                "key_legal_principles" : gen_result["key_legal_principles"],
                "sources_used"         : gen_result["sources_used"],
                "confidence"           : gen_result["confidence"],
                "sources"              : gen_result["sources"],
                "error"                : gen_result["error"],
            }
        except Exception as e:
            response["generated"] = {
                "answer" : "",
                "error"  : str(e),
            }

    return jsonify(response)


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
    print("\n🏛  Legal RAG — V4")
    print("   GET  http://localhost:5000/status")
    print("   GET  http://localhost:5000/cases")
    print("   POST http://localhost:5000/search")
    print("   POST http://localhost:5000/ingest\n")
    app.run(debug=True, port=5000)