# src/main.py
# V2 — adds filter params, /cases endpoint, richer response format

import os
import sys
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from retriever import search, list_cases
from ingest    import run_ingestion, get_collection

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
            "version"     : "V2"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── GET /cases ────────────────────────────────────────────────
@app.route("/cases", methods=["GET"])
def cases():
    """
    List all indexed cases with their metadata.
    No query needed — returns one entry per unique document.

    Optional query params for filtering the list:
      ?legal_domain=Constitutional Law
      ?year_from=1950&year_to=1980
      ?bench_type=Constitution Bench
    """
    try:
        all_cases = list_cases()

        # Apply optional URL query param filters to the case list
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

        return jsonify({
            "count": len(all_cases),
            "cases": all_cases
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── POST /search ──────────────────────────────────────────────
@app.route("/search", methods=["POST"])
def search_cases():
    """
    Semantic search with optional metadata filters.

    Request body (JSON):
    {
        "query"  : "preventive detention without trial",
        "top_k"  : 5,
        "filters": {
            "legal_domain"    : "Constitutional Law",
            "bench_type"      : "Constitution Bench",
            "year"            : 1950,
            "year_from"       : 1950,
            "year_to"         : 1980,
            "outcome"         : "Petition dismissed",
            "petitioner_type" : "Individual"
        }
    }

    All fields except "query" are optional.
    Omit "filters" entirely for unfiltered search (same as V1).
    """
    data = request.get_json()

    if not data or "query" not in data:
        return jsonify({"error": "Request body must include a 'query' field"}), 400

    query   = data["query"].strip()
    top_k   = int(data.get("top_k", 5))
    filters = data.get("filters", {})

    if not query:
        return jsonify({"error": "'query' cannot be empty"}), 400
    if not 1 <= top_k <= 20:
        return jsonify({"error": "'top_k' must be between 1 and 20"}), 400

    try:
        results = search(query, top_k=top_k, filters=filters)
    except Exception as e:
        return jsonify({"error": f"Retrieval failed: {str(e)}"}), 500

    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append({
            "rank"           : i,
            "score"          : r["score"],
            "chunk_num"      : r["chunk_num"],
            "text"           : r["text"],
            # metadata
            "case_name"      : r["case_name"],
            "citation"       : r["citation"],
            "year"           : r["year"],
            "bench_type"     : r["bench_type"],
            "bench_size"     : r.get("bench_size", 0),
            "legal_domain"   : r["legal_domain"],
            "key_provisions" : r["key_provisions"],
            "outcome"        : r["outcome"],
            "legal_principle": r["legal_principle"],
            "petitioner_type": r["petitioner_type"],
            "source_file"    : r["source_file"],
        })

    return jsonify({
        "query"          : query,
        "filters_applied": filters,
        "count"          : len(formatted),
        "results"        : formatted
    })


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
    print("\n🏛  Legal RAG — V2")
    print("   GET  http://localhost:5000/status")
    print("   GET  http://localhost:5000/cases")
    print("   POST http://localhost:5000/search")
    print("   POST http://localhost:5000/ingest\n")
    app.run(debug=True, port=5000)