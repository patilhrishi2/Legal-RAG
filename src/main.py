# src/main.py
# V6 — adds optional rerank flag

import os
import sys
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from retriever import search, list_cases
from ingest    import run_ingestion, get_collection
from generator import generate_answer
from extractor import extract_chunks

app = Flask(__name__)


@app.route("/status", methods=["GET"])
def status():
    try:
        collection = get_collection()
        return jsonify({
            "status"      : "ok",
            "total_chunks": collection.count(),
            "collection"  : "legal_cases",
            "version"     : "V6"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


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


@app.route("/search", methods=["POST"])
def search_cases():
    """
    Hybrid search with optional re-ranking, extraction, and generation.

    Request body:
    {
        "query"    : "...",
        "top_k"    : 5,
        "filters"  : { "legal_domain": "Constitutional Law" },
        "rerank"   : true,    <- NEW in V6
        "extract"  : true,    <- V5
        "generate" : true     <- V4
    }

    Full pipeline when all flags true:
      hybrid search (top 20) → re-rank (top 5) → extract → generate

    rerank=True adds ~1s latency (cross-encoder on CPU).
    Model loads once on first rerank request, cached for all subsequent.
    """
    data = request.get_json()

    if not data or "query" not in data:
        return jsonify({"error": "Request body must include a 'query' field"}), 400

    query    = data["query"].strip()
    top_k    = int(data.get("top_k",    5))
    filters  = data.get("filters",      {})
    rerank   = data.get("rerank",       False)   # ← new in V6
    extract  = data.get("extract",      False)
    generate = data.get("generate",     False)

    if not query:
        return jsonify({"error": "'query' cannot be empty"}), 400
    if not 1 <= top_k <= 20:
        return jsonify({"error": "'top_k' must be between 1 and 20"}), 400

    # ── Step 1: Retrieval + optional re-ranking ───────────────
    try:
        chunks = search(query, top_k=top_k, filters=filters, rerank=rerank)
    except Exception as e:
        return jsonify({"error": f"Retrieval failed: {str(e)}"}), 500

    # ── Step 2: Optional extraction ───────────────────────────
    if extract:
        try:
            chunks = extract_chunks(chunks)
        except Exception as e:
            return jsonify({"error": f"Extraction failed: {str(e)}"}), 500

    # ── Step 3: Format chunks ─────────────────────────────────
    formatted_chunks = []
    for i, r in enumerate(chunks, 1):
        chunk_entry = {
            "rank"            : i,
            "score"           : r["score"],
            "rrf_score"       : r.get("rrf_score"),
            "rerank_score"    : r.get("rerank_score"),   # None if not reranked
            "rerank_rank"     : r.get("rerank_rank"),    # None if not reranked
            "semantic_score"  : r.get("semantic_score"),
            "bm25_score"      : r.get("bm25_score"),
            "in_both"         : r.get("in_both"),
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
        }
        if extract and "extraction" in r:
            chunk_entry["extraction"] = r["extraction"]

        formatted_chunks.append(chunk_entry)

    # ── Step 4: Optional generation ───────────────────────────
    response = {
        "query"          : query,
        "filters_applied": filters,
        "rerank_used"    : rerank,
        "extract_used"   : extract,
        "count"          : len(formatted_chunks),
        "results"        : formatted_chunks,
        "generated"      : None,
    }

    if generate:
        try:
            gen_result = generate_answer(
                query,
                chunks,
                use_extraction=extract
            )
            response["generated"] = {
                "answer"               : gen_result["answer"],
                "key_legal_principles" : gen_result["key_legal_principles"],
                "sources_used"         : gen_result["sources_used"],
                "confidence"           : gen_result["confidence"],
                "sources"              : gen_result["sources"],
                "extraction_used"      : gen_result.get("extraction_used", False),
                "error"                : gen_result["error"],
            }
        except Exception as e:
            response["generated"] = {"answer": "", "error": str(e)}

    return jsonify(response)


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


if __name__ == "__main__":
    print("\n🏛  Legal RAG — V6")
    print("   GET  http://localhost:5000/status")
    print("   GET  http://localhost:5000/cases")
    print("   POST http://localhost:5000/search")
    print("   POST http://localhost:5000/ingest\n")
    app.run(debug=True, port=5000)