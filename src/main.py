# src/main.py
# ─────────────────────────────────────────────────────────────
# Flask API — wires ingest + retriever into HTTP endpoints
#
# Endpoints:
#   POST /ingest        → trigger PDF ingestion pipeline
#   POST /search        → semantic search over indexed cases
#   GET  /status        → how many chunks are in the index
# ─────────────────────────────────────────────────────────────

import os
import sys
import chromadb
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

# Add src/ to path so imports work when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from retriever import search
from ingest    import run_ingestion, get_collection

app = Flask(__name__)


# ── GET /status ───────────────────────────────────────────────
@app.route("/status", methods=["GET"])
def status():
    """
    Quick health check — confirms the index exists and shows chunk count.
    Always hit this first to make sure ChromaDB is reachable.
    """
    try:
        collection = get_collection()
        return jsonify({
            "status"      : "ok",
            "total_chunks": collection.count(),
            "collection"  : "legal_cases"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── POST /search ──────────────────────────────────────────────
@app.route("/search", methods=["POST"])
def search_cases():
    """
    Semantic search over indexed legal cases.

    Request body (JSON):
        { "query": "preventive detention without trial", "top_k": 5 }

    Response:
        {
            "query"  : "...",
            "count"  : 5,
            "results": [
                {
                    "rank"      : 1,
                    "score"     : 0.727,
                    "source_doc": "A_K_Gopalan_vs...",
                    "chunk_num" : 180,
                    "text"      : "...raw chunk text..."
                },
                ...
            ]
        }
    """
    data = request.get_json()

    # ── Validate input ────────────────────────────────────────
    if not data or "query" not in data:
        return jsonify({"error": "Request body must include a 'query' field"}), 400

    query  = data["query"].strip()
    top_k  = int(data.get("top_k", 5))

    if not query:
        return jsonify({"error": "'query' cannot be empty"}), 400

    if top_k < 1 or top_k > 20:
        return jsonify({"error": "'top_k' must be between 1 and 20"}), 400

    # ── Run retrieval ─────────────────────────────────────────
    try:
        results = search(query, top_k=top_k)
    except Exception as e:
        return jsonify({"error": f"Retrieval failed: {str(e)}"}), 500

    # ── Format response ───────────────────────────────────────
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append({
            "rank"      : i,
            "score"     : r["score"],
            "source_doc": r["source_doc"],
            "chunk_num" : r["chunk_num"],
            "text"      : r["text"]
        })

    return jsonify({
        "query"  : query,
        "count"  : len(formatted),
        "results": formatted
    })


# ── POST /ingest ──────────────────────────────────────────────
@app.route("/ingest", methods=["POST"])
def ingest():
    """
    Trigger the ingestion pipeline — scans data/cases/ and indexes new PDFs.
    This can take several minutes for large files (rate limit waits included).
    Returns immediately with a message; check /status after.

    In production you'd run this as a background task (Celery, etc.).
    For V1, synchronous is fine — you'll see output in the terminal.
    """
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
    print("\n🏛  Legal RAG — V1")
    print("   Endpoints:")
    print("   GET  http://localhost:5000/status")
    print("   POST http://localhost:5000/search")
    print("   POST http://localhost:5000/ingest\n")
    app.run(debug=True, port=5000)