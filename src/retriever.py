# src/retriever.py
# ─────────────────────────────────────────────────────────────
# V2 QUERY PIPELINE
#
# Changes from V1:
#   - Uses Gemini for query embedding (matches new index)
#   - Accepts optional filter parameters
#   - Builds ChromaDB where-clause from filters before searching
#   - Returns metadata alongside each result
#
# The core similarity search logic is identical to V1.
# The only new concept is the where-clause filter.
# ─────────────────────────────────────────────────────────────

import os
import requests
import chromadb
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_PATH  = "storage/chroma_db"
COLLECTION   = "legal_cases"
EMBED_MODEL  = "gemini-embedding-001"


# ── ChromaDB connection ───────────────────────────────────────
def get_collection():
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma.get_collection(name=COLLECTION)


# ── Query embedding via gemini ──────────────────────────────────

from google import genai
from google.genai import types

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
EMBED_MODEL   = "gemini-embedding-001"

def embed_query(query_text: str) -> list[float]:
    result = gemini_client.models.embed_content(
        model    = EMBED_MODEL,
        contents = [query_text],
        config   = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return result.embeddings[0].values


# ── Build ChromaDB where-clause from filters ──────────────────
def build_where_clause(filters: dict) -> dict | None:
    """
    Convert user-facing filter params into a ChromaDB where-clause.

    ChromaDB's where syntax:
      Exact match  : {"field": {"$eq": value}}
      Greater/equal: {"field": {"$gte": value}}
      Less/equal   : {"field": {"$lte": value}}
      AND          : {"$and": [clause1, clause2, ...]}

    We only add a clause for filters the user actually provided.
    If no filters → return None → ChromaDB searches the full index.

    Supported filters:
      legal_domain  → exact match  (e.g. "Constitutional Law")
      court         → not in schema, skipped (all cases are SC)
      year          → exact match  (e.g. 1950)
      year_from     → lower bound  (e.g. 1950)
      year_to       → upper bound  (e.g. 1980)
      bench_type    → exact match  (e.g. "Constitution Bench")
      outcome       → exact match  (e.g. "Petition dismissed")
      petitioner_type → exact match (e.g. "Individual")

    NOTE ON YEAR RANGES:
    ChromaDB cannot do a range with a single $eq.
    year_from and year_to produce $gte and $lte clauses
    that get combined with $and.
    """
    if not filters:
        return None

    clauses = []

    # Exact string matches
    for field in ["legal_domain", "bench_type", "outcome", "petitioner_type"]:
        if filters.get(field):
            clauses.append({field: {"$eq": filters[field]}})

    # Exact year match (if user gives a single year)
    if filters.get("year"):
        clauses.append({"year": {"$eq": int(filters["year"])}})

    # Year range — from
    if filters.get("year_from"):
        clauses.append({"year": {"$gte": int(filters["year_from"])}})

    # Year range — to
    if filters.get("year_to"):
        clauses.append({"year": {"$lte": int(filters["year_to"])}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]           # single clause — no $and needed
    return {"$and": clauses}        # multiple clauses — wrap in $and


# ── Main search function ──────────────────────────────────────
def search(query_text: str, top_k: int = 5, filters: dict = None) -> list[dict]:
    """
    Embed query → apply filters → similarity search → return results.

    HOW PRE-FILTERING WORKS:
    Without filters: ChromaDB scores ALL 319 chunks, returns top-k.
    With filters   : ChromaDB first discards chunks that don't match
                     the where-clause, then scores only the remainder.

    This is called pre-filtering and it's more efficient than
    retrieving everything and filtering afterwards. It also means
    your top-k results are always within the filtered subset —
    you won't get a score-0.9 result from the wrong domain
    pushing a score-0.75 result from the right domain out of top-5.

    IMPORTANT EDGE CASE:
    If filters are too strict and no chunks pass, ChromaDB returns
    an empty result. We handle this gracefully below.
    """
    collection = get_collection()
    query_vec  = embed_query(query_text)
    where      = build_where_clause(filters or {})

    query_kwargs = {
        "query_embeddings" : [query_vec],
        "n_results"        : top_k,
        "include"          : ["documents", "distances", "metadatas"],
    }

    # Only add where if we actually have filters
    # Passing where=None to ChromaDB raises an error
    if where:
        query_kwargs["where"] = where

    try:
        raw = collection.query(**query_kwargs)
    except Exception as e:
        # Often happens when filters match zero chunks
        error_msg = str(e)
        if "no documents" in error_msg.lower() or "does not exist" in error_msg.lower():
            return []
        raise

    # ── Unpack and format results ─────────────────────────────
    results = []
    for doc, dist, meta, chunk_id in zip(
        raw["documents"][0],
        raw["distances"][0],
        raw["metadatas"][0],
        raw["ids"][0],
    ):
        score     = round(1 - dist, 4)
        parts     = chunk_id.rsplit("__chunk_", 1)
        chunk_num = int(parts[1]) if len(parts) == 2 else -1

        results.append({
            "text"            : doc,
            "id"              : chunk_id,
            "score"           : score,
            "chunk_num"       : chunk_num,
            # ── metadata fields (new in V2) ───────────────────
            "case_name"       : meta.get("case_name",       "Unknown"),
            "citation"        : meta.get("citation",        "Unknown"),
            "year"            : meta.get("year",            0),
            "bench_type"      : meta.get("bench_type",      "Unknown"),
            "legal_domain"    : meta.get("legal_domain",    "Unknown"),
            "key_provisions"  : meta.get("key_provisions",  "Unknown"),
            "outcome"         : meta.get("outcome",         "Unknown"),
            "legal_principle" : meta.get("legal_principle", "Unknown"),
            "petitioner_type" : meta.get("petitioner_type", "Unknown"),
            "source_file"     : meta.get("source_file",     "Unknown"),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ── List all unique cases in the index ────────────────────────
def list_cases() -> list[dict]:
    """
    Return one metadata entry per unique case.
    Used by the GET /cases endpoint in main.py.

    ChromaDB doesn't have a GROUP BY — we fetch all metadatas
    and deduplicate by source_file ourselves.
    We only fetch metadatas (not documents or embeddings)
    to keep this fast.
    """
    collection = get_collection()
    raw        = collection.get(include=["metadatas"])

    seen  = set()
    cases = []

    for meta in raw["metadatas"]:
        src = meta.get("source_file", "")
        if src in seen:
            continue
        seen.add(src)
        cases.append({
            "case_name"      : meta.get("case_name",       "Unknown"),
            "citation"       : meta.get("citation",        "Unknown"),
            "year"           : meta.get("year",            0),
            "bench_type"     : meta.get("bench_type",      "Unknown"),
            "bench_size"     : meta.get("bench_size",      0),
            "legal_domain"   : meta.get("legal_domain",    "Unknown"),
            "key_provisions" : meta.get("key_provisions",  "Unknown"),
            "outcome"        : meta.get("outcome",         "Unknown"),
            "legal_principle": meta.get("legal_principle", "Unknown"),
            "petitioner_type": meta.get("petitioner_type", "Unknown"),
            "source_file"    : meta.get("source_file",     "Unknown"),
        })

    # Sort by year so the response is deterministic
    cases.sort(key=lambda x: x["year"])
    return cases


# ── Terminal test ─────────────────────────────────────────────
def print_results(query: str, results: list[dict], filters: dict = None):
    print(f"\n{'─'*65}")
    print(f"Query   : {query}")
    if filters:
        print(f"Filters : {filters}")
    print(f"{'─'*65}")

    if not results:
        print("  No results found (filters may be too strict).")
        return

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r['score']}  |  {r['case_name']}  |  {r['citation']}")
        print(f"    Domain : {r['legal_domain']}  |  Year: {r['year']}  |  {r['bench_type']}")
        print(f"    Outcome: {r['outcome']}")
        print(f"    Text   : {r['text'][:200]}...")


if __name__ == "__main__":
    # Test 1 — no filters (same as V1 but now returns metadata too)
    results = search("preventive detention without trial", top_k=3)
    print_results("preventive detention without trial", results)

    # Test 2 — filter by legal domain
    results = search("contract breach", top_k=3,
                     filters={"legal_domain": "Contract Law"})
    print_results("contract breach", results,
                  filters={"legal_domain": "Contract Law"})

    # Test 3 — filter by year range
    results = search("fundamental rights", top_k=3,
                     filters={"year_from": 1950, "year_to": 1955})
    print_results("fundamental rights", results,
                  filters={"year_from": 1950, "year_to": 1955})

    # Test 4 — filter by bench type
    results = search("constitutional validity", top_k=3,
                     filters={"bench_type": "Constitution Bench"})
    print_results("constitutional validity", results,
                  filters={"bench_type": "Constitution Bench"})

    # Test 5 — list all cases
    print(f"\n{'─'*65}")
    print("All indexed cases:")
    for c in list_cases():
        print(f"  {c['year']}  {c['case_name']:<50}  {c['legal_domain']}")