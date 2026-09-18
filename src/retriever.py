# src/retriever.py
# V6 — adds optional re-ranking step after hybrid search

import os
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv
from bm25_index import search_bm25, tokenise

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
CHROMA_PATH   = "storage/chroma_db"
COLLECTION    = "legal_cases"
EMBED_MODEL   = "gemini-embedding-001"

# Retrieve more candidates when re-ranking is on —
# re-ranker needs a larger pool to make meaningful selections.
# Without re-ranking: retrieve top_k directly.
# With re-ranking   : retrieve RERANK_CANDIDATE_K, re-rank, return top_k.
CANDIDATE_K        = 20
RERANK_CANDIDATE_K = 20   # can increase to 30-40 as index grows
RRF_K              = 60


def get_collection():
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma.get_collection(name=COLLECTION)


def embed_query(query_text: str) -> list[float]:
    result = gemini_client.models.embed_content(
        model    = EMBED_MODEL,
        contents = [query_text],
        config   = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return result.embeddings[0].values


def build_where_clause(filters: dict) -> dict | None:
    if not filters:
        return None

    clauses = []
    for field in ["legal_domain", "bench_type", "outcome", "petitioner_type"]:
        if filters.get(field):
            clauses.append({field: {"$eq": filters[field]}})

    if filters.get("year"):
        clauses.append({"year": {"$eq": int(filters["year"])}})
    if filters.get("year_from"):
        clauses.append({"year": {"$gte": int(filters["year_from"])}})
    if filters.get("year_to"):
        clauses.append({"year": {"$lte": int(filters["year_to"])}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def semantic_search(query_text: str, top_k: int,
                    where: dict | None) -> list[dict]:
    collection = get_collection()
    query_vec  = embed_query(query_text)

    kwargs = {
        "query_embeddings": [query_vec],
        "n_results"       : top_k,
        "include"         : ["documents", "distances", "metadatas"],
    }
    if where:
        kwargs["where"] = where

    try:
        raw = collection.query(**kwargs)
    except Exception:
        return []

    results = []
    for rank, (doc, dist, meta, chunk_id) in enumerate(zip(
        raw["documents"][0],
        raw["distances"][0],
        raw["metadatas"][0],
        raw["ids"][0],
    ), 1):
        results.append({
            "id"            : chunk_id,
            "semantic_rank" : rank,
            "semantic_score": round(1 - dist, 4),
            "text"          : doc,
            "metadata"      : meta,
        })

    return results


def metadata_matches(meta: dict, where: dict) -> bool:
    if "$and" in where:
        return all(metadata_matches(meta, clause) for clause in where["$and"])

    for field, condition in where.items():
        val = meta.get(field)
        if isinstance(condition, dict):
            if "$eq"  in condition and val != condition["$eq"]:
                return False
            if "$gte" in condition and (val is None or val < condition["$gte"]):
                return False
            if "$lte" in condition and (val is None or val > condition["$lte"]):
                return False
        else:
            if val != condition:
                return False
    return True


def bm25_search_filtered(query_text: str, top_k: int,
                         where: dict | None) -> list[dict]:
    raw_results = search_bm25(query_text, top_k=top_k * 4)

    if not raw_results or not where:
        for rank, r in enumerate(raw_results[:top_k], 1):
            r["bm25_rank"] = rank
        return raw_results[:top_k]

    collection    = get_collection()
    candidate_ids = [r["id"] for r in raw_results]

    try:
        meta_fetch = collection.get(
            ids     = candidate_ids,
            include = ["metadatas"]
        )
    except Exception:
        return []

    meta_lookup = {
        cid: meta
        for cid, meta in zip(meta_fetch["ids"], meta_fetch["metadatas"])
    }

    filtered = []
    for r in raw_results:
        meta = meta_lookup.get(r["id"], {})
        if metadata_matches(meta, where):
            filtered.append(r)
        if len(filtered) == top_k:
            break

    for rank, r in enumerate(filtered, 1):
        r["bm25_rank"] = rank

    return filtered


def reciprocal_rank_fusion(semantic_results: list[dict],
                           bm25_results: list[dict],
                           top_k: int) -> list[dict]:
    semantic_lookup = {r["id"]: r for r in semantic_results}
    bm25_lookup     = {r["id"]: r for r in bm25_results}
    all_ids         = set(semantic_lookup.keys()) | set(bm25_lookup.keys())

    fused = []
    for chunk_id in all_ids:
        sem  = semantic_lookup.get(chunk_id)
        bm25 = bm25_lookup.get(chunk_id)

        rrf_score = 0.0
        if sem:
            rrf_score += 1.0 / (RRF_K + sem["semantic_rank"])
        if bm25:
            rrf_score += 1.0 / (RRF_K + bm25["bm25_rank"])

        result_source = sem or {}
        fused.append({
            "id"            : chunk_id,
            "rrf_score"     : round(rrf_score, 6),
            "semantic_score": sem["semantic_score"]  if sem  else None,
            "semantic_rank" : sem["semantic_rank"]   if sem  else None,
            "bm25_score"    : bm25["bm25_score"]     if bm25 else None,
            "bm25_rank"     : bm25["bm25_rank"]      if bm25 else None,
            "text"          : result_source.get("text",     ""),
            "metadata"      : result_source.get("metadata", {}),
            "in_both"       : sem is not None and bm25 is not None,
        })

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)

    needs_text = [r for r in fused[:top_k] if not r["text"]]
    if needs_text:
        collection = get_collection()
        fetched    = collection.get(
            ids     = [r["id"] for r in needs_text],
            include = ["documents", "metadatas"]
        )
        text_lookup = {
            cid: {"text": doc, "metadata": meta}
            for cid, doc, meta in zip(
                fetched["ids"],
                fetched["documents"],
                fetched["metadatas"]
            )
        }
        for r in needs_text:
            data          = text_lookup.get(r["id"], {})
            r["text"]     = data.get("text",     "")
            r["metadata"] = data.get("metadata", {})

    return fused[:top_k]


def search(query_text: str, top_k: int = 5,
           filters: dict = None, rerank: bool = False) -> list[dict]:
    """
    Full hybrid search pipeline with optional re-ranking.

    rerank=False (default) → V3/V4/V5 behaviour unchanged
    rerank=True            → retrieve RERANK_CANDIDATE_K candidates,
                             re-rank with cross-encoder, return top_k

    WHY IMPORT RERANKER INSIDE THE FUNCTION?
    reranker.py loads the cross-encoder model at import time (~1s, 85MB).
    If we import at the top of retriever.py, the model loads every time
    retriever.py is imported — even when reranking isn't needed.
    Importing inside the function means the model only loads on first
    rerank=True call, then stays cached in memory for subsequent calls.
    """
    where = build_where_clause(filters or {})

    # Decide candidate pool size
    candidate_k = RERANK_CANDIDATE_K if rerank else CANDIDATE_K

    # Run both pipelines
    semantic_results = semantic_search(query_text, top_k=candidate_k, where=where)
    bm25_results     = bm25_search_filtered(query_text, top_k=candidate_k, where=where)

    # Fuse — get more candidates if re-ranking, else get top_k directly
    fuse_k = RERANK_CANDIDATE_K if rerank else top_k
    fused  = reciprocal_rank_fusion(semantic_results, bm25_results, top_k=fuse_k)

    # ── Optional re-ranking step (new in V6) ──────────────────
    if rerank and fused:
        from reranker import rerank as rerank_fn
        fused = rerank_fn(query_text, fused, top_k=top_k)

    # Format results — same structure as V5
    results = []
    for i, r in enumerate(fused, 1):
        meta = r.get("metadata", {})
        results.append({
            "text"            : r["text"],
            "id"              : r["id"],
            # Score field: rerank_score if reranked, else rrf_score
            "score"           : r.get("rerank_score", r["rrf_score"]),
            "rrf_score"       : r["rrf_score"],
            "rerank_score"    : r.get("rerank_score"),   # None if not reranked
            "rerank_rank"     : r.get("rerank_rank"),    # None if not reranked
            "semantic_score"  : r["semantic_score"],
            "bm25_score"      : r["bm25_score"],
            "in_both"         : r["in_both"],
            "chunk_num"       : int(r["id"].rsplit("__chunk_", 1)[1])
                                if "__chunk_" in r["id"] else -1,
            "case_name"       : meta.get("case_name",       "Unknown"),
            "citation"        : meta.get("citation",        "Unknown"),
            "year"            : meta.get("year",            0),
            "bench_type"      : meta.get("bench_type",      "Unknown"),
            "bench_size"      : meta.get("bench_size",      0),
            "legal_domain"    : meta.get("legal_domain",    "Unknown"),
            "key_provisions"  : meta.get("key_provisions",  "Unknown"),
            "outcome"         : meta.get("outcome",         "Unknown"),
            "legal_principle" : meta.get("legal_principle", "Unknown"),
            "petitioner_type" : meta.get("petitioner_type", "Unknown"),
            "source_file"     : meta.get("source_file",     "Unknown"),
        })

    return results


def list_cases() -> list[dict]:
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

    cases.sort(key=lambda x: x["year"])
    return cases


def print_results(query: str, results: list, filters: dict = None,
                  reranked: bool = False):
    print(f"\n{'─'*65}")
    print(f"Query  : {query}")
    if filters:
        print(f"Filters: {filters}")
    print(f"Reranked: {reranked}")
    print(f"{'─'*65}")

    if not results:
        print("  No results.")
        return

    for r in results:
        score_str = (
            f"rerank={r['rerank_score']:+.4f}" if r.get("rerank_score") is not None
            else f"rrf={r['score']:.5f}"
        )
        both = "✓ both" if r["in_both"] else "  one "
        print(f"\n[{r['chunk_num']:>3}] {score_str}  {both}  {r['case_name']}")
        print(f"       {r['text'][:160]}...")


if __name__ == "__main__":
    # Compare V5 vs V6 on same query
    query = "What are the constitutional limits on preventive detention?"

    print("\n── V5 (no re-rank) ─────────────────────────────────────")
    results_v5 = search(query, top_k=5, rerank=False)
    print_results(query, results_v5, reranked=False)

    print("\n── V6 (with re-rank) ───────────────────────────────────")
    results_v6 = search(query, top_k=5, rerank=True)
    print_results(query, results_v6, reranked=True)