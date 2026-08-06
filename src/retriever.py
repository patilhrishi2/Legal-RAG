# src/retriever.py
# ─────────────────────────────────────────────────────────────
# V3 QUERY PIPELINE — Hybrid Search with RRF
#
# Changes from V2:
#   - Runs BM25 keyword search alongside semantic search
#   - Fuses both ranked lists using Reciprocal Rank Fusion (RRF)
#   - Fetches full chunk text + metadata from ChromaDB by ID
#     (BM25 returns IDs only — text lives in ChromaDB)
#   - Metadata filtering applied to BOTH pipelines
#
# Everything else identical to V2:
#   - Same Gemini embedding for query
#   - Same ChromaDB where-clause for filters
#   - Same response structure
# ─────────────────────────────────────────────────────────────

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

# How many candidates each pipeline retrieves before fusion.
# More candidates = better fusion coverage but more ChromaDB lookups.
# 20 is the standard starting point for hybrid RAG systems.
CANDIDATE_K = 20

# RRF constant — dampens the impact of top-ranked results.
# k=60 is the value used in the original RRF paper (Cormack et al. 2009).
# Lower k = top ranks matter more. Higher k = more even distribution.
RRF_K = 60


# ── ChromaDB connection ───────────────────────────────────────
def get_collection():
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma.get_collection(name=COLLECTION)


# ── Query embedding (unchanged from V2) ──────────────────────
def embed_query(query_text: str) -> list[float]:
    result = gemini_client.models.embed_content(
        model    = EMBED_MODEL,
        contents = [query_text],
        config   = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return result.embeddings[0].values


# ── Where-clause builder (unchanged from V2) ─────────────────
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


# ── Pipeline 1: Semantic search ───────────────────────────────
def semantic_search(query_text: str, top_k: int, where: dict | None) -> list[dict]:
    """
    Embed query → cosine similarity search in ChromaDB.
    Returns top_k results with semantic rank and distance score.

    Identical to V2 retrieval — just renamed and returns rank too.
    """
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
            "id"           : chunk_id,
            "semantic_rank": rank,
            "semantic_score": round(1 - dist, 4),
            "text"         : doc,
            "metadata"     : meta,
        })

    return results


# ── Pipeline 2: BM25 search with metadata filter ──────────────
def bm25_search_filtered(
    query_text : str,
    top_k      : int,
    where      : dict | None
) -> list[dict]:
    """
    BM25 keyword search, then filter results by metadata if needed.

    WHY POST-FILTER FOR BM25?
    ChromaDB's where-clause only works with its own vector search.
    BM25 runs on our pickle index which has no metadata awareness.
    So we: retrieve more candidates (top_k * 4), then discard any
    chunk whose metadata doesn't match the filter.

    This is the one place we post-filter — it's acceptable here because:
    (a) BM25 is fast (no API calls), so fetching extra candidates is cheap
    (b) The BM25 index doesn't store metadata — adding it would duplicate storage
    """
    # Get raw BM25 results — more candidates to account for post-filter loss
    raw_results = search_bm25(query_text, top_k=top_k * 4)

    if not raw_results or not where:
        # No filter needed — just return top_k as-is with ranks
        for rank, r in enumerate(raw_results[:top_k], 1):
            r["bm25_rank"] = rank
        return raw_results[:top_k]

    # Apply metadata filter: fetch metadata for each candidate from ChromaDB
    collection   = get_collection()
    candidate_ids = [r["id"] for r in raw_results]

    try:
        meta_fetch = collection.get(
            ids     = candidate_ids,
            include = ["metadatas"]
        )
    except Exception:
        return []

    # Build a lookup: chunk_id → metadata
    meta_lookup = {
        chunk_id: meta
        for chunk_id, meta in zip(meta_fetch["ids"], meta_fetch["metadatas"])
    }

    # Filter: keep only chunks that satisfy the where-clause
    filtered = []
    for r in raw_results:
        meta = meta_lookup.get(r["id"], {})
        if metadata_matches(meta, where):
            filtered.append(r)
        if len(filtered) == top_k:
            break

    # Assign BM25 ranks within the filtered list
    for rank, r in enumerate(filtered, 1):
        r["bm25_rank"] = rank

    return filtered


def metadata_matches(meta: dict, where: dict) -> bool:
    """
    Check if a chunk's metadata satisfies a ChromaDB-style where-clause.
    Used to post-filter BM25 results which don't go through ChromaDB's filter.

    Supports: $eq, $gte, $lte, $and
    Mirrors the logic in build_where_clause() so both pipelines
    apply the same filter semantics.
    """
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


# ── Reciprocal Rank Fusion ────────────────────────────────────
def reciprocal_rank_fusion(
    semantic_results : list[dict],
    bm25_results     : list[dict],
    top_k            : int
) -> list[dict]:
    """
    Combine two ranked lists into one using RRF.

    Formula for each chunk:
        RRF_score = 1/(k + semantic_rank) + 1/(k + bm25_rank)

    If a chunk appears in only one list, it gets a partial score.
    If it appears in both, the scores add — rewarding agreement.

    WHY NOT JUST AVERAGE THE RAW SCORES?
    BM25 scores (0–25) and semantic scores (0–1) are on completely
    different scales. You can't meaningfully average them without
    careful normalisation. RRF sidesteps this entirely by only
    using rank positions — ranks are always comparable integers.

    Example with k=60:
      Chunk A: semantic_rank=1, bm25_rank=3
        RRF = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226

      Chunk B: semantic_rank=2, no BM25 result
        RRF = 1/(60+2) + 0         = 0.01613

      Chunk C: semantic_rank=15, bm25_rank=1
        RRF = 1/(60+15) + 1/(60+1) = 0.01333 + 0.01639 = 0.02972

      Final order: A > C > B
      A wins because both systems agree it's relevant.
    """
    # Build lookup: chunk_id → result dict for each pipeline
    semantic_lookup = {r["id"]: r for r in semantic_results}
    bm25_lookup     = {r["id"]: r for r in bm25_results}

    # Collect all unique chunk IDs seen in either list
    all_ids = set(semantic_lookup.keys()) | set(bm25_lookup.keys())

    fused = []
    for chunk_id in all_ids:
        sem  = semantic_lookup.get(chunk_id)
        bm25 = bm25_lookup.get(chunk_id)

        # RRF score — add contribution from whichever lists contain this chunk
        rrf_score = 0.0
        if sem:
            rrf_score += 1.0 / (RRF_K + sem["semantic_rank"])
        if bm25:
            rrf_score += 1.0 / (RRF_K + bm25["bm25_rank"])

        # Prefer semantic result for text/metadata (more complete)
        # Fall back to fetching from ChromaDB if only in BM25
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

    # For chunks only in BM25 (no semantic result), fetch text from ChromaDB
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
            data       = text_lookup.get(r["id"], {})
            r["text"]  = data.get("text",     "")
            r["metadata"] = data.get("metadata", {})

    return fused[:top_k]


# ── Main search function ──────────────────────────────────────
def search(query_text: str, top_k: int = 5, filters: dict = None) -> list[dict]:
    """
    Full hybrid search pipeline:
      1. Build where-clause from filters
      2. Run semantic search  → top CANDIDATE_K results
      3. Run BM25 search      → top CANDIDATE_K results (post-filtered)
      4. Fuse with RRF        → top_k final results
      5. Format and return

    The response structure is identical to V2 — main.py needs no changes.
    """
    where = build_where_clause(filters or {})

    # Run both pipelines
    semantic_results = semantic_search(query_text, top_k=CANDIDATE_K, where=where)
    bm25_results     = bm25_search_filtered(query_text, top_k=CANDIDATE_K, where=where)

    # Fuse
    fused = reciprocal_rank_fusion(semantic_results, bm25_results, top_k=top_k)

    # Format — same structure as V2 so main.py works unchanged
    results = []
    for i, r in enumerate(fused, 1):
        meta = r.get("metadata", {})
        results.append({
            "text"            : r["text"],
            "id"              : r["id"],
            "score"           : r["rrf_score"],        # RRF score (replaces cosine)
            "semantic_score"  : r["semantic_score"],   # V2 score, now visible
            "bm25_score"      : r["bm25_score"],       # new in V3
            "bm25_rank"       : r["bm25_rank"],        # new in V3
            "in_both"         : r["in_both"],          # True = both systems agreed
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


# ── List all cases (unchanged from V2) ───────────────────────
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


# ── Terminal test ─────────────────────────────────────────────
def print_results(query: str, results: list, filters: dict = None):
    print(f"\n{'─'*65}")
    print(f"Query  : {query}")
    if filters:
        print(f"Filters: {filters}")
    print(f"{'─'*65}")

    if not results:
        print("  No results.")
        return

    for r in results:
        both = "✓ both" if r["in_both"] else "  one "
        sem  = f"{r['semantic_score']:.4f}" if r["semantic_score"] else "  n/a "
        bm25 = f"{r['bm25_score']:.2f}"     if r["bm25_score"]     else "  n/a"
        print(f"\n[{r['chunk_num']:>3}] RRF={r['score']:.5f}  sem={sem}  bm25={bm25}  {both}")
        print(f"       {r['case_name']}  |  {r['legal_domain']}  |  {r['bench_type']}")
        print(f"       {r['text'][:180]}...")


if __name__ == "__main__":
    # Test 1 — semantic query (paraphrasing, no exact terms)
    print_results(
        "personal liberty cannot be taken away arbitrarily",
        search("personal liberty cannot be taken away arbitrarily", top_k=3)
    )

    # Test 2 — exact citation (BM25 should dominate)
    print_results(
        "AIR 1950 SC 27",
        search("AIR 1950 SC 27", top_k=3)
    )

    # Test 3 — mixed (both pipelines contribute)
    print_results(
        "Article 22 preventive detention fundamental rights",
        search("Article 22 preventive detention fundamental rights", top_k=3)
    )

    # Test 4 — with filter
    print_results(
        "contract commission agent",
        search("contract commission agent", top_k=3,
               filters={"legal_domain": "Contract Law"}),
        filters={"legal_domain": "Contract Law"}
    )