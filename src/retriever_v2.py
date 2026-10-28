# src/retriever_v2.py  (new)
import chromadb
from embedder import QwenEmbedder
from store import connect, search_fts

CHROMA_PATH, COLLECTION = "storage_v2/chroma", "legal_cases_qwen"
CANDIDATE_K, RRF_K = 20, 60

_state = {}

def _col():
    if "col" not in _state:
        _state["col"] = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(COLLECTION)
    return _state["col"]

def _emb():
    if "emb" not in _state:
        _state["emb"] = QwenEmbedder()
    return _state["emb"]

def _conn():
    if "conn" not in _state:
        _state["conn"] = connect()
    return _state["conn"]


def build_where(filters: dict):
    clauses = []
    for f in ("legal_domain", "bench_type", "outcome", "petitioner_type"):
        if filters.get(f):
            clauses.append({f: {"$eq": filters[f]}})
    if filters.get("year"):      clauses.append({"year": {"$eq":  int(filters["year"])}})
    if filters.get("year_from"): clauses.append({"year": {"$gte": int(filters["year_from"])}})
    if filters.get("year_to"):   clauses.append({"year": {"$lte": int(filters["year_to"])}})
    if not clauses: return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def semantic_search(query, top_k, where):
    kw = dict(query_embeddings=[_emb().embed_query(query)], n_results=top_k,
              include=["documents", "distances", "metadatas"])
    if where:
        kw["where"] = where
    raw = _col().query(**kw)          # errors are raised, not hidden
    return [{"id": i, "semantic_rank": r, "semantic_score": round(1 - d, 4),
             "text": doc, "metadata": m}
            for r, (i, doc, d, m) in enumerate(zip(
                raw["ids"][0], raw["documents"][0], raw["distances"][0], raw["metadatas"][0]), 1)]


def rrf(sem, kw, top_k):
    S = {r["id"]: r for r in sem}
    K = {r["id"]: r for r in kw}
    fused = []
    for cid in set(S) | set(K):
        s, k = S.get(cid), K.get(cid)
        score = (1 / (RRF_K + s["semantic_rank"]) if s else 0) + \
                (1 / (RRF_K + k["bm25_rank"]) if k else 0)
        fused.append({"id": cid, "rrf_score": round(score, 6),
                      "semantic_score": s["semantic_score"] if s else None,
                      "bm25_score": k["bm25_score"] if k else None,
                      "text": s["text"] if s else "", "metadata": s["metadata"] if s else {},
                      "in_both": bool(s and k)})
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    fused = fused[:top_k]
    missing = [r for r in fused if not r["text"]]      # keyword-only hits
    if missing:
        got = _col().get(ids=[r["id"] for r in missing], include=["documents", "metadatas"])
        look = {i: (d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}
        for r in missing:
            r["text"], r["metadata"] = look.get(r["id"], ("", {}))
    return fused


def search(query_text: str, top_k: int = 5, filters: dict = None, rerank: bool = False):
    filters = filters or {}
    where = build_where(filters)
    sem = semantic_search(query_text, CANDIDATE_K, where)
    kw  = search_fts(_conn(), query_text, CANDIDATE_K, filters)
    fused = rrf(sem, kw, CANDIDATE_K if rerank else top_k)
    for r in fused:
        r["score"] = r["rrf_score"]          # reranker reads this
    if rerank and fused:
        from reranker import rerank as rerank_fn
        fused = rerank_fn(query_text, fused, top_k=top_k)

    out = []
    for r in fused:
        m = r["metadata"]
        out.append({
            "text": r["text"], "id": r["id"],
            "score": r.get("rerank_score", r["rrf_score"]),
            "rrf_score": r["rrf_score"],
            "rerank_score": r.get("rerank_score"), "rerank_rank": r.get("rerank_rank"),
            "semantic_score": r["semantic_score"], "bm25_score": r["bm25_score"],
            "in_both": r["in_both"],
            "chunk_num": int(r["id"].rsplit("__chunk_", 1)[1]) if "__chunk_" in r["id"] else -1,
            "opinion": m.get("opinion", ""),
            **{k: m.get(k, d) for k, d in [
                ("case_name", "Unknown"), ("citation", "Unknown"), ("year", 0),
                ("bench_type", "Unknown"), ("bench_size", 0), ("legal_domain", "Unknown"),
                ("key_provisions", "Unknown"), ("outcome", "Unknown"),
                ("legal_principle", "Unknown"), ("petitioner_type", "Unknown"),
                ("source_file", "Unknown")]},
        })
    return out


if __name__ == "__main__":
    q = "What are the constitutional limits on preventive detention?"
    for rr in (False, True):
        print(f"\n── rerank={rr} ──")
        for r in search(q, top_k=5, rerank=rr):
            tag = "both" if r["in_both"] else "one "
            print(f"[{r['chunk_num']:>3}] {r['score']:+.4f} {tag} {r['case_name'][:45]} "
                  f"({r['year']}) opinion={r['opinion'] or '-'}")
    print("\n── filter year=1950, Criminal Law ──")
    for r in search("detention without trial", 3, {"year": 1950, "legal_domain": "Criminal Law"}):
        print(r["case_name"][:50], r["year"], r["legal_domain"])