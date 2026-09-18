# src/reranker.py
# ─────────────────────────────────────────────────────────────
# V6 RE-RANKING
#
# What this file does:
#   Takes retrieved chunks and scores each one against the query
#   using a cross-encoder model. Returns the same chunks sorted
#   by re-rank score instead of RRF score.
#
# WHY CROSS-ENCODER OVER BI-ENCODER?
#
#   Bi-encoder (what retriever.py uses):
#     query → embed → vector A
#     chunk → embed → vector B
#     similarity = cosine(A, B)
#     Query and chunk are embedded INDEPENDENTLY then compared.
#     Fast (embeddings pre-computed at index time) but imprecise —
#     the model never sees query and chunk together.
#
#   Cross-encoder (what this file uses):
#     [query + chunk] → model → single relevance score
#     Query and chunk are processed TOGETHER in one pass.
#     The model can attend to every word in the query while
#     reading every word in the chunk simultaneously.
#     Slower (must run per query) but much more precise.
#
# WHY THIS MODEL?
#   cross-encoder/ms-marco-MiniLM-L-6-v2 from HuggingFace.
#   - Trained on MS MARCO passage ranking (130M query-passage pairs)
#   - 85MB — runs on CPU, no GPU needed
#   - Standard model used in production RAG re-ranking
#   - Free, no API key, runs locally
#   - Inference time: ~50ms per chunk on CPU
#     For 20 chunks: ~1 second total
#
# STANDARD RAG RE-RANKING PATTERN:
#   1. Retrieve top 20 (generous recall)
#   2. Re-rank all 20 against query (precise scoring)
#   3. Take top 5 (high precision subset)
#   This way you never miss a relevant chunk (step 1)
#   and never dilute context with irrelevant ones (step 3).
# ─────────────────────────────────────────────────────────────

import os
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

# ── Model setup ───────────────────────────────────────────────
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Load once at module level — avoids reloading on every request.
# First run downloads ~85MB to HuggingFace cache (~/.cache/huggingface).
# Subsequent runs load from cache in <1 second.
print(f"Loading re-ranker model: {MODEL_NAME}")
_cross_encoder = CrossEncoder(MODEL_NAME, max_length=512)
print("Re-ranker ready.")


# ── Re-rank function ──────────────────────────────────────────
def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Score each chunk against the query using the cross-encoder.
    Return top_k chunks sorted by re-rank score descending.

    Input:
        query  : the user's original question
        chunks : list of chunk dicts from retriever.search()
                 (each has "text", "score", metadata, etc.)
        top_k  : how many to return after re-ranking

    Output:
        Same chunk dicts, sorted by rerank_score descending,
        with two new fields added per chunk:
          rerank_score : float from cross-encoder (-10 to +10 range)
          rerank_rank  : integer position after re-ranking (1 = best)

    WHY -10 TO +10?
    CrossEncoder with ms-marco outputs raw logits, not probabilities.
    Positive scores = relevant, negative = not relevant.
    You don't need to normalise — only relative ordering matters.

    WHY top_k=5 DEFAULT?
    Standard RAG context window. More than 5 chunks rarely improves
    generation quality and increases latency + token cost.
    Retrieval uses top 20 candidates; re-ranking selects the best 5.
    """
    if not chunks:
        return []

    # Build (query, chunk_text) pairs for the cross-encoder
    # The model sees both together in one forward pass
    pairs = [(query, chunk["text"]) for chunk in chunks]

    # Score all pairs — returns numpy array of floats
    # This is the expensive step: O(n_chunks) forward passes
    scores = _cross_encoder.predict(pairs)

    # Attach scores to chunk dicts
    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])
        chunk["retrieval_score"] = chunk.get("score", 0)  # preserve original RRF score

    # Sort by re-rank score descending
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    # Assign re-rank ranks
    for rank, chunk in enumerate(reranked, 1):
        chunk["rerank_rank"] = rank

    return reranked[:top_k]


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from retriever import search

    queries = [
        "What are the constitutional limits on preventive detention?",
        "What arguments did the petitioner make about Article 22?",
    ]

    for query in queries:
        print(f"\n{'═'*65}")
        print(f"Query: {query}")
        print(f"{'─'*65}")

        # Retrieve more candidates than we'll keep
        chunks = search(query, top_k=10)
        reranked = rerank(query, chunks, top_k=5)

        print(f"\n{'Before re-ranking (RRF order)':^65}")
        print(f"{'─'*65}")
        for c in chunks[:10]:
            print(
                f"  RRF={c['score']:.5f}  "
                f"chunk #{c['chunk_num']:<4}  "
                f"{c['text'][:80].strip()}..."
            )

        print(f"\n{'After re-ranking (cross-encoder order)':^65}")
        print(f"{'─'*65}")
        for c in reranked:
            rank_change = ""
            orig_rank   = next(
                (i+1 for i, ch in enumerate(chunks[:10])
                 if ch["chunk_num"] == c["chunk_num"]), "?"
            )
            new_rank = c["rerank_rank"]
            if isinstance(orig_rank, int):
                diff = orig_rank - new_rank
                if diff > 0:
                    rank_change = f"↑{diff}"
                elif diff < 0:
                    rank_change = f"↓{abs(diff)}"
                else:
                    rank_change = "─"

            print(
                f"  rerank={c['rerank_score']:+.4f}  "
                f"[was #{orig_rank} → now #{new_rank}] {rank_change:<4}  "
                f"chunk #{c['chunk_num']:<4}  "
                f"{c['text'][:70].strip()}..."
            )