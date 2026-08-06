# src/bm25_index.py
# ─────────────────────────────────────────────────────────────
# BM25 INDEX — keyword search over all chunks
#
# What this file does:
#   1. Loads all chunk texts from ChromaDB
#   2. Tokenises them (lowercase, remove punctuation, stopwords)
#   3. Builds a BM25 index from those tokens
#   4. Persists the index to disk with pickle
#   5. Provides a search() function that returns ranked chunk IDs
#
# WHY BM25?
#   BM25 (Best Match 25) is the ranking function behind most
#   traditional search engines. It scores documents by:
#     - Term frequency (TF): how often the query term appears in the chunk
#     - Inverse document frequency (IDF): how rare the term is across all chunks
#       (rare terms like "AIR 1950 SC 27" get higher weight than common words)
#     - Length normalisation: penalises very long chunks that match by chance
#
#   This makes BM25 excellent at exact legal references, citation numbers,
#   statute names, and proper nouns — exactly what semantic search misses.
# ─────────────────────────────────────────────────────────────

import os
import re
import pickle
import chromadb
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH  = "storage/chroma_db"
COLLECTION   = "legal_cases"
BM25_PATH    = "storage/bm25_index.pkl"   # where we persist the index

# Common English + legal stopwords to strip before tokenising
# Keeping legal terms like "court", "act", "section" — they carry meaning
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall",
    "that", "this", "these", "those", "it", "its", "they", "their",
    "he", "she", "we", "you", "i", "me", "him", "her", "us", "them",
    "as", "if", "not", "no", "so", "than", "then", "when", "where",
    "which", "who", "whom", "what", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "into",
    "through", "during", "before", "after", "above", "below", "up",
    "down", "out", "off", "over", "under", "again", "further",
}


# ── Tokeniser ────────────────────────────────────────────────
def tokenise(text: str) -> list[str]:
    """
    Convert raw text into a list of meaningful tokens.

    Steps:
      1. Lowercase everything
      2. Keep letters, digits, and hyphens (hyphens matter in legal terms)
      3. Split on whitespace
      4. Remove stopwords and very short tokens

    WHY KEEP DIGITS?
    Legal citations like "AIR 1950 SC 27" and statute sections like
    "Section 302" contain numbers that are highly discriminative.
    Stripping digits would destroy BM25's advantage for exact citation search.

    WHY KEEP HYPHENS?
    Legal terms like "sub-section", "18-B", "writ-petition" are compound
    words where the hyphen is meaningful.
    """
    text   = text.lower()
    text   = re.sub(r"[^a-z0-9\s\-]", " ", text)   # keep letters, digits, hyphens
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


# ── Build the BM25 index from ChromaDB ───────────────────────
def build_index() -> tuple[BM25Okapi, list[str]]:
    """
    Load all chunks from ChromaDB, tokenise them, build BM25 index.

    Returns:
        bm25    : the BM25Okapi index object
        all_ids : list of chunk IDs in the same order as the index
                  (position i in the index corresponds to all_ids[i])

    WHY CHROMADB AS THE SOURCE?
    ChromaDB is the single source of truth for chunk text. Building BM25
    from the same data guarantees the two indices are always in sync.
    If you add new documents and rebuild, both indices update together.

    MEMORY NOTE:
    Loading all chunk texts at once uses more RAM than streaming,
    but BM25Okapi requires the full corpus to calculate IDF.
    For 319 chunks this is negligible. At 100K+ chunks you'd want
    to consider approximate BM25 implementations.
    """
    chroma     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_collection(name=COLLECTION)

    print(f"  Loading {collection.count()} chunks from ChromaDB...")

    # Fetch in batches to avoid OOM (same lesson learned in V2)
    BATCH      = 100
    all_ids    = []
    all_texts  = []
    offset     = 0

    while True:
        batch = collection.get(
            limit   = BATCH,
            offset  = offset,
            include = ["documents"]   # only text — don't load embeddings into RAM
        )
        if not batch["ids"]:
            break
        all_ids.extend(batch["ids"])
        all_texts.extend(batch["documents"])
        offset += BATCH

    print(f"  Loaded {len(all_ids)} chunks")

    # Tokenise every chunk
    print("  Tokenising...")
    tokenised_corpus = [tokenise(text) for text in all_texts]

    # Build BM25 index
    # BM25Okapi is the standard variant with k1=1.5, b=0.75 defaults
    # k1 controls term frequency saturation (higher = more weight on TF)
    # b controls length normalisation (1.0 = full normalisation)
    print("  Building BM25 index...")
    bm25 = BM25Okapi(tokenised_corpus)

    return bm25, all_ids


# ── Persist index to disk ─────────────────────────────────────
def save_index(bm25: BM25Okapi, all_ids: list[str]):
    """
    Pickle the BM25 index and ID list together.

    WHY PICKLE?
    BM25Okapi doesn't have a built-in save/load method.
    Pickle serialises the Python object to bytes.
    It's not human-readable but it's fast and requires no schema.

    WHAT GETS SAVED:
      bm25    : the index object (IDF weights, term frequencies)
      all_ids : the ordered list of chunk IDs
                (critical — position in index must map to chunk ID)
    """
    os.makedirs(os.path.dirname(BM25_PATH), exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": all_ids}, f)
    print(f"  ✓ BM25 index saved to {BM25_PATH}")


# ── Load index from disk ──────────────────────────────────────
def load_index() -> tuple[BM25Okapi, list[str]]:
    """
    Load the persisted BM25 index.
    Called by retriever.py on every search request.
    Fast — pickle load takes <1 second for 319 chunks.
    """
    if not os.path.exists(BM25_PATH):
        raise FileNotFoundError(
            f"BM25 index not found at {BM25_PATH}. "
            "Run: python src/bm25_index.py"
        )
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["ids"]


# ── BM25 search ───────────────────────────────────────────────
def search_bm25(query: str, top_k: int = 20) -> list[dict]:
    """
    Search the BM25 index for a query string.

    Returns a list of dicts sorted by BM25 score descending:
        [{"id": "chunk_id", "bm25_score": 4.21, "rank": 1}, ...]

    WHY TOP 20 (not top 5)?
    In hybrid search, we retrieve more candidates than we'll return
    so the fusion step has enough material to work with.
    Retrieving only top 5 from each system risks missing chunks
    that rank 6th in both systems but would be top 1 after fusion.

    WHY RETURN RANK AS WELL AS SCORE?
    RRF (the fusion method in retriever.py) only uses rank position,
    not the raw BM25 score. Returning both lets retriever.py use
    whichever it needs without re-computing.
    """
    bm25, all_ids = load_index()

    query_tokens = tokenise(query)

    if not query_tokens:
        return []   # empty query after stopword removal → no results

    scores = bm25.get_scores(query_tokens)
    # scores is a numpy array — position i = BM25 score for chunk all_ids[i]

    # Pair each score with its chunk ID
    scored = [
        {"id": all_ids[i], "bm25_score": float(scores[i])}
        for i in range(len(all_ids))
        if scores[i] > 0   # skip chunks with zero score (no term overlap)
    ]

    # Sort by score descending, take top_k, add rank
    scored.sort(key=lambda x: x["bm25_score"], reverse=True)
    scored = scored[:top_k]

    for rank, item in enumerate(scored, 1):
        item["rank"] = rank

    return scored


# ── Build and save ────────────────────────────────────────────
def build_and_save():
    print("\n🔨 Building BM25 index...")
    bm25, all_ids = build_index()
    save_index(bm25, all_ids)
    print(f"  Index covers {len(all_ids)} chunks")


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    # Step 1: build and save
    build_and_save()

    # Step 2: test search
    print("\n── BM25 search tests ──────────────────────────")

    test_queries = [
        "Article 22 preventive detention",   # exact legal reference
        "AIR 1950 SC 27",                    # exact citation — semantic search fails here
        "commission agent property sale",     # keyword matching
        "fundamental rights constitution",    # general legal terms
    ]

    for q in test_queries:
        results = search_bm25(q, top_k=3)
        print(f"\nQuery: {q}")
        for r in results:
            parts     = r["id"].rsplit("__chunk_", 1)
            source    = parts[0] if len(parts) == 2 else r["id"]
            print(f"  [{r['rank']}] score={r['bm25_score']:.4f}  {source}")