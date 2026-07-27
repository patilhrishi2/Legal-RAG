# src/retriever.py

import os
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

CHROMA_PATH = "storage/chroma_db"
COLLECTION  = "legal_cases"
EMBED_MODEL = "gemini-embedding-001"   # must match ingest.py exactly


def get_collection():
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma.get_collection(name=COLLECTION)   # get_, not get_or_create_


def embed_query(query_text: str) -> list[float]:
    """
    Embed the user's question.

    RETRIEVAL_QUERY vs RETRIEVAL_DOCUMENT:
    The model was trained asymmetrically. Short questions and long
    legal paragraphs have different structure, so Gemini has separate
    modes for each. Using the wrong mode hurts recall noticeably.
    """
    result = client.models.embed_content(
        model    = EMBED_MODEL,
        contents = [query_text],
        config   = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return result.embeddings[0].values


def search(query_text: str, top_k: int = 5) -> list[dict]:
    """
    Convert query → vector → find closest chunks in ChromaDB.

    ChromaDB returns 'distances' (not scores).
    In cosine space: distance = 1 - similarity
    So we convert: score = 1 - distance
    Score of 1.0 = identical, 0.0 = unrelated.
    """
    collection = get_collection()
    query_vec  = embed_query(query_text)

    raw = collection.query(
        query_embeddings = [query_vec],
        n_results        = top_k,
        include          = ["documents", "distances"]
    )

    results = []
    for doc, dist, chunk_id in zip(
        raw["documents"][0],
        raw["distances"][0],
        raw["ids"][0]
    ):
        score = round(1 - dist, 4)

        # ID format: "Some_Case_Name__chunk_7"
        parts     = chunk_id.rsplit("__chunk_", 1)
        source    = parts[0] if len(parts) == 2 else chunk_id
        chunk_num = int(parts[1]) if len(parts) == 2 else -1

        results.append({
            "text"       : doc,
            "id"         : chunk_id,
            "score"      : score,
            "source_doc" : source,
            "chunk_num"  : chunk_num
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def print_results(query: str, results: list[dict]):
    print(f"\n{'─'*60}")
    print(f"Query : {query}")
    print(f"{'─'*60}")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r['score']}  |  {r['source_doc']}  |  chunk #{r['chunk_num']}")
        print(f"    {r['text'][:300]}...")


if __name__ == "__main__":
    queries = [
        "preventive detention without trial",
        "fundamental rights of the accused",
        "contract obligation between merchant parties",
    ]
    for q in queries:
        results = search(q, top_k=3)
        print_results(q, results)