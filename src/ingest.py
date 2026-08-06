# src/ingest.py
# V2 — Gemini embeddings, correct TPM-aware batching

import os
import time
import pdfplumber
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv
from metadata import extract_metadata
from bm25_index import build_and_save

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PDF_DIR     = "data/cases"
CHROMA_PATH = "storage/chroma_db"
COLLECTION  = "legal_cases"
CHUNK_SIZE  = 400
OVERLAP     = 50
EMBED_MODEL = "gemini-embedding-001"

# ── Batch sizing logic ────────────────────────────────────────
# Gemini free tier: 30,000 tokens/minute
# Each chunk ≈ 400 words × 1.3 tokens/word ≈ 520 tokens
# Safe batch: 25 chunks × 520 tokens = 13,000 tokens — well under 30K TPM
# This means no rate limit waits for typical cases
BATCH_SIZE = 25


def extract_text(pdf_path: str) -> str:
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                full_text += t + "\n"
    return full_text


def chunk_text(text: str) -> list[str]:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append(" ".join(words[start:end]))
        start += CHUNK_SIZE - OVERLAP
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed with Gemini using TPM-aware batching.

    Why 25 chunks per batch:
      25 chunks × ~520 tokens = ~13,000 tokens per request
      30,000 TPM limit / 13,000 = ~2 batches per minute safely
      With a 2s sleep between batches we stay well clear of limits.

    For the large Gopalan case (273 chunks = 11 batches):
      11 batches × ~3s each = ~33 seconds total. No 60s waits.
    """
    all_vectors = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]

        while True:
            try:
                result = gemini_client.models.embed_content(
                    model    = EMBED_MODEL,
                    contents = batch,
                    config   = types.EmbedContentConfig(
                        task_type = "RETRIEVAL_DOCUMENT"
                    )
                )
                all_vectors.extend([e.values for e in result.embeddings])
                print(f"    embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)} chunks...")
                time.sleep(2)   # 2s between batches → ~13K tokens per 2s = well under 30K TPM
                break

            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"\n  ⏳ Rate limit hit — waiting 60s...")
                    time.sleep(60)
                else:
                    raise

    return all_vectors


def get_collection():
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma.get_or_create_collection(
        name     = COLLECTION,
        metadata = {"hnsw:space": "cosine"}
    )


def ingest_pdf(pdf_path: str, collection) -> int:
    filename = os.path.basename(pdf_path)
    print(f"\n📄 Processing: {filename}")

    text = extract_text(pdf_path)
    if not text.strip():
        print("  ⚠️  No extractable text — skipping.")
        return 0
    print(f"  ✓ Extracted {len(text.split()):,} words")

    print(f"  ⏳ Extracting metadata via Groq...")
    metadata = extract_metadata(text, filename)
    print(f"  ✓ {metadata['case_name']} | {metadata['citation']}")
    print(f"    {metadata['bench_type']} ({metadata['bench_size']} judges) | {metadata['legal_domain']}")

    chunks = chunk_text(text)
    print(f"  ✓ Split into {len(chunks)} chunks")

    print(f"  ⏳ Embedding via Gemini (batch={BATCH_SIZE})...")
    vectors = embed_texts(chunks)
    print(f"  ✓ {len(vectors)} vectors (dim={len(vectors[0])})")

    doc_id    = filename.replace(".pdf","").replace(".PDF","").replace(" ","_")
    ids       = [f"{doc_id}__chunk_{i}" for i in range(len(chunks))]
    metadatas = [metadata.copy() for _ in chunks]

    collection.add(
        documents  = chunks,
        embeddings = vectors,
        ids        = ids,
        metadatas  = metadatas,
    )
    print(f"  ✓ Stored — collection total: {collection.count()} chunks")
    return len(chunks)


def run_ingestion():
    collection = get_collection()
    pdf_files  = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("No PDFs found in data/cases/")
        return

    print(f"Found {len(pdf_files)} PDF(s)...\n")
    total = 0

    for filename in pdf_files:
        path     = os.path.join(PDF_DIR, filename)
        doc_id   = filename.replace(".pdf","").replace(".PDF","").replace(" ","_")
        first_id = f"{doc_id}__chunk_0"

        if collection.get(ids=[first_id])["ids"]:
            print(f"  ⏭  Skipping {filename} (already indexed)")
            continue

        total += ingest_pdf(path, collection)

    print(f"\n✅ Done. Total chunks: {collection.count()}")

    # Only rebuild BM25 if new documents were actually added
    if total > 0:
        print("\n🔨 Rebuilding BM25 index...")
        build_and_save()
    else:
        print("\n ℹ️  No new documents — BM25 index unchanged.")


if __name__ == "__main__":
    run_ingestion()