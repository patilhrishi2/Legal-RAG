# src/ingest.py
# ─────────────────────────────────────────────────────────────
# V2 INDEXING PIPELINE
#
# Changes from V1:
#   - Uses Jina AI for embeddings (faster, higher rate limits)
#   - Calls metadata.py once per document before chunking
#   - Attaches metadata dict to every chunk in ChromaDB
#
# Pipeline:
#   PDF → extract text → extract metadata (Groq)
#       → chunk text → embed chunks (Jina) → store with metadata (ChromaDB)
# ─────────────────────────────────────────────────────────────

import os
import time
import requests
import pdfplumber
import chromadb
from dotenv import load_dotenv
from metadata import extract_metadata

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")

# ── Constants ─────────────────────────────────────────────────
PDF_DIR     = "data/cases"
CHROMA_PATH = "storage/chroma_db"
COLLECTION  = "legal_cases"
CHUNK_SIZE  = 400
OVERLAP     = 50
JINA_MODEL  = "jina-embeddings-v3"
JINA_URL    = "https://api.jina.ai/v1/embeddings"
BATCH_SIZE  = 50    # Jina has no batch size limit but 50 is clean


# ── PDF → text ────────────────────────────────────────────────
def extract_text(pdf_path: str) -> str:
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    return full_text


# ── Text → chunks ─────────────────────────────────────────────
def chunk_text(text: str) -> list[str]:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append(" ".join(words[start:end]))
        start += CHUNK_SIZE - OVERLAP
    return chunks


# ── Chunks → embeddings via Jina AI ──────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type" : "application/json",
        "Accept"       : "application/json",
    }

    all_vectors = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]

        payload = {
            "model"     : JINA_MODEL,
            "task"      : "retrieval.passage",
            "dimensions": 1024,
            "input"     : batch,
        }

        while True:
            response = requests.post(JINA_URL, headers=headers, json=payload)

            if response.status_code == 200:
                data        = response.json()
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                all_vectors.extend([item["embedding"] for item in sorted_data])
                print(f"    embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)} chunks...")
                time.sleep(0.5)
                break

            elif response.status_code == 429:
                print(f"\n  ⏳ Jina rate limit hit at chunk {i+1}. Waiting 60s...")
                time.sleep(60)
                # loop retries the same batch

            else:
                raise RuntimeError(
                    f"Jina API error {response.status_code}: {response.text}"
                )

    return all_vectors


# ── ChromaDB collection ───────────────────────────────────────
def get_collection():
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma.get_or_create_collection(
        name     = COLLECTION,
        metadata = {"hnsw:space": "cosine"}
    )


# ── Ingest one PDF ────────────────────────────────────────────
def ingest_pdf(pdf_path: str, collection) -> int:
    filename = os.path.basename(pdf_path)
    print(f"\n📄 Processing: {filename}")

    # ── Step 1: extract text ──────────────────────────────────
    text = extract_text(pdf_path)
    if not text.strip():
        print("  ⚠️  No extractable text — skipping.")
        return 0
    print(f"  ✓ Extracted {len(text.split()):,} words")

    # ── Step 2: extract metadata (NEW in V2) ─────────────────
    # One Groq call per document — NOT per chunk.
    # Returns a dict with 11 fields (case_name, citation, year, etc.)
    print(f"  ⏳ Extracting metadata via Groq...")
    metadata = extract_metadata(text, filename)
    print(f"  ✓ Metadata: {metadata['case_name']} | {metadata['citation']} | {metadata['legal_domain']}")

    # ── Step 3: chunk ─────────────────────────────────────────
    chunks = chunk_text(text)
    print(f"  ✓ Split into {len(chunks)} chunks")

    # ── Step 4: embed via Jina ────────────────────────────────
    print(f"  ⏳ Embedding {len(chunks)} chunks via Jina...")
    vectors = embed_texts(chunks)
    print(f"  ✓ Got {len(vectors)} vectors (dim={len(vectors[0])})")

    # ── Step 5: store vectors + metadata in ChromaDB ──────────
    # Every chunk of this document gets the same metadata dict.
    # This is how ChromaDB's where-clause filtering works —
    # metadata is stored per chunk, not per document.
    doc_id    = filename.replace(".pdf", "").replace(".PDF", "").replace(" ", "_")
    ids       = [f"{doc_id}__chunk_{i}" for i in range(len(chunks))]
    metadatas = [metadata.copy() for _ in chunks]   # same dict, one copy per chunk

    collection.add(
        documents  = chunks,
        embeddings = vectors,
        ids        = ids,
        metadatas  = metadatas,    # ← the only new argument vs V1
    )
    print(f"  ✓ Stored — collection total: {collection.count()} chunks")
    return len(chunks)


# ── Run all PDFs ──────────────────────────────────────────────
def run_ingestion():
    collection = get_collection()
    pdf_files  = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("No PDFs found in data/cases/ — add PDFs and re-run.")
        return

    print(f"Found {len(pdf_files)} PDF(s)...\n")
    total = 0

    for filename in pdf_files:
        path      = os.path.join(PDF_DIR, filename)
        doc_id    = filename.replace(".pdf", "").replace(".PDF", "").replace(" ", "_")
        first_id  = f"{doc_id}__chunk_0"
        existing  = collection.get(ids=[first_id])

        if existing["ids"]:
            print(f"  ⏭  Skipping {filename} (already indexed)")
            continue

        total += ingest_pdf(path, collection)

    print(f"\n✅ Done. Total chunks in index: {collection.count()}")


if __name__ == "__main__":
    run_ingestion()