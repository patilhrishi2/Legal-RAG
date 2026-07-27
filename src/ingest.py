# src/ingest.py

import os
import time
import pdfplumber
import chromadb
from google import genai                              # ← CHANGED (new SDK)
from google.genai import types                        # ← CHANGED
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# ── Constants ────────────────────────────────────────────────
PDF_DIR     = "data/cases"
CHROMA_PATH = "storage/chroma_db"
COLLECTION  = "legal_cases"
CHUNK_SIZE  = 400
OVERLAP     = 50
EMBED_MODEL = "gemini-embedding-001"   # no "models/" prefix in new SDK


# ── Step 2: PDF → raw text ───────────────────────────────────
def extract_text(pdf_path: str) -> str:
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    return full_text


# ── Step 3: text → overlapping chunks ────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


# ── Step 4: chunks → embeddings via new SDK ──────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    BATCH_SIZE  = 5    # fewer chunks per call = fewer tokens per minute
    all_vectors = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]

        while True:
            try:
                result = client.models.embed_content(
                    model    = EMBED_MODEL,
                    contents = batch,
                    config   = types.EmbedContentConfig(
                        task_type = "RETRIEVAL_DOCUMENT"
                    )
                )
                all_vectors.extend([e.values for e in result.embeddings])
                print(f"    embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)} chunks...", end="\r")
                time.sleep(3)   # 3s gap → ~20 req/min, TPM stays well under 30k
                break

            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"\n  ⏳ Rate limit hit at chunk {i}. Waiting 60s...")
                    time.sleep(60)
                else:
                    raise

    print()
    return all_vectors


# ── Step 5: ChromaDB collection ──────────────────────────────
def get_collection():
    chroma  = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma.get_or_create_collection(
        name     = COLLECTION,
        metadata = {"hnsw:space": "cosine"}
    )


# ── Main ingestion for one PDF ────────────────────────────────
def ingest_pdf(pdf_path: str, collection) -> int:
    filename = os.path.basename(pdf_path)
    print(f"\n📄 Processing: {filename}")

    text = extract_text(pdf_path)
    if not text.strip():
        print("  ⚠️  No extractable text — skipping.")
        return 0
    print(f"  ✓ Extracted {len(text.split())} words")

    chunks = chunk_text(text)
    print(f"  ✓ Split into {len(chunks)} chunks")

    print(f"  ⏳ Embedding {len(chunks)} chunks...")
    vectors = embed_texts(chunks)
    print(f"  ✓ Got {len(vectors)} vectors (dim={len(vectors[0])})")

    doc_id = filename.replace(".pdf", "").replace(".PDF", "").replace(" ", "_")
    ids    = [f"{doc_id}__chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents  = chunks,
        embeddings = vectors,
        ids        = ids
    )
    print(f"  ✓ Stored — collection total: {collection.count()} chunks")
    return len(chunks)


# ── Run all PDFs ─────────────────────────────────────────────
def run_ingestion():
    collection = get_collection()
    pdf_files  = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("No PDFs found in data/cases/")
        return

    print(f"Found {len(pdf_files)} PDF(s)...")
    total = 0

    for filename in pdf_files:
        path       = os.path.join(PDF_DIR, filename)
        doc_id     = filename.replace(".pdf", "").replace(".PDF", "").replace(" ", "_")
        first_id   = f"{doc_id}__chunk_0"
        existing   = collection.get(ids=[first_id])

        if existing["ids"]:
            print(f"  ⏭  Skipping {filename} (already indexed)")
            continue

        total += ingest_pdf(path, collection)

    print(f"\n✅ Done. Total chunks in index: {collection.count()}")


if __name__ == "__main__":
    run_ingestion()