# src/ingest.py
# V11 — large-scale ingestion support
# Changes:
#   - PDF_DIR configurable via --year argument or SOURCE_DIR env var
#   - Daily RPD counter with automatic pause at 950 requests
#   - Checkpoint file for crash recovery without ChromaDB lookups
#   - Progress summary at end showing estimated days remaining

import os
import sys
import time
import json
import argparse
import pdfplumber
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv
from metadata import extract_metadata

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── Constants ─────────────────────────────────────────────────
BASE_DATASET_DIR = os.path.join("legal_dataset", "supreme_court_judgements")
CHROMA_PATH      = "storage/chroma_db"
COLLECTION       = "legal_cases"
CHUNK_SIZE       = 400
OVERLAP          = 50
EMBED_MODEL      = "gemini-embedding-001"
BATCH_SIZE       = 25

# RPD safety limit — stop at 950 to leave headroom
# (Gemini free tier: 1000 RPD)
RPD_LIMIT        = 950

# Checkpoint file — tracks which files have been successfully ingested
# Faster than querying ChromaDB for every file
CHECKPOINT_FILE  = "storage/ingestion_checkpoint.json"


# ── Checkpoint management ──────────────────────────────────────
def load_checkpoint() -> set:
    """Load set of already-ingested filenames from checkpoint file."""
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
        return set(data.get("ingested", []))
    except Exception:
        return set()


def save_checkpoint(ingested: set):
    """Save checkpoint to disk after each successful file."""
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"ingested": list(ingested), "count": len(ingested)}, f)


# ── PDF → text ────────────────────────────────────────────────
def extract_text(pdf_path: str) -> str:
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    full_text += t + "\n"
    except Exception as e:
        print(f"  ⚠️  PDF read error: {e}")
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


# ── Chunks → embeddings ───────────────────────────────────────
def embed_texts(texts: list[str], rpm_counter: list) -> list[list[float]]:
    """
    Embed chunks with RPD tracking.
    rpm_counter is a mutable list [current_count] shared across calls.
    Returns empty list if RPD limit would be exceeded.
    """
    all_vectors = []

    for i in range(0, len(texts), BATCH_SIZE):
        # Check RPD before each batch
        if rpm_counter[0] >= RPD_LIMIT:
            print(f"\n  🛑 Daily request limit reached ({RPD_LIMIT} requests).")
            print(f"     Stopping ingestion. Run again tomorrow to continue.")
            return []   # signal to stop

        batch = texts[i : i + BATCH_SIZE]

        while True:
            try:
                result = gemini_client.models.embed_content(
                    model    = EMBED_MODEL,
                    contents = batch,
                    config   = types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                )
                all_vectors.extend([e.values for e in result.embeddings])
                rpm_counter[0] += 1
                print(f"    embedded {min(i+BATCH_SIZE, len(texts))}/{len(texts)} "
                      f"chunks [API calls today: {rpm_counter[0]}/{RPD_LIMIT}]",
                      end="\r")
                time.sleep(2)
                break

            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"\n  ⏳ Rate limit — waiting 60s...")
                    time.sleep(60)
                else:
                    raise

    return all_vectors


# ── ChromaDB ──────────────────────────────────────────────────
def get_collection():
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma.get_or_create_collection(
        name     = COLLECTION,
        metadata = {"hnsw:space": "cosine"}
    )


# ── Ingest one PDF ────────────────────────────────────────────
def ingest_pdf(pdf_path: str, collection, rpm_counter: list) -> int:
    """Returns number of chunks added, or -1 if RPD limit hit."""
    filename = os.path.basename(pdf_path)

    text = extract_text(pdf_path)
    if not text.strip():
        print(f"  ⚠️  No text — skipping")
        return 0

    # Metadata extraction (Qwen — separate rate limit, handled in metadata.py)
    meta = extract_metadata(text, filename)

    chunks = chunk_text(text)
    if not chunks:
        return 0

    vectors = embed_texts(chunks, rpm_counter)
    if not vectors:
        return -1   # RPD limit hit — stop processing

    if len(vectors) != len(chunks):
        # Partial embedding — RPD hit mid-file, skip this file
        print(f"\n  ⚠️  Partial embedding — skipping file to maintain consistency")
        return -1

    doc_id    = filename.replace(".pdf","").replace(".PDF","").replace(" ","_")
    ids       = [f"{doc_id}__chunk_{i}" for i in range(len(chunks))]
    metadatas = [meta.copy() for _ in chunks]

    collection.add(
        documents  = chunks,
        embeddings = vectors,
        ids        = ids,
        metadatas  = metadatas,
    )
    return len(chunks)


# ── Main ──────────────────────────────────────────────────────
def run_ingestion(year: str = None, source_dir: str = None):
    """
    Ingest PDFs from a year folder or a custom directory.

    year       : "1950" → ingests from legal_dataset/.../1950/
    source_dir : explicit path override (used for data/cases/ compatibility)
    """
    if source_dir:
        pdf_dir = source_dir
    elif year:
        pdf_dir = os.path.join(BASE_DATASET_DIR, str(year))
    else:
        pdf_dir = "data/cases"   # default fallback

    if not os.path.isdir(pdf_dir):
        print(f"Directory not found: {pdf_dir}")
        return

    pdf_files = sorted([
        f for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    ])

    if not pdf_files:
        print(f"No PDFs found in {pdf_dir}")
        return

    print(f"\n📂 Source     : {pdf_dir}")
    print(f"📄 Files found: {len(pdf_files)}")

    collection    = get_collection()
    ingested_set  = load_checkpoint()
    rpm_counter   = [0]   # mutable — passed by reference
    total_chunks  = 0
    skipped       = 0
    processed     = 0
    limit_hit     = False

    for filename in pdf_files:
        path = os.path.join(pdf_dir, filename)

        # Skip if already ingested (checkpoint)
        if filename in ingested_set:
            skipped += 1
            continue

        print(f"\n📄 [{processed+1}] {filename[:70]}")

        result = ingest_pdf(path, collection, rpm_counter)

        if result == -1:
            limit_hit = True
            print(f"\n  Daily limit reached. Progress saved.")
            break

        if result > 0:
            total_chunks   += result
            ingested_set.add(filename)
            save_checkpoint(ingested_set)
            processed      += 1
            print(f"  ✓ {result} chunks | total indexed: {collection.count()}")
        else:
            # Empty or unreadable file — mark as done so we skip next run
            ingested_set.add(filename)
            save_checkpoint(ingested_set)

    # Rebuild BM25 if anything was added
    if total_chunks > 0:
        print(f"\n🔨 Rebuilding BM25 index...")
        from bm25_index import build_and_save
        build_and_save()

    # Summary
    print(f"\n{'═'*55}")
    print(f"  Session summary")
    print(f"  Files processed this run : {processed}")
    print(f"  Files skipped (done)     : {skipped}")
    print(f"  Chunks added this run    : {total_chunks}")
    print(f"  Total chunks in index   : {collection.count()}")
    print(f"  Total files ingested     : {len(ingested_set)}")
    print(f"  API calls used today     : {rpm_counter[0]}/{RPD_LIMIT}")
    if limit_hit:
        remaining = len(pdf_files) - skipped - processed
        print(f"  Files remaining in year  : {remaining}")
        print(f"  ⚠️  Run again tomorrow to continue")
    print(f"{'═'*55}\n")


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legal RAG ingestion pipeline")
    parser.add_argument("--year",  type=str, help="Year to ingest e.g. 1950")
    parser.add_argument("--dir",   type=str, help="Custom directory path")
    args = parser.parse_args()

    run_ingestion(year=args.year, source_dir=args.dir)