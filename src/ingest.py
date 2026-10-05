# src/ingest.py
# Large-scale ingestion with key rotation, checkpoint, and strict error handling

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
from metadata import extract_metadata_strict

load_dotenv()

# ── Constants ─────────────────────────────────────────────────
BASE_DATASET_DIR = os.path.join("legal_dataset", "supreme_court_judgments")
CHROMA_PATH      = "storage/chroma_db"
COLLECTION       = "legal_cases"
CHUNK_SIZE       = 400
OVERLAP          = 50
EMBED_MODEL      = "gemini-embedding-001"
BATCH_SIZE       = 25
RPD_LIMIT        = 950
CHECKPOINT_FILE  = "storage/ingestion_checkpoint.json"


# ── Gemini key rotation ───────────────────────────────────────
def load_api_keys() -> list[str]:
    keys = []
    i = 1
    while True:
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if not key:
            break
        keys.append(key)
        i += 1
    if not keys:
        fallback = os.getenv("GEMINI_API_KEY")
        if fallback:
            keys.append(fallback)
    if not keys:
        raise RuntimeError("No Gemini API keys found in .env")
    print(f"  Loaded {len(keys)} Gemini key(s)")
    return keys


API_KEYS        = load_api_keys()
current_key_idx = [0]


def get_client():
    return genai.Client(api_key=API_KEYS[current_key_idx[0]])


def rotate_gemini_key() -> bool:
    next_idx = current_key_idx[0] + 1
    if next_idx >= len(API_KEYS):
        return False
    current_key_idx[0] = next_idx
    print(f"\n  🔑 Switching to Gemini key {next_idx + 1}/{len(API_KEYS)}")
    return True


# ── Checkpoint ────────────────────────────────────────────────
def load_checkpoint() -> set:
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
        return set(data.get("ingested", []))
    except Exception:
        return set()


def save_checkpoint(ingested: set):
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"ingested": list(ingested), "count": len(ingested)}, f)


# ── PDF → text ────────────────────────────────────────────────
def extract_text(pdf_path: str) -> str:
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    t = page.extract_text()
                    if t:
                        full_text += t + "\n"
                except Exception:
                    continue   # skip bad pages, don't crash
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
    Returns:
      list of vectors  → success
      []               → RPD limit hit on all keys (stop ingestion)
    Raises on unexpected errors.
    """
    all_vectors = []

    for i in range(0, len(texts), BATCH_SIZE):
        # Check RPD for current key
        if rpm_counter[0] >= RPD_LIMIT:
            if rotate_gemini_key():
                rpm_counter[0] = 0
                print(f"  Continuing with new Gemini key...")
            else:
                print(f"\n  🛑 All Gemini keys exhausted for today.")
                return []

        batch = texts[i : i + BATCH_SIZE]

        while True:
            try:
                client = get_client()
                result = client.models.embed_content(
                    model    = EMBED_MODEL,
                    contents = batch,
                    config   = types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                )
                all_vectors.extend([e.values for e in result.embeddings])
                rpm_counter[0] += 1
                total_calls = rpm_counter[0] + current_key_idx[0] * RPD_LIMIT
                print(
                    f"    embedded {min(i+BATCH_SIZE, len(texts))}/{len(texts)} chunks "
                    f"[key {current_key_idx[0]+1}, calls today: {rpm_counter[0]}/{RPD_LIMIT}, "
                    f"total: {total_calls}]",
                    end="\r"
                )
                time.sleep(2)
                break

            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    if any(x in err.lower() for x in ["day", "quota", "daily"]):
                        # Daily quota hit — rotate key
                        print(f"\n  ⚠️  Gemini daily quota hit")
                        if rotate_gemini_key():
                            rpm_counter[0] = 0
                        else:
                            print(f"  🛑 All Gemini keys exhausted.")
                            return []
                    else:
                        # TPM hit — wait and retry
                        print(f"\n  ⏳ Gemini TPM limit — waiting 60s...")
                        time.sleep(60)
                else:
                    raise   # unexpected error — bubble up

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
    """
    Returns:
       N > 0  → N chunks added successfully
       0      → file was empty or unreadable (skip, mark done)
      -1      → Gemini RPD exhausted (stop ingestion)
    Raises on metadata API errors (Groq quota exhausted).
    """
    filename = os.path.basename(pdf_path)

    # Extract text
    text = extract_text(pdf_path)
    if not text.strip():
        print(f"  ⚠️  No extractable text — skipping")
        return 0

    # Extract metadata — STRICT: raises on API errors
    print(f"  ⏳ Extracting metadata...")
    meta = extract_metadata_strict(text, filename)
    print(f"  ✓ {meta['case_name'][:60]} | {meta['legal_domain']}")

    # Chunk
    chunks = chunk_text(text)
    if not chunks:
        return 0
    print(f"  ✓ {len(chunks)} chunks")

    # Embed
    print(f"  ⏳ Embedding...")
    vectors = embed_texts(chunks, rpm_counter)

    if not vectors:
        return -1   # Gemini keys exhausted

    if len(vectors) != len(chunks):
        # Partial result — keys exhausted mid-file
        print(f"\n  ⚠️  Partial embedding — stopping to maintain consistency")
        return -1

    # Store
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
    if source_dir:
        pdf_dir = source_dir
    elif year:
        pdf_dir = os.path.join(BASE_DATASET_DIR, str(year))
    else:
        pdf_dir = "data/cases"

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

    collection   = get_collection()
    ingested_set = load_checkpoint()
    rpm_counter  = [0]
    total_chunks = 0
    skipped      = 0
    processed    = 0
    limit_hit    = False

    # Bootstrap checkpoint from ChromaDB on first run
    if not ingested_set:
        print("  Bootstrapping checkpoint from ChromaDB...")
        all_ids = collection.get(include=[])["ids"]
        for chunk_id in all_ids:
            parts = chunk_id.rsplit("__chunk_", 1)
            if len(parts) == 2:
                doc_id = parts[0]
                ingested_set.add(doc_id + ".PDF")
                ingested_set.add(doc_id + ".pdf")
        save_checkpoint(ingested_set)
        print(f"  Bootstrapped {len(ingested_set)} entries from ChromaDB")

    for filename in pdf_files:
        path = os.path.join(pdf_dir, filename)

        # Skip if already done
        if filename in ingested_set:
            skipped += 1
            continue

        # Also check ChromaDB directly (handles edge cases)
        doc_id   = filename.replace(".pdf","").replace(".PDF","").replace(" ","_")
        first_id = f"{doc_id}__chunk_0"
        if collection.get(ids=[first_id])["ids"]:
            ingested_set.add(filename)
            save_checkpoint(ingested_set)
            skipped += 1
            continue

        print(f"\n📄 [{processed+1}] {filename[:70]}")

        try:
            result = ingest_pdf(path, collection, rpm_counter)

        except RuntimeError as e:
            # Groq keys exhausted — hard stop
            err = str(e)
            print(f"\n  🛑 Metadata API exhausted: {err}")
            print(f"     Add more GROQ_API_KEY_N keys to .env or wait until tomorrow.")
            print(f"     Progress saved — run again to continue from here.")
            limit_hit = True
            break

        except Exception as e:
            # Unexpected error — skip file, continue
            print(f"\n  ⚠️  Unexpected error: {e}")
            print(f"     Skipping file and continuing...")
            ingested_set.add(filename)
            save_checkpoint(ingested_set)
            continue

        if result == -1:
            # Gemini keys exhausted
            limit_hit = True
            print(f"\n  🛑 Gemini embedding keys exhausted. Progress saved.")
            break

        if result > 0:
            total_chunks  += result
            ingested_set.add(filename)
            save_checkpoint(ingested_set)
            processed     += 1
            print(f"\n  ✓ {result} chunks | total in index: {collection.count()}")
        else:
            # Empty file — mark done, skip
            ingested_set.add(filename)
            save_checkpoint(ingested_set)

    # Rebuild BM25 if anything was added
    if total_chunks > 0:
        print(f"\n🔨 Rebuilding BM25 index...")
        from bm25_index import build_and_save
        build_and_save()

    # Summary
    remaining = len(pdf_files) - skipped - processed
    print(f"\n{'═'*55}")
    print(f"  Session summary")
    print(f"  Files processed this run : {processed}")
    print(f"  Files skipped (done)     : {skipped}")
    print(f"  Chunks added this run    : {total_chunks}")
    print(f"  Total chunks in index    : {collection.count()}")
    print(f"  Total files ingested     : {len(ingested_set)}")
    print(f"  Gemini calls this run    : {rpm_counter[0]}")
    if limit_hit:
        print(f"  Files remaining          : {remaining}")
        print(f"  ⚠️  Run again tomorrow (or with new keys) to continue")
    else:
        print(f"  ✅ Year complete")
    print(f"{'═'*55}\n")


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legal RAG ingestion pipeline")
    parser.add_argument("--year", type=str, help="Year folder to ingest e.g. 1950")
    parser.add_argument("--dir",  type=str, help="Custom directory path")
    args = parser.parse_args()

    run_ingestion(year=args.year, source_dir=args.dir)