# src/ingest_v2.py  (new)
import os, sys, time, argparse
import pdfplumber
import chromadb

from store import (connect, get_status, set_status, record_failure,
                   save_case, get_case, save_chunks)
from chunker import chunk_judgment, embed_text, get_title
from embedder import QwenEmbedder
from metadata_v2 import extract_metadata_strict

BASE_DATASET_DIR = os.path.join("legal_dataset", "supreme_court_judgments")
CHROMA_PATH      = "storage_v2/chroma"
COLLECTION       = "legal_cases_qwen"
SKIP_STATUSES    = {"done", "empty", "failed"}


def extract_text(pdf_path: str) -> str:
    out = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    t = page.extract_text()
                    if t:
                        out.append(t)
                except Exception as e:
                    print(f"  ⚠️  page error (skipped): {e}")
    except Exception as e:
        print(f"  ⚠️  PDF read error: {e}")
    return "\n".join(out)


def chroma_meta(meta: dict, opinion) -> dict:
    keys = ["case_name", "citation", "year", "bench_size", "bench_type", "legal_domain",
            "key_provisions", "outcome", "legal_principle", "petitioner_type", "source_file"]
    m = {k: meta.get(k) if meta.get(k) is not None else "" for k in keys}
    m["opinion"] = opinion or ""
    return m


def run(year=None, source_dir=None, limit=None):
    if source_dir:
        pdf_dir, yr = source_dir, os.path.basename(os.path.normpath(source_dir))
    elif year:
        pdf_dir, yr = os.path.join(BASE_DATASET_DIR, str(year)), str(year)
    else:
        print("Give --year or --dir"); return
    if not os.path.isdir(pdf_dir):
        print(f"Directory not found: {pdf_dir}"); return

    files = sorted(f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf"))
    conn  = connect()
    col   = chromadb.PersistentClient(path=CHROMA_PATH).get_or_create_collection(
                COLLECTION, metadata={"hnsw:space": "cosine"})
    embedder = QwenEmbedder()

    print(f"\n📂 {pdf_dir}  |  {len(files)} files")
    done = skipped = failed = total_chunks = 0
    stopped = False

    for filename in files:
        file_key = f"{yr}/{filename}"
        status = get_status(conn, file_key)
        if status in SKIP_STATUSES:
            skipped += 1
            continue
        if limit and done >= limit:
            break

        print(f"\n📄 [{done + 1}] {filename[:70]}")
        try:
            text = extract_text(os.path.join(pdf_dir, filename))
            if not text.strip():
                set_status(conn, file_key, "empty")
                print("  ⚠️  no extractable text — marked empty")
                continue

            # 1) metadata (reuse if already saved by an earlier, interrupted run)
            meta = get_case(conn, file_key) if status == "metadata_done" else None
            if meta is None:
                print("  ⏳ metadata...")
                meta = extract_metadata_strict(text, filename)
                save_case(conn, file_key, meta)
            print(f"  ✓ {meta['case_name'][:60]} | {meta['legal_domain']}")

            # 2) chunk + embed
            chunks = chunk_judgment(text)
            if not chunks:
                set_status(conn, file_key, "empty")
                print("  ⚠️  no chunks — marked empty")
                continue
            title   = get_title(text)
            vectors = embedder.embed_documents([embed_text(c, title) for c in chunks])

            # 3) write: Chroma (upsert) -> SQLite chunks -> status done
            doc_id = filename.rsplit(".", 1)[0].replace(" ", "_")
            ids    = [f"{doc_id}__chunk_{i}" for i in range(len(chunks))]
            texts  = [c["text"] for c in chunks]
            col.upsert(ids=ids, documents=texts, embeddings=vectors,
                       metadatas=[chroma_meta(meta, c["opinion"]) for c in chunks])
            save_chunks(conn, file_key, ids, texts)
            set_status(conn, file_key, "done", len(chunks))

            done += 1
            total_chunks += len(chunks)
            print(f"  ✓ {len(chunks)} chunks | index total: {col.count()}")

        except RuntimeError as e:
            print(f"\n  🛑 Metadata keys exhausted: {str(e)[:200]}")
            print("     Progress saved. Run again after the quota reset (12:30 PM IST).")
            stopped = True
            break
        except Exception as e:
            failed += 1
            record_failure(conn, file_key, f"{type(e).__name__}: {e}")
            print(f"  ⚠️  {type(e).__name__}: {str(e)[:200]} — counted as failure")
            continue

    total_done = conn.execute("SELECT COUNT(*) FROM files WHERE status='done'").fetchone()[0]
    unfinished = sum(1 for f in files if get_status(conn, f"{yr}/{f}") not in SKIP_STATUSES)
    year_complete = (unfinished == 0)

    print(f"\n{'═'*55}")
    print(f"  Processed this run : {done}")
    print(f"  Skipped (finished) : {skipped}")
    print(f"  Failed this run    : {failed}")
    print(f"  Chunks added       : {total_chunks}")
    print(f"  Chunks in index    : {col.count()}")
    print(f"  Total files ingested (all years) : {total_done}")
    if year_complete:
        print(f"  ✅ Year {yr} COMPLETED")
    else:
        print(f"  ⏳ Year {yr} not complete — {unfinished} files remaining")
        if stopped:
            print("  ⚠️  Stopped early (quota) — rerun after 12:30 PM IST")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=str)
    ap.add_argument("--dir", type=str)
    ap.add_argument("--limit", type=int, help="process at most N files (for testing)")
    a = ap.parse_args()
    run(a.year, a.dir, a.limit)