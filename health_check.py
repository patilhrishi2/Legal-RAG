# health_check.py
# Run from project root: python health_check.py
# Checks ingestion health — which files are indexed, metadata quality,
# chunk counts, and which files from the dataset are missing entirely.

import os
import sys
import json
import chromadb
from collections import defaultdict, Counter

sys.path.insert(0, "src")

CHROMA_PATH      = "storage/chroma_db"
COLLECTION       = "legal_cases"
CHECKPOINT_FILE  = "storage/ingestion_checkpoint.json"
BASE_DATASET_DIR = os.path.join("legal_dataset", "supreme_court_judgments")


# ── Load data ─────────────────────────────────────────────────
def load_collection():
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma.get_collection(name=COLLECTION)


def load_checkpoint() -> set:
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, "r") as f:
        data = json.load(f)
    return set(data.get("ingested", []))


def get_all_pdf_files() -> dict[str, list[str]]:
    """
    Returns {year: [filename, ...]} for all PDFs in the dataset directory.
    Only scans year folders that exist.
    """
    result = {}
    if not os.path.isdir(BASE_DATASET_DIR):
        return result
    for entry in sorted(os.listdir(BASE_DATASET_DIR)):
        year_path = os.path.join(BASE_DATASET_DIR, entry)
        if os.path.isdir(year_path) and entry.isdigit():
            pdfs = sorted([
                f for f in os.listdir(year_path)
                if f.lower().endswith(".pdf")
            ])
            if pdfs:
                result[entry] = pdfs
    return result


# ── Analyse indexed data ──────────────────────────────────────
def analyse_index(collection) -> dict:
    """
    Pull all metadata from ChromaDB and build per-file stats.
    Returns dict of source_file → analysis dict.
    """
    print("  Loading all metadata from ChromaDB...", flush=True)
    raw = collection.get(include=["metadatas"])

    if not raw["ids"]:
        return {}

    # Group by source file
    by_file = defaultdict(list)
    for chunk_id, meta in zip(raw["ids"], raw["metadatas"]):
        src = meta.get("source_file", "unknown")
        by_file[src].append({
            "chunk_id": chunk_id,
            "meta"    : meta,
        })

    results = {}
    for source_file, chunks in by_file.items():
        meta_sample = chunks[0]["meta"]

        # Check all chunk IDs are sequential (no gaps)
        chunk_nums = []
        for c in chunks:
            cid = c["chunk_id"]
            if "__chunk_" in cid:
                try:
                    chunk_nums.append(int(cid.rsplit("__chunk_", 1)[1]))
                except ValueError:
                    pass
        chunk_nums.sort()
        expected   = list(range(len(chunk_nums)))
        has_gaps   = chunk_nums != expected

        # Check metadata completeness
        unknown_fields = [
            k for k in [
                "case_name", "citation", "year", "bench_type",
                "legal_domain", "outcome", "legal_principle"
            ]
            if str(meta_sample.get(k, "Unknown")).strip() in
               ["Unknown", "0", "", "unknown"]
        ]

        results[source_file] = {
            "chunk_count"    : len(chunks),
            "has_gaps"       : has_gaps,
            "chunk_nums"     : chunk_nums,
            "case_name"      : meta_sample.get("case_name",       "Unknown"),
            "citation"       : meta_sample.get("citation",        "Unknown"),
            "year"           : meta_sample.get("year",            0),
            "bench_type"     : meta_sample.get("bench_type",      "Unknown"),
            "bench_size"     : meta_sample.get("bench_size",      0),
            "legal_domain"   : meta_sample.get("legal_domain",    "Unknown"),
            "outcome"        : meta_sample.get("outcome",         "Unknown"),
            "legal_principle": meta_sample.get("legal_principle", "Unknown"),
            "unknown_fields" : unknown_fields,
            "metadata_ok"    : len(unknown_fields) == 0,
        }

    return results


# ── Main health check ─────────────────────────────────────────
def run_health_check(
    year_filter   : str  = None,
    show_good     : bool = False,
    show_missing  : bool = True,
    verbose       : bool = False,
):
    print("\n" + "═"*65)
    print("  INGESTION HEALTH CHECK")
    print("═"*65)

    # Load ChromaDB
    try:
        collection = load_collection()
        total_chunks = collection.count()
        print(f"\n  ChromaDB       : ✓ reachable")
        print(f"  Total chunks   : {total_chunks:,}")
    except Exception as e:
        print(f"\n  ❌ ChromaDB unreachable: {e}")
        return

    # Load checkpoint
    checkpoint = load_checkpoint()
    print(f"  Checkpoint     : {len(checkpoint):,} files marked done")

    # Load dataset file tree
    all_pdfs_by_year = get_all_pdf_files()
    total_dataset    = sum(len(v) for v in all_pdfs_by_year.values())
    print(f"  Dataset files  : {total_dataset:,} across "
          f"{len(all_pdfs_by_year)} year folders")

    # Analyse index
    print(f"\n  Analysing index...", flush=True)
    indexed = analyse_index(collection)
    print(f"  Unique files in index: {len(indexed):,}")

    # ── Per-year breakdown ────────────────────────────────────
    years_to_check = (
        [year_filter] if year_filter
        else sorted(all_pdfs_by_year.keys())
    )

    total_files       = 0
    total_indexed     = 0
    total_missing     = 0
    total_bad_meta    = 0
    total_gaps        = 0
    files_bad_meta    = []
    files_with_gaps   = []
    files_missing     = []

    for year in years_to_check:
        year_pdfs = all_pdfs_by_year.get(year, [])
        if not year_pdfs:
            continue

        year_indexed     = 0
        year_missing     = []
        year_bad_meta    = []
        year_gaps        = []

        for filename in year_pdfs:
            total_files += 1

            # Find this file in the index
            # The source_file in metadata matches the PDF filename
            file_data = indexed.get(filename)

            if file_data is None:
                year_missing.append(filename)
                files_missing.append((year, filename))
            else:
                year_indexed += 1
                if not file_data["metadata_ok"]:
                    year_bad_meta.append((filename, file_data["unknown_fields"]))
                    files_bad_meta.append((year, filename, file_data["unknown_fields"]))
                if file_data["has_gaps"]:
                    year_gaps.append((filename, file_data["chunk_nums"]))
                    files_with_gaps.append((year, filename))

        total_indexed  += year_indexed
        total_missing  += len(year_missing)
        total_bad_meta += len(year_bad_meta)
        total_gaps     += len(year_gaps)

        # Print year summary
        status = "✅" if not year_missing and not year_bad_meta and not year_gaps else "⚠️ "
        print(f"\n  {status} {year}  —  "
              f"{year_indexed}/{len(year_pdfs)} indexed  |  "
              f"{len(year_missing)} missing  |  "
              f"{len(year_bad_meta)} bad metadata  |  "
              f"{len(year_gaps)} chunk gaps")

        if verbose and show_good and year_indexed:
            for filename in year_pdfs:
                fd = indexed.get(filename)
                if fd and fd["metadata_ok"] and not fd["has_gaps"]:
                    print(f"      ✓ {filename[:60]:<60}  "
                          f"{fd['chunk_count']:>4} chunks  "
                          f"{fd['case_name'][:40]}")

        if year_missing and show_missing:
            print(f"    Missing ({len(year_missing)}):")
            for f in year_missing:
                in_chk = "✓ in checkpoint" if f in checkpoint else "✗ not in checkpoint"
                print(f"      – {f[:65]}  [{in_chk}]")

        if year_bad_meta:
            print(f"    Bad metadata ({len(year_bad_meta)}):")
            for fname, bad_fields in year_bad_meta:
                print(f"      – {fname[:55]}  unknown: {bad_fields}")

        if year_gaps:
            print(f"    Chunk gaps ({len(year_gaps)}):")
            for fname, nums in year_gaps:
                expected = list(range(len(nums)))
                gap_pos  = [i for i, n in enumerate(nums) if n != i]
                print(f"      – {fname[:55]}  "
                      f"{len(nums)} chunks, gaps at positions {gap_pos[:5]}")

    # ── Overall summary ───────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  SUMMARY")
    print(f"{'─'*65}")
    print(f"  Dataset files total   : {total_files:,}")
    print(f"  Indexed               : {total_indexed:,}  "
          f"({100*total_indexed//max(total_files,1)}%)")
    print(f"  Missing from index    : {total_missing:,}")
    print(f"  Bad metadata          : {total_bad_meta:,}")
    print(f"  Chunk gap issues      : {total_gaps:,}")
    print(f"  Total chunks          : {total_chunks:,}")

    if total_indexed > 0:
        avg_chunks = sum(
            d["chunk_count"] for d in indexed.values()
        ) / len(indexed)
        print(f"  Avg chunks per file   : {avg_chunks:.0f}")

    # Domain breakdown
    domain_counts = Counter(
        d["legal_domain"] for d in indexed.values()
        if d["legal_domain"] not in ("Unknown", "")
    )
    if domain_counts:
        print(f"\n  Legal domain breakdown:")
        for domain, count in domain_counts.most_common():
            bar = "█" * min(30, count // max(1, total_indexed // 30))
            print(f"    {domain:<25} {count:>5}  {bar}")

    # Bench type breakdown
    bench_counts = Counter(
        d["bench_type"] for d in indexed.values()
        if d["bench_type"] not in ("Unknown", "")
    )
    if bench_counts:
        print(f"\n  Bench type breakdown:")
        for bench, count in bench_counts.most_common():
            print(f"    {bench:<25} {count:>5}")

    # Checkpoint vs index consistency
    # Files in checkpoint but not in index — indicates failed ingestion
    indexed_filenames = set(indexed.keys())
    in_chk_not_indexed = [
        f for f in checkpoint
        if f not in indexed_filenames
        and f.endswith(".PDF") or f.endswith(".pdf")
    ]
    if in_chk_not_indexed:
        print(f"\n  ⚠️  In checkpoint but NOT in index ({len(in_chk_not_indexed)}):")
        print(f"     These were marked done but have no vectors — "
              f"remove from checkpoint to re-ingest")
        for f in sorted(in_chk_not_indexed)[:20]:
            print(f"    – {f}")
        if len(in_chk_not_indexed) > 20:
            print(f"    ... and {len(in_chk_not_indexed)-20} more")
        print(f"\n  To fix, run:")
        print(f"    python health_check.py --fix-checkpoint")

    # Overall health signal
    print(f"\n{'─'*65}")
    if total_missing == 0 and total_bad_meta == 0 and total_gaps == 0:
        print(f"  ✅  HEALTHY — all files indexed with clean metadata")
    elif total_missing == 0 and total_bad_meta < 5 and total_gaps == 0:
        print(f"  🟡  MOSTLY HEALTHY — {total_bad_meta} files have partial metadata")
    else:
        issues = []
        if total_missing  > 0: issues.append(f"{total_missing} files missing")
        if total_bad_meta > 0: issues.append(f"{total_bad_meta} bad metadata")
        if total_gaps     > 0: issues.append(f"{total_gaps} chunk gaps")
        print(f"  🔴  ISSUES FOUND — {', '.join(issues)}")

    print(f"{'═'*65}\n")

    return {
        "total_files"   : total_files,
        "total_indexed" : total_indexed,
        "total_missing" : total_missing,
        "total_bad_meta": total_bad_meta,
        "total_gaps"    : total_gaps,
        "missing"       : files_missing,
        "bad_meta"      : files_bad_meta,
        "gaps"          : files_with_gaps,
    }


# ── Fix checkpoint ────────────────────────────────────────────
def fix_checkpoint():
    """
    Remove from checkpoint any file that is not actually in ChromaDB.
    Safe to run at any time — only removes ghost entries.
    """
    print("\n  Fixing checkpoint...")
    try:
        collection = load_collection()
    except Exception as e:
        print(f"  ❌ ChromaDB unreachable: {e}")
        return

    raw        = collection.get(include=[])
    indexed_filenames = set()
    for chunk_id in raw["ids"]:
        if "__chunk_" in chunk_id:
            doc_id = chunk_id.rsplit("__chunk_", 1)[0]
            indexed_filenames.add(doc_id + ".PDF")
            indexed_filenames.add(doc_id + ".pdf")

    checkpoint  = load_checkpoint()
    before      = len(checkpoint)
    clean       = {f for f in checkpoint if f in indexed_filenames}
    removed     = before - len(clean)

    save_checkpoint(clean)
    print(f"  Removed {removed} ghost entries from checkpoint")
    print(f"  Checkpoint now has {len(clean)} entries")


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingestion health check for Legal RAG"
    )
    parser.add_argument(
        "--year",    type=str,
        help="Only check a specific year e.g. 1950"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show all files including healthy ones"
    )
    parser.add_argument(
        "--hide-missing", action="store_true",
        help="Hide missing file lists (cleaner output for large datasets)"
    )
    parser.add_argument(
        "--fix-checkpoint", action="store_true",
        help="Remove checkpoint entries for files not actually in ChromaDB"
    )
    args = parser.parse_args()

    if args.fix_checkpoint:
        fix_checkpoint()
    else:
        run_health_check(
            year_filter  = args.year,
            show_good    = args.verbose,
            show_missing = not args.hide_missing,
            verbose      = args.verbose,
        )