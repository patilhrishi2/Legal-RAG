# src/store.py  (new)
# SQLite layer: per-file status, case metadata, chunk text, FTS5 keyword search.
import os, re, sqlite3, time

DB_PATH = "storage_v2/legal.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    file_key    TEXT PRIMARY KEY,          -- "1951/Name.PDF"
    year        INTEGER,
    filename    TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending | metadata_done | done | empty | failed
    fail_count  INTEGER NOT NULL DEFAULT 0,
    last_error  TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    updated_at  REAL
);
CREATE TABLE IF NOT EXISTS cases (
    file_key        TEXT PRIMARY KEY REFERENCES files(file_key),
    case_name TEXT, citation TEXT, year INTEGER, bench_size INTEGER, bench_type TEXT,
    legal_domain TEXT, key_provisions TEXT, outcome TEXT, legal_principle TEXT,
    petitioner_type TEXT, source_file TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id  TEXT PRIMARY KEY,
    file_key  TEXT NOT NULL,
    chunk_num INTEGER NOT NULL,
    text      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_key);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='rowid', tokenize='porter unicode61');

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
"""

CASE_FIELDS = ["case_name", "citation", "year", "bench_size", "bench_type", "legal_domain",
               "key_provisions", "outcome", "legal_principle", "petitioner_type", "source_file"]

_STOP = {"a","an","the","and","or","of","to","in","on","for","by","with","is","was","are",
         "be","that","this","it","as","at","from","does","do","can","what","when","how","which"}


def connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


# ── per-file status ───────────────────────────────────────────
def get_status(conn, file_key: str) -> str | None:
    r = conn.execute("SELECT status FROM files WHERE file_key=?", (file_key,)).fetchone()
    return r["status"] if r else None

def set_status(conn, file_key: str, status: str, chunk_count: int | None = None):
    year = int(file_key.split("/")[0]) if file_key.split("/")[0].isdigit() else 0
    conn.execute(
        """INSERT INTO files(file_key, year, filename, status, chunk_count, updated_at)
           VALUES (?,?,?,?,?,?)
           ON CONFLICT(file_key) DO UPDATE SET status=excluded.status,
               chunk_count=COALESCE(?, files.chunk_count), updated_at=excluded.updated_at""",
        (file_key, year, file_key.split("/", 1)[-1], status, chunk_count or 0, time.time(), chunk_count))
    conn.commit()

def record_failure(conn, file_key: str, error: str, max_failures: int = 3):
    """Count a failure; after max_failures the file is parked as 'failed' instead of retried forever."""
    set_status(conn, file_key, get_status(conn, file_key) or "pending")
    conn.execute("UPDATE files SET fail_count=fail_count+1, last_error=? WHERE file_key=?",
                 (error[:300], file_key))
    conn.execute("UPDATE files SET status='failed' WHERE file_key=? AND fail_count>=?",
                 (file_key, max_failures))
    conn.commit()


# ── cases ─────────────────────────────────────────────────────
def save_case(conn, file_key: str, meta: dict):
    set_status(conn, file_key, "metadata_done")
    cols = ",".join(CASE_FIELDS)
    conn.execute(f"INSERT OR REPLACE INTO cases(file_key,{cols}) VALUES (?,{','.join('?'*len(CASE_FIELDS))})",
                 (file_key, *[meta.get(f) for f in CASE_FIELDS]))
    conn.commit()

def get_case(conn, file_key: str) -> dict | None:
    r = conn.execute("SELECT * FROM cases WHERE file_key=?", (file_key,)).fetchone()
    return dict(r) if r else None


# ── chunks ────────────────────────────────────────────────────
def save_chunks(conn, file_key: str, chunk_ids: list[str], texts: list[str]):
    """Replace all chunks of a file in one transaction (safe to repeat)."""
    with conn:
        conn.execute("DELETE FROM chunks WHERE file_key=?", (file_key,))
        conn.executemany("INSERT INTO chunks(chunk_id,file_key,chunk_num,text) VALUES (?,?,?,?)",
                         [(cid, file_key, i, t) for i, (cid, t) in enumerate(zip(chunk_ids, texts))])

def get_chunks(conn, ids: list[str]) -> dict[str, dict]:
    out = {}
    for i in range(0, len(ids), 500):
        part = ids[i:i+500]
        q = ",".join("?" * len(part))
        rows = conn.execute(
            f"""SELECT c.chunk_id, c.text, k.* FROM chunks c
                LEFT JOIN cases k ON k.file_key=c.file_key WHERE c.chunk_id IN ({q})""", part)
        for r in rows:
            d = dict(r); out[d["chunk_id"]] = d
    return out


# ── keyword search (replaces pickle BM25) ─────────────────────
def _fts_query(q: str) -> str:
    toks = [t for t in re.findall(r"[A-Za-z0-9]+", q.lower()) if t not in _STOP and len(t) > 1]
    return " OR ".join(f'"{t}"' for t in toks)

def search_fts(conn, query: str, top_k: int = 20, filters: dict | None = None) -> list[dict]:
    match = _fts_query(query)
    if not match:
        return []
    filters = filters or {}
    where, params = ["chunks_fts MATCH ?"], [match]
    for f in ("legal_domain", "bench_type", "outcome", "petitioner_type"):
        if filters.get(f):
            where.append(f"k.{f}=?"); params.append(filters[f])
    if filters.get("year"):      where.append("k.year=?");  params.append(int(filters["year"]))
    if filters.get("year_from"): where.append("k.year>=?"); params.append(int(filters["year_from"]))
    if filters.get("year_to"):   where.append("k.year<=?"); params.append(int(filters["year_to"]))
    sql = f"""SELECT c.chunk_id, bm25(chunks_fts) AS s FROM chunks_fts
              JOIN chunks c ON c.rowid=chunks_fts.rowid
              JOIN cases k ON k.file_key=c.file_key
              WHERE {' AND '.join(where)} ORDER BY s LIMIT ?"""
    rows = conn.execute(sql, params + [top_k]).fetchall()
    return [{"id": r["chunk_id"], "bm25_score": -r["s"], "bm25_rank": i}
            for i, r in enumerate(rows, 1)]


if __name__ == "__main__":
    c = connect()
    fk = "1950/Test_Case.PDF"
    save_case(c, fk, {"case_name": "Test v State", "year": 1950, "legal_domain": "Criminal Law", "source_file": "Test_Case.PDF"})
    save_chunks(c, fk, ["Test_Case__chunk_0", "Test_Case__chunk_1"],
                ["The appellant was detained under the Preventive Detention Act, Article 22.",
                 "The contract for sale of jute was breached by the seller."])
    set_status(c, fk, "done", 2)
    print(search_fts(c, "preventive detention Article 22"))
    print(search_fts(c, "jute sale", filters={"year": 1951}))   # expect []
    save_chunks(c, fk, ["Test_Case__chunk_0"], ["Only one chunk now about detention."])  # re-run safe
    print(search_fts(c, "jute"))                                 # expect [] (old chunk gone from FTS)
    c.close()
    for ext in ("", "-wal", "-shm"):
        if os.path.exists(DB_PATH + ext): os.remove(DB_PATH + ext)
    print("test db removed")