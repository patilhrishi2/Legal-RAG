# src/database.py
# ─────────────────────────────────────────────────────────────
# V10 REQUEST LOGGING AND QUERY HISTORY
#
# What this file does:
#   - Creates and manages a SQLite database for request logging
#   - Logs every search request with query, flags, response time,
#     retrieved cases, and timestamp
#   - Provides query history retrieval with filtering
#   - Tracks system statistics (total queries, avg response time,
#     most common queries, most retrieved cases)
#
# WHY SQLITE?
#   SQLite is a file-based database — no server needed, zero config,
#   works on any machine. Perfect for logging on a single-server
#   deployment. If you scale to multiple servers in V14, you'd
#   switch to PostgreSQL (Supabase free tier). For now SQLite
#   is the right tool — simple, reliable, always available.
#
# WHY LOG TO DATABASE RATHER THAN FILES?
#   Log files are write-only. To answer "what are users searching
#   for?" you'd grep through text files. A database lets you query:
#   SELECT query, COUNT(*) FROM logs GROUP BY query ORDER BY COUNT
#   You also get filtering, pagination, and aggregation for free.
# ─────────────────────────────────────────────────────────────

import os
import json
import sqlite3
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("LOG_DB_PATH", "storage/logs.db")


# ── Database initialisation ───────────────────────────────────
def init_db():
    """
    Create the database and tables if they don't exist.
    Called once at server startup from main.py.
    SQLite creates the file automatically if it doesn't exist.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # Main request log table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS request_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            query           TEXT    NOT NULL,
            top_k           INTEGER DEFAULT 5,
            rerank_used     INTEGER DEFAULT 0,
            extract_used    INTEGER DEFAULT 0,
            contradictions  INTEGER DEFAULT 0,
            generate_used   INTEGER DEFAULT 0,
            strategize_used INTEGER DEFAULT 0,
            filters         TEXT    DEFAULT '{}',
            retrieved_cases TEXT    DEFAULT '[]',
            result_count    INTEGER DEFAULT 0,
            response_time_ms INTEGER DEFAULT 0,
            had_error       INTEGER DEFAULT 0,
            error_message   TEXT    DEFAULT ''
        )
    """)

    # Index on timestamp for efficient history queries
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp
        ON request_log (timestamp DESC)
    """)

    # Index on query text for frequency analysis
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_query
        ON request_log (query)
    """)

    conn.commit()
    conn.close()
    print(f"  ✓ Database initialised at {DB_PATH}")


# ── Log a request ─────────────────────────────────────────────
def log_request(
    query           : str,
    top_k           : int   = 5,
    rerank_used     : bool  = False,
    extract_used    : bool  = False,
    contradictions  : bool  = False,
    generate_used   : bool  = False,
    strategize_used : bool  = False,
    filters         : dict  = None,
    retrieved_cases : list  = None,
    result_count    : int   = 0,
    response_time_ms: int   = 0,
    had_error       : bool  = False,
    error_message   : str   = "",
) -> int:
    """
    Log a search request to the database.
    Returns the row ID of the inserted log entry.

    Called at the end of every /search request in main.py,
    whether it succeeded or failed.

    WHY LOG ERRORS TOO?
    Failed requests are the most valuable debugging signal.
    Knowing that "Article 22" queries fail 30% of the time
    (maybe a rate limit pattern) is something you'd never
    see if you only logged successes.
    """
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    cur.execute("""
        INSERT INTO request_log (
            timestamp, query, top_k, rerank_used, extract_used,
            contradictions, generate_used, strategize_used,
            filters, retrieved_cases, result_count,
            response_time_ms, had_error, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        query,
        top_k,
        int(rerank_used),
        int(extract_used),
        int(contradictions),
        int(generate_used),
        int(strategize_used),
        json.dumps(filters or {}),
        json.dumps(retrieved_cases or []),
        result_count,
        response_time_ms,
        int(had_error),
        error_message or "",
    ))

    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


# ── Get query history ─────────────────────────────────────────
def get_history(
    limit       : int  = 20,
    offset      : int  = 0,
    query_filter: str  = None,
    errors_only : bool = False,
) -> dict:
    """
    Return paginated query history with optional filters.

    Parameters:
        limit        : max rows to return (default 20, max 100)
        offset       : pagination offset
        query_filter : filter by query text (case-insensitive contains)
        errors_only  : if True, return only failed requests

    Returns dict with:
        total   : total matching rows (for pagination)
        entries : list of log entry dicts
    """
    limit = min(limit, 100)   # cap at 100 to prevent huge responses

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # return rows as dict-like objects
    cur  = conn.cursor()

    # Build WHERE clause
    conditions = []
    params     = []

    if query_filter:
        conditions.append("LOWER(query) LIKE LOWER(?)")
        params.append(f"%{query_filter}%")

    if errors_only:
        conditions.append("had_error = 1")

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    # Total count for pagination
    cur.execute(f"SELECT COUNT(*) FROM request_log {where}", params)
    total = cur.fetchone()[0]

    # Paginated results
    cur.execute(f"""
        SELECT * FROM request_log
        {where}
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
    """, params + [limit, offset])

    rows = cur.fetchall()
    conn.close()

    entries = []
    for row in rows:
        entry = dict(row)
        # Parse JSON fields back to Python objects
        entry["filters"]         = json.loads(entry["filters"] or "{}")
        entry["retrieved_cases"] = json.loads(entry["retrieved_cases"] or "[]")
        entry["rerank_used"]     = bool(entry["rerank_used"])
        entry["extract_used"]    = bool(entry["extract_used"])
        entry["contradictions"]  = bool(entry["contradictions"])
        entry["generate_used"]   = bool(entry["generate_used"])
        entry["strategize_used"] = bool(entry["strategize_used"])
        entry["had_error"]       = bool(entry["had_error"])
        entries.append(entry)

    return {"total": total, "entries": entries}


# ── Get system statistics ─────────────────────────────────────
def get_stats() -> dict:
    """
    Aggregate statistics over all logged requests.
    Used by the /health endpoint to show system usage.
    """
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # Total requests
    cur.execute("SELECT COUNT(*) FROM request_log")
    total_requests = cur.fetchone()[0]

    if total_requests == 0:
        conn.close()
        return {
            "total_requests"    : 0,
            "total_errors"      : 0,
            "error_rate"        : 0.0,
            "avg_response_ms"   : 0,
            "top_queries"       : [],
            "top_cases"         : [],
            "flag_usage"        : {},
            "requests_today"    : 0,
        }

    # Error count
    cur.execute("SELECT COUNT(*) FROM request_log WHERE had_error = 1")
    total_errors = cur.fetchone()[0]

    # Average response time
    cur.execute("SELECT AVG(response_time_ms) FROM request_log WHERE had_error = 0")
    avg_response = cur.fetchone()[0] or 0

    # Top 5 most common queries
    cur.execute("""
        SELECT query, COUNT(*) as count
        FROM request_log
        GROUP BY LOWER(query)
        ORDER BY count DESC
        LIMIT 5
    """)
    top_queries = [{"query": r[0], "count": r[1]} for r in cur.fetchall()]

    # Flag usage rates
    cur.execute("""
        SELECT
            SUM(rerank_used)     as rerank,
            SUM(extract_used)    as extract,
            SUM(contradictions)  as contradictions,
            SUM(generate_used)   as generate,
            SUM(strategize_used) as strategize
        FROM request_log
    """)
    flags = cur.fetchone()
    flag_usage = {
        "rerank"        : int(flags[0] or 0),
        "extract"       : int(flags[1] or 0),
        "contradictions": int(flags[2] or 0),
        "generate"      : int(flags[3] or 0),
        "strategize"    : int(flags[4] or 0),
    }

    # Requests today
    today = datetime.utcnow().date().isoformat()
    cur.execute("""
        SELECT COUNT(*) FROM request_log
        WHERE timestamp LIKE ?
    """, (f"{today}%",))
    requests_today = cur.fetchone()[0]

    # Top 5 most retrieved cases
    cur.execute("SELECT retrieved_cases FROM request_log WHERE retrieved_cases != '[]'")
    case_counts = {}
    for row in cur.fetchall():
        cases = json.loads(row[0])
        for case in cases:
            # Shorten filename for display
            short = case.replace(".PDF", "").replace(".pdf", "")
            short = short.split("_on_")[0].replace("_", " ").strip()
            case_counts[short] = case_counts.get(short, 0) + 1

    top_cases = sorted(
        [{"case": k, "count": v} for k, v in case_counts.items()],
        key=lambda x: x["count"],
        reverse=True
    )[:5]

    conn.close()

    return {
        "total_requests" : total_requests,
        "total_errors"   : total_errors,
        "error_rate"     : round(total_errors / total_requests, 4),
        "avg_response_ms": round(avg_response),
        "top_queries"    : top_queries,
        "top_cases"      : top_cases,
        "flag_usage"     : flag_usage,
        "requests_today" : requests_today,
    }


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing database.py...")
    init_db()

    # Log a few test entries
    log_request(
        query            = "preventive detention Article 22",
        top_k            = 5,
        rerank_used      = True,
        generate_used    = True,
        retrieved_cases  = ["A_K_Gopalan_vs_...PDF", "Maneka_Gandhi_vs_...PDF"],
        result_count     = 5,
        response_time_ms = 3241,
    )
    log_request(
        query            = "contract breach merchant",
        top_k            = 3,
        rerank_used      = True,
        retrieved_cases  = ["Abdulla_Ahmed_vs_...PDF"],
        result_count     = 3,
        response_time_ms = 1102,
    )
    log_request(
        query            = "fundamental rights Article 21",
        top_k            = 5,
        rerank_used      = True,
        generate_used    = True,
        strategize_used  = True,
        retrieved_cases  = ["Maneka_Gandhi_vs_...PDF"],
        result_count     = 5,
        response_time_ms = 28450,
    )
    log_request(
        query            = "test error query",
        had_error        = True,
        error_message    = "Gemini API 503",
        response_time_ms = 500,
    )

    print("\nHistory (last 10):")
    history = get_history(limit=10)
    print(f"  Total entries: {history['total']}")
    for e in history["entries"]:
        print(f"  [{e['timestamp'][:19]}] {e['query'][:50]} "
              f"| {e['response_time_ms']}ms "
              f"| error={e['had_error']}")

    print("\nStats:")
    stats = get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n✅ database.py working correctly")