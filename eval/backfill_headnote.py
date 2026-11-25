import sys; sys.path.insert(0, "src")
import store
from headnote import extract

db = sys.argv[1] if len(sys.argv) > 1 else "storage_v2_copy"
store.DB_PATH = f"{db}/legal.db"
c = store.connect()
c.execute("""CREATE TABLE IF NOT EXISTS case_notes(
    source_file TEXT PRIMARY KEY, dissenters TEXT, held_line TEXT)""")
cases = c.execute("""SELECT f.file_key, k.source_file FROM files f
                     JOIN cases k ON k.file_key=f.file_key WHERE f.status='done'""").fetchall()
n = 0
for fk, src in cases:
    txt = " ".join(r[0] for r in c.execute(
        "SELECT text FROM chunks WHERE file_key=? AND chunk_num<8 ORDER BY chunk_num", (fk,)))
    d, held = extract(txt)
    c.execute("INSERT OR REPLACE INTO case_notes VALUES (?,?,?)", (src, "|".join(d), held))
    n += bool(d)
c.commit()
print(f"{len(cases)} cases scanned, {n} with dissenters found")
print(c.execute("SELECT dissenters, held_line FROM case_notes WHERE dissenters!='' LIMIT 5").fetchall())