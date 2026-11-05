# eval/gen_questions.py  (run from project root, venv311)
import os, json, random, sqlite3, time, requests
from dotenv import load_dotenv
load_dotenv()

DB, OUT, N = "storage_v2_copy/legal.db", "eval/eval_questions_draft.json", 45
MODEL = "gemini-3.5-flash-lite"
KEY = os.getenv("GEMINI_API_KEY_1") or os.getenv("GEMINI_API_KEY")
random.seed(42)

conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row

# one candidate case per file, only finished ones with enough chunks
cases = conn.execute("""SELECT f.file_key, c.source_file, c.year, c.case_name, f.chunk_count
                        FROM files f JOIN cases c ON c.file_key=f.file_key
                        WHERE f.status='done' AND f.chunk_count>=4""").fetchall()
by_year = {}
for r in cases:
    by_year.setdefault(r["year"], []).append(r)
for v in by_year.values():
    random.shuffle(v)

# round-robin across years so questions are spread evenly
picked, years = [], sorted(by_year)
while len(picked) < N * 2 and any(by_year.values()):      # extra, in case some fail
    for y in years:
        if by_year[y]:
            picked.append(by_year[y].pop())

def ask(passage, case_name):
    prompt = ("Passage from an Indian Supreme Court judgment:\n\n" + passage[:2500] +
              "\n\nWrite ONE question a lawyer might ask that this passage answers. "
              "It must be specific enough that mostly this case would answer it "
              "(mention the particular legal issue or facts), but do NOT mention party names, "
              "the case name or dates. 12-30 words. Return only the question.")
    for _ in range(5):
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
            headers={"x-goog-api-key": KEY, "Content-Type": "application/json"}, timeout=60,
            json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                  "generationConfig": {"temperature": 0.3, "maxOutputTokens": 300}})
        if r.status_code == 429:
            print("  rate limited, waiting 35s"); time.sleep(35); continue
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    raise RuntimeError("rate limited too long")

out = []
for r in picked:
    if len(out) >= N:
        break
    num = random.randint(1, r["chunk_count"] - 1)          # skip chunk 0 (header)
    row = conn.execute("SELECT chunk_id, text FROM chunks WHERE file_key=? AND chunk_num=?",
                       (r["file_key"], num)).fetchone()
    if not row or len(row["text"].split()) < 200:
        continue
    try:
        q = ask(row["text"], r["case_name"])
    except Exception as e:
        print("  skip:", e); continue
    out.append({"query": q, "relevant": [r["source_file"]], "year": r["year"],
                "case_name": r["case_name"], "chunk_id": row["chunk_id"]})
    print(f"[{len(out)}] ({r['year']}) {q}\n      -> {r['case_name'][:60]}")
    time.sleep(4.5)                                         # 15 requests/min limit

json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"\nSaved {len(out)} questions to {OUT}")