# eval/run_eval.py  (run from project root, venv311)
# Usage: python eval/run_eval.py                -> retriever_v2, no rerank + rerank
#        python eval/run_eval.py --db storage_v2_copy
import os, sys, json, argparse, time
sys.path.insert(0, "src")

QFILE = "eval/eval_questions.json"


def norm(s):
    s = s.lower()
    return (s[:-4] if s.endswith(".pdf") else s).rstrip("_")


def score(search_fn, questions, k=10):
    """search_fn(query) -> list of dicts with 'source_file', best first (>= k results)."""
    h1 = h5 = h10 = 0
    rr = 0.0
    misses = []
    for q in questions:
        rel = {norm(x) for x in q["relevant"]}
        results = search_fn(q["query"])
        seen, rank = [], None
        for r in results:                       # collapse chunks to unique cases
            s = norm(r["source_file"])
            if s not in seen:
                seen.append(s)
        for i, s in enumerate(seen[:k], 1):
            if s in rel:
                rank = i
                break
        if rank:
            rr += 1 / rank
            h1 += rank <= 1
            h5 += rank <= 5
            h10 += rank <= 10
        else:
            misses.append(q["query"][:80])
    n = len(questions)
    return {"Hit@1": h1 / n, "Hit@5": h5 / n, "Hit@10": h10 / n, "MRR": rr / n, "misses": misses}


def print_table(rows):
    print(f"\n{'retriever':<28}{'Hit@1':>8}{'Hit@5':>8}{'Hit@10':>8}{'MRR':>8}{'sec':>7}")
    for name, m, sec in rows:
        print(f"{name:<28}{m['Hit@1']:8.2f}{m['Hit@5']:8.2f}{m['Hit@10']:8.2f}{m['MRR']:8.2f}{sec:7.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="storage_v2_copy",
                    help="folder with chroma/ and legal.db (use the snapshot)")
    a = ap.parse_args()

    # point retriever_v2 and store at the chosen folder
    import store, retriever_v2
    store.DB_PATH = f"{a.db}/legal.db"
    retriever_v2.CHROMA_PATH = f"{a.db}/chroma"

    questions = json.load(open(QFILE, encoding="utf-8"))
    print(f"{len(questions)} questions | store: {a.db}")

    rows = []
    for name, rerank in (("v2 hybrid (no rerank)", False), ("v2 hybrid + rerank", True)):
        t0 = time.time()
        # top_k=30 chunks so we can see ~10 distinct cases after collapsing
        fn = lambda q, rr=rerank: retriever_v2.search(q, top_k=30, rerank=rr)
        m = score(fn, questions)
        rows.append((name, m, time.time() - t0))
        print(f"\n{name}: missed {len(m['misses'])}")
        for q in m["misses"][:8]:
            print("   -", q)

    print_table(rows)
    os.makedirs("eval/results", exist_ok=True)
    json.dump([{"name": n, **{k: v for k, v in m.items()}, "sec": s} for n, m, s in rows],
              open(f"eval/results/v2_{int(time.time())}.json", "w"), indent=2)