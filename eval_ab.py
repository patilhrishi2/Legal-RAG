# eval_ab.py  (new file; run from project root in venv311)
# Usage:  python eval_ab.py gen | build | run
import os, sys, json, time, requests
import chromadb
from dotenv import load_dotenv
load_dotenv()

DB, GEM, QWEN, QFILE = "storage/chroma_eval", "legal_cases", "legal_cases_qwen", "eval_questions.json"
FILES = """Karnani_Industrial_Bank_Limited_vs_The_Province_Of_Bengal_And_Others_on_14_May_1951_1.PDF
Keshavan_Madhava_Menon_vs_The_State_Of_Bombay_on_22_January_1951_1.PDF
Kidar_Lall_Seal_And_Another_vs_Hari_Lall_Seal_on_18_December_1951_1.PDF
Kishori_Lal_vs_Sm_Shanti_Devi_on_18_September_1951_1.PDF
Kumar_Pashupathinath_Malia_Another_vs_Deba_Prosanna_Mukherjee_on_4_May_1951_1.PDF
Lakshminarain_Bhadani_vs_C_I_T_Bihar_Orissa_on_11_September_1951_1.PDF
Logendra_Nath_Jha_And_Ors_vs_Shri_Loilal_Biswas_on_24_May_1951_1.PDF
Logendranath_Jha_Others_vs_Shri_Loilal_Biswas_on_24_May_1951_1.PDF
Makhan_Singh_Tarsikka_vs_The_State_Of_Punjab_on_10_December_1951_1.PDF
Manohar_Lal_vs_The_State_on_23_May_1951_1.PDF
Mothey_Anja_Ratna_Raja_Kumar_vs_Koney_Narayana_Rao_And_Others_on_23_November_1951_1.PDF
Mummareddi_Nagi_Reddi_And_Others_vs_Pitti_Durairaja_Naidu_And_Others_on_8_May_1951_1.PDF
Muthuswami_vs_State_Of_Madras_on_22_October_1951_1.PDF
P_D_Shamdasani_vs_Central_Bank_Of_India_Ltd_on_21_December_1951_1.PDF
Pushkar_Singh_vs_State_Of_Madhya_Bharat_And_Anr_on_18_September_1951_1.PDF
R_R_Chari_vs_The_State_Of_Uttar_Pradesh_on_19_March_1951_1.PDF
Rai_Brij_Raj_Krishna_And_Another_vs_S_K_Shaw_And_Brothers_on_2_February_1951_""".split()

def norm(s):  # tolerant filename compare (handles the truncated last name)
    s = s.lower()
    s = s[:-4] if s.endswith(".pdf") else s
    return s.rstrip("_")

def key():
    return os.getenv("GEMINI_API_KEY_1") or os.getenv("GEMINI_API_KEY")

def gem_post(path, payload):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{path}"
    for attempt in range(5):
        r = requests.post(url, json=payload, timeout=60,
                          headers={"x-goog-api-key": key(), "Content-Type": "application/json"})
        if r.status_code == 429:
            print("  rate limited, waiting 35s..."); time.sleep(35); continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Gemini kept rate-limiting")

def all_rows(col, include):
    off = 0
    while True:
        b = col.get(limit=1000, offset=off, include=include)
        if not b["ids"]: return
        yield b
        off += 1000

def chunk_num(cid): return int(cid.rsplit("__chunk_", 1)[1])

# ── gen ───────────────────────────────────────────────────────
def gen():
    col = chromadb.PersistentClient(path=DB).get_collection(GEM)
    targets = {norm(f): [] for f in FILES}
    for b in all_rows(col, ["documents", "metadatas"]):
        for cid, doc, m in zip(b["ids"], b["documents"], b["metadatas"]):
            n = norm(m.get("source_file", ""))
            for t in targets:
                if n.startswith(t): targets[t].append((chunk_num(cid), cid, doc, m["source_file"]))
    out = []
    for t, rows in targets.items():
        if not rows:
            print(f"NOT FOUND in index: {t}"); continue
        rows.sort()
        src = rows[0][3]
        picks = [rows[min(len(rows)-1, max(1, len(rows)//4))], rows[min(len(rows)-1, len(rows)//2)]]
        if picks[0][1] == picks[1][1]: picks = picks[:1]
        # the two Logendra files are the same case: either counts as correct
        relevant = [r for r in {x[3] for v in targets.values() for x in v}
                    if t.startswith("logendra") and norm(r).startswith("logendra")] or [src]
        for _, cid, doc, _ in picks:
            prompt = ("Passage from an Indian Supreme Court judgment:\n\n" + doc[:2500] +
                      "\n\nWrite ONE question a lawyer might ask that this passage helps answer. "
                      "Do NOT mention party names, the case name or dates. Ask about the legal issue or rule. "
                      "10-25 words. Return only the question.")
            j = gem_post("gemini-3.5-flash-lite:generateContent", {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 400}})
            q = j["candidates"][0]["content"]["parts"][0]["text"].strip()
            out.append({"query": q, "relevant": relevant, "from_chunk": cid})
            print(f"- {q}\n    -> {src[:60]}")
    json.dump(out, open(QFILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved {len(out)} questions to {QFILE}. Read/edit it, then run: python eval_ab.py build")

# ── build ─────────────────────────────────────────────────────
def get_model():
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda",
                            model_kwargs={"torch_dtype": torch.float16})
    m.max_seq_length = 768
    return m

def build():
    client = chromadb.PersistentClient(path=DB)
    src = client.get_collection(GEM)
    dst = client.get_or_create_collection(QWEN, metadata={"hnsw:space": "cosine"})
    done = set(dst.get(include=[])["ids"])
    model = get_model()
    print(f"Source {src.count()} chunks, already in Qwen collection: {len(done)}")
    t0, n = time.time(), 0
    for b in all_rows(src, ["documents", "metadatas"]):
        idx = [i for i, c in enumerate(b["ids"]) if c not in done]
        if not idx: continue
        ids = [b["ids"][i] for i in idx]; docs = [b["documents"][i] for i in idx]
        mets = [b["metadatas"][i] for i in idx]
        vecs = model.encode(docs, batch_size=8, normalize_embeddings=True)
        dst.add(ids=ids, documents=docs, metadatas=mets, embeddings=vecs.tolist())
        n += len(ids)
        print(f"  {n} embedded, {n/(time.time()-t0):.1f} chunks/s", end="\r")
    print(f"\nDone. Qwen collection has {dst.count()} chunks.")

# ── run ───────────────────────────────────────────────────────
def score(col, qvecs, qs, k=10):
    hits1 = hits5 = hits10 = 0; rr = 0.0
    for vec, q in zip(qvecs, qs):
        rel = {norm(x) for x in q["relevant"]}
        res = col.query(query_embeddings=[vec], n_results=k, include=["metadatas"])
        rank = next((i for i, m in enumerate(res["metadatas"][0], 1)
                     if norm(m["source_file"]) in rel), None)
        if rank:
            rr += 1 / rank
            hits1 += rank <= 1; hits5 += rank <= 5; hits10 += rank <= 10
    n = len(qs)
    return hits1/n, hits5/n, hits10/n, rr/n

def run():
    qs = json.load(open(QFILE, encoding="utf-8"))
    client = chromadb.PersistentClient(path=DB)
    gcol, qcol = client.get_collection(GEM), client.get_collection(QWEN)
    gvecs = []
    for q in qs:
        j = gem_post("gemini-embedding-001:embedContent", {
            "content": {"parts": [{"text": q["query"]}]}, "taskType": "RETRIEVAL_QUERY"})
        gvecs.append(j["embedding"]["values"])
    model = get_model()
    qvecs = model.encode([q["query"] for q in qs], prompt_name="query",
                         normalize_embeddings=True).tolist()
    print(f"\n{len(qs)} questions, semantic search only (no BM25/rerank)\n")
    print(f"{'model':<22}{'Hit@1':>8}{'Hit@5':>8}{'Hit@10':>8}{'MRR':>8}")
    for name, col, v in (("Gemini 3072d", gcol, gvecs), ("Qwen3-0.6B 1024d", qcol, qvecs)):
        h1, h5, h10, mrr = score(col, v, qs)
        print(f"{name:<22}{h1:8.2f}{h5:8.2f}{h10:8.2f}{mrr:8.2f}")

if __name__ == "__main__":
    {"gen": gen, "build": build, "run": run}[sys.argv[1]]()