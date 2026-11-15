# v12/langgraph_flow.py  (run from project root, venv311)
# python v12/langgraph_flow.py --ask "Does Article 21 apply to preventive detention?"
import sys, argparse
from typing import TypedDict, Optional, Any
sys.path.insert(0, "src"); sys.path.insert(0, "v12")

from langgraph.graph import StateGraph, END
import langchain_rag as lr
from guardrails import detect_contradictions_verified as detect_contradictions
from generator import generate_answer
from strategist import generate_strategy


class State(TypedDict, total=False):
    question: str
    chunks: list
    contradictions: dict
    answer: dict
    strategy: dict
    path: list


def to_chunk(d, rank):
    m = d.metadata
    cid = d.id or ""
    return {
        "text": d.page_content, "id": cid, "score": 1.0 / rank, "rrf_score": 1.0 / rank,
        "chunk_num": int(cid.rsplit("__chunk_", 1)[1]) if "__chunk_" in cid else -1,
        "opinion": m.get("opinion", ""),
        **{k: m.get(k, dflt) for k, dflt in [
            ("case_name", "Unknown"), ("citation", "Unknown"), ("year", 0),
            ("bench_type", "Unknown"), ("bench_size", 0), ("legal_domain", "Unknown"),
            ("key_provisions", "Unknown"), ("outcome", "Unknown"),
            ("legal_principle", "Unknown"), ("petitioner_type", "Unknown"),
            ("source_file", "Unknown")]},
    }


# ---- nodes ----
def retrieve(s: State):
    q = s["question"]
    docs = lr.rerank_all(q, lr.HybridRetriever(k=20).invoke(q))[:5]
    return {"chunks": [to_chunk(d, i) for i, d in enumerate(docs, 1)], "path": ["retrieve"]}

def detect(s: State):
    return {"contradictions": detect_contradictions(s["question"], s["chunks"]),
            "path": s["path"] + ["detect"]}

def gen_with_report(s: State):
    r = generate_answer(s["question"], s["chunks"], use_extraction=False,
                        contradiction_report=s["contradictions"])
    return {"answer": r, "path": s["path"] + ["generate(with conflicts)"]}

def gen_plain(s: State):
    r = generate_answer(s["question"], s["chunks"], use_extraction=False)
    return {"answer": r, "path": s["path"] + ["generate(plain)"]}

def strategize(s: State):
    r = generate_strategy(s["question"], s["chunks"],
                          contradiction_report=s.get("contradictions"),
                          generated_answer=s["answer"].get("answer"))
    return {"strategy": r, "path": s["path"] + ["strategize"]}

FAKE_REPORT = {
    "contradictions_found": True,
    "contradiction_count": 1,
    "contradictions": [{
        "legal_question": "Does Article 21 apply to preventive detention?",
        "case_1": "A.K. Gopalan vs State of Madras (1950)",
        "case_1_position": "Article 22 is a self-contained code; Article 21 does not control preventive detention laws.",
        "case_2": "(simulated later case)",
        "case_2_position": "Article 21 applies to preventive detention; the procedure must be fair, just and reasonable.",
        "significance": "SIMULATED conflict, injected to test the graph branch.",
        "overruled": True}],
    "analysis_note": "SIMULATED report for testing the conflict branch.",
    "error": None,
}

def detect_forced(s: State):
    return {"contradictions": FAKE_REPORT, "path": s["path"] + ["detect(FORCED)"]}

# ---- the branch ----
def route(s: State):
    return "conflict" if s["contradictions"].get("contradictions_found") else "plain"


def build(force_conflict: bool = False):
    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("detect", detect_forced if force_conflict else detect)
    g.add_node("gen_with_report", gen_with_report)
    g.add_node("gen_plain", gen_plain)
    g.add_node("strategize", strategize)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "detect")
    g.add_conditional_edges("detect", route, {"conflict": "gen_with_report", "plain": "gen_plain"})
    g.add_edge("gen_with_report", "strategize")
    g.add_edge("gen_plain", "strategize")
    g.add_edge("strategize", END)
    return g.compile()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", required=True)
    ap.add_argument("--force-conflict", action="store_true")
    a = ap.parse_args()
    out = build(a.force_conflict).invoke({"question": a.ask})
    print("\nPATH:", " -> ".join(out["path"]))
    print("\nCONTRADICTIONS:", out["contradictions"].get("contradictions_found"),
          "|", out["contradictions"].get("analysis_note", "")[:120])
    print("\nANSWER:\n", out["answer"].get("answer", out["answer"].get("error")))
    st = out["strategy"]
    print("\nSTRENGTH:", st.get("strength_assessment"), "| CONFIDENCE:", st.get("confidence"))
    print("RECOMMENDED:\n", st.get("recommended_strategy") or st.get("error"))
    print("RISKS:\n", st.get("risk_factors"))