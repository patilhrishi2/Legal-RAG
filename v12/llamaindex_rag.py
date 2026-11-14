# v12/llamaindex_rag.py  (run from project root, venv311)
# python v12/llamaindex_rag.py --eval
# python v12/llamaindex_rag.py --ask "Does Article 21 apply to preventive detention?"
import sys, argparse, json, time
sys.path.insert(0, "src"); sys.path.insert(0, "eval")

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.vector_stores.chroma import ChromaVectorStore

import store
from embedder import QwenEmbedder

DB_DIR = "storage_v2_copy"
store.DB_PATH = f"{DB_DIR}/legal.db"
RRF_K = 60


# 1) Wrap Qwen as a LlamaIndex embedding model
class QwenLI(BaseEmbedding):
    _m: object = None

    def __init__(self):
        super().__init__(model_name="qwen3-0.6b")
        self._m = QwenEmbedder()

    def _get_query_embedding(self, query):  return self._m.embed_query(query)
    def _get_text_embedding(self, text):    return self._m.embed_documents([text])[0]
    async def _aget_query_embedding(self, query): return self._get_query_embedding(query)


# 2) Existing Chroma collection as a VectorStoreIndex (no re-ingestion)
col = chromadb.PersistentClient(path=f"{DB_DIR}/chroma").get_collection("legal_cases_qwen")
index = VectorStoreIndex.from_vector_store(ChromaVectorStore(chroma_collection=col),
                                           embed_model=QwenLI())
vector_retriever = index.as_retriever(similarity_top_k=20)


# 3) Hybrid retriever: LlamaIndex vector search + our FTS5, fused with RRF
class HybridLI(BaseRetriever):
    def _retrieve(self, query_bundle):
        q = query_bundle.query_str
        scores, nodes = {}, {}
        for rank, n in enumerate(vector_retriever.retrieve(q), 1):
            nodes[n.node.node_id] = n
            scores[n.node.node_id] = scores.get(n.node.node_id, 0) + 1 / (RRF_K + rank)

        conn = store.connect()
        try:
            kw = store.search_fts(conn, q, 20)
            rows = store.get_chunks(conn, [r["id"] for r in kw])
        finally:
            conn.close()
        for r in kw:
            cid, row = r["id"], rows.get(r["id"])
            if row is None:
                continue
            if cid not in nodes:
                meta = {k: v for k, v in row.items() if k not in ("text", "chunk_id", "file_key")}
                nodes[cid] = NodeWithScore(node=TextNode(id_=cid, text=row["text"], metadata=meta))
            scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + r["bm25_rank"])

        best = sorted(scores, key=scores.get, reverse=True)[:20]
        return [nodes[c] for c in best]


# 4) Built-in cross-encoder reranker (same model as before)
reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=20, keep_retrieval_score=False)


def search(query, rerank):
    nodes = HybridLI().retrieve(query)
    if rerank:
        nodes = reranker.postprocess_nodes(nodes, query_str=query)
    return nodes


def eval_search(query, rerank):
    return [{"source_file": n.node.metadata.get("source_file", "")} for n in search(query, rerank)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--ask", type=str)
    a = ap.parse_args()

    if a.ask:
        for i, n in enumerate(search(a.ask, True)[:5], 1):
            m = n.node.metadata
            print(f"[{i}] {m.get('case_name')} ({m.get('year')}) opinion={m.get('opinion') or '-'}")
            print("    ", n.node.get_content()[:200].replace("\n", " "), "\n")

    if a.eval:
        import run_eval
        qs = json.load(open("eval/eval_questions.json", encoding="utf-8"))
        rows = []
        for name, rr in (("llamaindex hybrid", False), ("llamaindex hybrid + rerank", True)):
            t0 = time.time()
            m = run_eval.score(lambda q, rr=rr: eval_search(q, rr), qs)
            rows.append((name, m, time.time() - t0))
        run_eval.print_table(rows)
        print("\nBaseline: Hit@1 0.82  Hit@5 0.97  MRR 0.90 (v2 + rerank)")