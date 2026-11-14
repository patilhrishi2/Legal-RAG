import sys
sys.path.insert(0, "v12"); sys.path.insert(0, "src")
import langchain_rag as lr

q = "Does Article 21 apply to preventive detention?"
docs = lr.rerank_all(q, lr.HybridRetriever(k=20).invoke(q))[:5]
for i, d in enumerate(docs, 1):
    print(f"[{i}] opinion={d.metadata.get('opinion') or '-'} | id={d.id}")
    print("    ", d.page_content[:300].replace("\n", " "), "\n")