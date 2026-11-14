import sys; sys.path.insert(0,"src")
import chromadb
c = chromadb.PersistentClient(path="storage_v2_copy/chroma").get_collection("legal_cases_qwen")
r = c.get(where={"source_file": {"$eq": "A_K_Gopalan_vs_The_State_Of_Madras_Union_Of_India__on_19_May_1950_1.PDF"}},
          include=["documents","metadatas"])
for i,(d,m) in enumerate(zip(r["documents"], r["metadatas"])):
    if "preventive" in d.lower() and "article 21" in d.lower():
        print(m["opinion"], "|", d[:200].replace("\n"," "), "\n")