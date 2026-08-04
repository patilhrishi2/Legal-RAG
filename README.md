# Legal Strategy Intelligence System

A Retrieval-Augmented Generation (RAG) system for analysing Supreme Court of India
judgments (1950–2024). Built as a progressive learning project — each version
introduces new RAG concepts on top of the previous one.

**Stack:** Python · Flask · ChromaDB · Gemini Embedding API · Groq (Llama 3.3 70B) · pdfplumber

---

## Project structure

legal-rag/
├── data/cases/ # PDF judgments (not committed)
├── storage/chroma_db/ # Vector index (not committed)
├── src/
│ ├── ingest.py # Indexing pipeline (PDF → vectors → ChromaDB)
│ ├── retriever.py # Query pipeline (query → vectors → results)
│ ├── metadata.py # LLM-based metadata extraction (Groq)
│ └── main.py # Flask API
├── .env # API keys (not committed)
└── requirements.txt


---

## V1 — Basic Case Retrieval Engine

### What it does
The simplest possible RAG pipeline. Indexes legal PDFs as vector embeddings
and retrieves semantically similar chunks for any natural language query.

### Features
- PDF text extraction with pdfplumber
- Fixed-size chunking (400 words, 50-word overlap)
- Embeddings via Gemini embedding-001 (3072 dimensions)
- Vector storage and cosine similarity search with ChromaDB
- Flask API with `/search`, `/ingest`, `/status` endpoints

### Key concepts learned
- **Embeddings:** each chunk becomes a 3072-float vector encoding semantic meaning
- **Cosine similarity:** measures angle between vectors, not magnitude
- **Chunking with overlap:** prevents sentences from being split across boundaries
- **Two-phase architecture:** indexing (offline, runs once) vs querying (online, per request)
- **RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY:** asymmetric embedding modes improve recall

### API

GET /status → index health check
POST /search → { "query": "...", "top_k": 5 }
POST /ingest → trigger PDF ingestion pipeline


### Limitations that motivated V2
- No metadata — impossible to filter by court, year, or legal domain
- All chunks are equal — no structure distinguishing case header from reasoning
- Naive chunking ignores sentence and paragraph boundaries
- No generated answer — raw chunks returned, user must read manually

---

## V2 — Structured Legal Knowledge

### What it does
Attaches structured metadata to every chunk at index time. Enables
pre-filtered retrieval — search within a domain, year range, or bench type
before similarity search runs.

### New features
- **Metadata extraction** via Groq Llama 3.3 70B — one LLM call per document
- **11-field metadata schema** purpose-built for SC judgments
- **Pre-filtering** via ChromaDB where-clause (more efficient than post-filtering)
- **`/cases` endpoint** — list all indexed cases with metadata
- **Indian Kanoon citator block detection** — skips citation tables to find actual case content
- **Indian legal bench classification** — Constitution Bench vs Full Bench distinction
- Switched to Gemini embedding-001 from Jina v3 for higher accuracy on legal text
- TPM-aware batching — batch size calculated from tokens/minute limit, not request count

### Metadata schema
| Field | Type | Example |
|---|---|---|
| case_name | string | "A.K. Gopalan vs State of Madras" |
| citation | string | "AIR 1950 SC 27" |
| year | integer | 1950 |
| bench_size | integer | 6 |
| bench_type | string | "Constitution Bench" |
| legal_domain | string | "Constitutional Law" |
| key_provisions | string | "Article 21, Article 22" |
| outcome | string | "Petition dismissed" |
| legal_principle | string | "Preventive detention..." |
| petitioner_type | string | "Individual" |
| source_file | string | "A_K_Gopalan_vs_..." |

### Key concepts learned
- **Metadata-driven retrieval:** structured fields enable pre-filtering before vector search
- **Pre-filtering vs post-filtering:** where-clause excludes irrelevant chunks before scoring
- **LLM-based extraction:** using a generative model to produce structured data from unstructured text
- **Prompt engineering for domain tasks:** bench classification required Indian legal domain knowledge in the prompt
- **ChromaDB constraints:** metadata values must be flat types (string/int/float) — no lists or nested objects
- **PDF structure awareness:** Indian Kanoon PDFs have citator blocks that consume token budgets

### API additions

GET /cases → list all indexed cases
GET /cases?legal_domain=Constitutional Law → filtered case list
GET /cases?year_from=1950&year_to=1980
POST /search { "query": "...",
"filters": {
"legal_domain": "Constitutional Law",
"bench_type": "Constitution Bench",
"year_from": 1950,
"year_to": 1980,
"outcome": "Petition dismissed"
}}


### Problems encountered and fixed
- Groq misclassifying bench types → fixed with decision-tree prompt and domain examples
- Indian Kanoon citator block consuming 3000-word header budget → fixed with content marker detection
- OpenBLAS OOM error on ChromaDB get() → fixed by fetching IDs only, then batching updates
- Jina v3 lower accuracy on legal text → reverted to Gemini embedding-001
- __pycache__ serving stale metadata.py → cleared before re-ingestion

### Limitations that motivated V3
- Semantic search alone misses exact legal terms — "Article 22" or "AIR 1950 SC 27"
  retrieve poorly if the embedding model treats them as generic tokens
- BM25 keyword search would catch these exact matches that semantic search misses
- Score fusion between keyword and semantic signals would improve overall recall

---

## Roadmap

| Version | Focus | Status |
|---|---|---|
| V1 | Basic retrieval — embeddings, chunking, vector search | ✅ Done |
| V2 | Metadata — extraction, filtering, structured search | ✅ Done |
| V3 | Hybrid search — BM25 + vector + score fusion | 🔜 Next |
| V4 | Citation-aware generation — grounded LLM answers | ⬜ |
| V5 | Legal argument extraction — structured reasoning | ⬜ |
| V6 | Re-ranking — cross-encoder for precision | ⬜ |
| V7 | Contradiction detection — conflicting judgments | ⬜ |
| V8 | Legal strategy intelligence — synthesis | ⬜ |
| V9 | Evaluation framework — precision, recall, faithfulness | ⬜ |
| V10 | Production architecture — auth, logging, monitoring | ⬜ |

---

## Setup

```bash
pip install -r requirements.txt
```

Create `.env`:

GEMINI_API_KEY=your_key
GROQ_API_KEY=your_key


Add PDFs to `data/cases/`, then:
```bash
python src/ingest.py   # index documents
python src/main.py     # start API
```

