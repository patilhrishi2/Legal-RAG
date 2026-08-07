# Legal Strategy Intelligence System

A Retrieval-Augmented Generation (RAG) system for analysing Supreme Court of India
judgments (1950-2024). Built as a progressive learning project — each version
introduces new RAG concepts on top of the previous one.

**Stack:** Python · Flask · ChromaDB · Gemini Embedding API · Groq (Llama 3.3 70B) · pdfplumber · rank_bm25

---

## Project structure

legal-rag/
├── data/cases/ # PDF judgments (not committed)
├── storage/
│ ├── chroma_db/ # Vector index (not committed)
│ └── bm25_index.pkl # BM25 keyword index (not committed)
├── src/
│ ├── ingest.py # Indexing pipeline (PDF -> vectors + BM25 -> storage)
│ ├── retriever.py # Query pipeline (hybrid search + RRF fusion)
│ ├── metadata.py # LLM-based metadata extraction (Groq)
│ ├── bm25_index.py # BM25 index build, save, load, search
│ └── main.py # Flask API
├── .env # API keys (not committed)
└── requirements.txt


---

## V1 - Basic Case Retrieval Engine

### What it does
The simplest possible RAG pipeline. Indexes legal PDFs as vector embeddings
and retrieves semantically similar chunks for any natural language query.
No filters, no generated answer, no metadata - pure retrieval only.

### Features
- PDF text extraction with pdfplumber
- Fixed-size chunking (400 words, 50-word overlap)
- Embeddings via Gemini embedding-001 (3072 dimensions)
- Vector storage and cosine similarity search with ChromaDB
- Skip logic - already indexed files are not re-processed
- Flask API with /search, /ingest, /status endpoints

### Key concepts learned
- **Embeddings:** each chunk becomes a 3072-float vector encoding semantic meaning
- **Cosine similarity:** measures angle between vectors, not magnitude
- **Chunking with overlap:** prevents sentences from being split across chunk boundaries
- **Two-phase architecture:** indexing (offline, runs once) vs querying (online, per request)
- **RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY:** Gemini uses asymmetric modes - wrong mode hurts recall
- **Score thresholds:** above 0.72 = strong match, 0.65-0.72 = relevant, below 0.65 = weak

### API

GET /status -> index health check
POST /search -> { "query": "...", "top_k": 5 }
POST /ingest -> trigger PDF ingestion pipeline


### Limitations that motivated V2
- No metadata - impossible to filter by court, year, bench type, or legal domain
- All 319 chunks are equal - no structure distinguishing case headers from reasoning
- Naive chunking ignores sentence and paragraph boundaries
- No generated answer - raw chunks returned, user must read manually
- Score alone cannot tell you which case a chunk came from

---

## V2 - Structured Legal Knowledge

### What it does
Attaches structured metadata to every chunk at index time using an LLM.
Enables pre-filtered retrieval - search within a domain, year range, or bench
type before similarity search runs. Adds a /cases endpoint to browse the index.

### New features
- **metadata.py** - new file, Groq Llama 3.3 70B extracts structured metadata once per document
- **11-field metadata schema** purpose-built for SC judgments (see schema below)
- **Pre-filtering** via ChromaDB where-clause - filters applied before similarity search
- **Indian Kanoon citator block detection** - skips citation tables to find actual case content
- **Indian legal bench classification** - Constitution Bench vs Full Bench distinction in prompt
- **TPM-aware batching** - batch size calculated from tokens/minute limit, not request count
- **/cases endpoint** - list and filter all indexed cases by metadata
- Switched back to Gemini embedding-001 from Jina v3 for higher accuracy on legal text

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
| legal_principle | string | "Preventive detention does not..." |
| petitioner_type | string | "Individual" |
| source_file | string | "A_K_Gopalan_vs_...PDF" |

### Key concepts learned
- **Metadata-driven retrieval:** structured fields enable pre-filtering before vector search
- **Pre-filtering vs post-filtering:** where-clause excludes irrelevant chunks before scoring
- **LLM-based extraction:** using a generative model to produce structured data from unstructured text
- **Prompt engineering for domain tasks:** bench classification required Indian legal domain knowledge
- **ChromaDB flat-type constraint:** metadata values must be strings or integers - no lists or dicts
- **PDF structure awareness:** Indian Kanoon PDFs have citator blocks that consume token budgets
- **Embedding dimensions matter:** 3072-dim Gemini outperforms 1024-dim Jina on dense legal text
- **Think in tokens per minute:** rate limits are token-based, not request-based

### API additions

GET /cases -> list all indexed cases
GET /cases?legal_domain=Constitutional Law -> filtered case list
GET /cases?bench_type=Constitution Bench
GET /cases?year_from=1950&year_to=1980
POST /search {
"query": "...",
"filters": {
"legal_domain": "Constitutional Law",
"bench_type": "Constitution Bench",
"year_from": 1950,
"year_to": 1980,
"outcome": "Petition dismissed"
}
}


### Problems encountered and fixed
- Groq misclassifying bench types -> fixed with decision-tree prompt and Indian legal examples
- Indian Kanoon citator block consuming word budget -> fixed with content marker detection
- OpenBLAS OOM on ChromaDB get() -> fixed by fetching IDs only then batching updates
- Jina v3 lower accuracy on legal text -> reverted to Gemini embedding-001
- __pycache__ serving stale code -> must clear before re-ingestion after prompt changes

### Limitations that motivated V3
- Semantic search alone misses exact legal terms - "Article 22" or "AIR 1950 SC 27"
  score poorly if the embedding model treats citation numbers as generic tokens
- No way to catch exact terminology that must appear verbatim
- A single retrieval signal (cosine similarity) is a single point of failure

---

## V3 - Hybrid Search

### What it does
Runs two independent search pipelines on every query - semantic (vector) and
BM25 (keyword) - then fuses the results using Reciprocal Rank Fusion. Each
system catches what the other misses. Chunks that both systems agree on are
ranked highest.

### New features
- **bm25_index.py** - new file, builds and persists a BM25 index over all chunks
- **Dual pipeline retrieval** - semantic and BM25 run in parallel on every query
- **Reciprocal Rank Fusion (RRF)** - rank-based fusion that handles incomparable score scales
- **metadata_matches()** - mirrors ChromaDB where-clause logic for BM25 post-filtering
- **in_both field** - response now shows whether both systems agreed on each result
- **BM25 auto-rebuild** - ingest.py rebuilds BM25 index after new documents are added
- **Candidate pool of 20** - each pipeline retrieves top 20 before fusion (not top 5)

### How BM25 differs from semantic search
| | Semantic Search | BM25 |
|---|---|---|
| Finds | Similar meaning, paraphrasing | Exact keywords, citations, proper nouns |
| Fails on | Exact citations like "AIR 1950 SC 27" | Paraphrasing, different terminology |
| Score basis | Cosine similarity (0-1) | TF-IDF with length normalisation (0-25+) |
| Speed | Requires Gemini API call | Pure Python, sub-millisecond |

### How RRF works

RRF(chunk) = 1/(60 + semantic_rank) + 1/(60 + bm25_rank)

k=60 from Cormack et al. 2009 paper. Dampens top-rank dominance.
Chunks in both lists get two terms added - always beats chunks in one list only.
Raw scores (BM25=24.75, cosine=0.73) are never compared - only rank positions.


### Key concepts learned
- **BM25:** TF x IDF with length normalisation - the algorithm behind traditional search engines
- **IDF:** rare terms like "AIR 1950" get high weight, common terms like "court" get low weight
- **RRF:** rank-based fusion handles incomparable score scales without normalisation
- **in_both=True is the strongest signal:** two independent systems agreeing is more trustworthy than either alone
- **Candidate pool size:** retrieve top 20 from each system so fusion has enough material to work with
- **Pre-filter vs post-filter by system:** ChromaDB pre-filters semantic search natively; BM25 post-filters because the pickle index has no metadata awareness
- **Two stores must stay in sync:** ChromaDB and bm25_index.pkl are separate - ingest.py writes to both

### Storage after V3

storage/
├── chroma_db/ <- vectors + text + metadata (ChromaDB)
└── bm25_index.pkl <- BM25 index + chunk ID position map (pickle)


### API changes

POST /search response now also includes per result:
"semantic_score": 0.7267, <- cosine similarity from pipeline 1
"bm25_score": 6.23, <- BM25 score from pipeline 2 (null if not found by BM25)
"bm25_rank": 3, <- rank within BM25 results (null if not found)
"in_both": true <- whether both pipelines found this chunk
"score": 0.03279 <- RRF score (replaces raw cosine as the primary rank signal)


### Limitations that motivated V4
- System still returns raw text chunks - user must read and interpret them manually
- No generated natural language answer grounded in the retrieved evidence
- No citations in the response - user has to trace back which case each chunk came from
- Retrieved chunks may be individually relevant but lack a synthesised conclusion

---

## Roadmap

| Version | Focus | Status |
|---|---|---|
| V1 | Basic retrieval - embeddings, chunking, vector search | Done |
| V2 | Metadata - extraction, filtering, structured search | Done |
| V3 | Hybrid search - BM25 + vector + RRF fusion | Done |
| V4 | Citation-aware generation - grounded LLM answers | Next |
| V5 | Legal argument extraction - structured reasoning | Planned |
| V6 | Re-ranking - cross-encoder for precision | Planned |
| V7 | Contradiction detection - conflicting judgments | Planned |
| V8 | Legal strategy intelligence - synthesis | Planned |
| V9 | Evaluation framework - precision, recall, faithfulness | Planned |
| V10 | Production architecture - auth, logging, monitoring | Planned |

---

## Setup

```bash
pip install -r requirements.txt
```

Create .env:

GEMINI_API_KEY=your_key
GROQ_API_KEY=your_key


Add PDFs to data/cases/ then:
```bash
python src/ingest.py    # index documents + build BM25
python src/main.py      # start API on localhost:5000
```
