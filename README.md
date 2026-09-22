# Legal Strategy Intelligence System

A Retrieval-Augmented Generation (RAG) system for analysing Supreme Court of India
judgments (1950-2024). Built as a progressive learning project — each version
introduces new RAG concepts on top of the previous one.

**Stack:** Python · Flask · ChromaDB · Gemini Embedding API · Gemini 3.5 Flash · Groq (Qwen 3.8 27B) · cross-encoder/ms-marco-MiniLM-L-6-v2 · pdfplumber · rank_bm25 · sentence-transformers

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

## V4 - Citation-Aware RAG Generation

### What it does
Adds a generation layer on top of V3 hybrid retrieval. When the user sets
"generate": true in their request, the retrieved chunks are passed to
Gemini 3.5 Flash with a grounded prompt. The model produces a structured
natural language answer with inline citations, key legal principles, and
a confidence assessment. Raw chunks are always returned alongside the
generated answer so every claim can be verified against the source.

### New features
- **generator.py** - new file, assembles context, builds grounded prompt,
  calls Gemini 3.5 Flash, parses structured response
- **Grounded generation** - LLM is explicitly forbidden from using its own
  training knowledge. Only the retrieved chunks are allowed as source of truth
- **Inline citations** - every factual claim in the answer is tagged with
  (Case Name, Citation) so it is traceable to a source document
- **Structured output** - response always follows: ANSWER / KEY LEGAL
  PRINCIPLES / SOURCES USED / CONFIDENCE
- **Hallucination refusal** - when retrieved chunks do not contain the answer,
  the model says so explicitly rather than generating a confident wrong answer
- **Confidence scoring** - HIGH / MEDIUM / LOW based on how well the
  retrieved chunks support the answer
- **Opt-in generation** - "generate": false (default) returns V3 behaviour
  unchanged. Generation only runs when explicitly requested, preserving
  backward compatibility and avoiding unnecessary latency

### Key concepts learned
- **Grounded generation:** constraining an LLM to a specific context window
  prevents it from mixing retrieved evidence with training knowledge
- **Context assembly:** numbering chunks as [EXCERPT 1], [EXCERPT 2] gives
  the model a stable reference system for citation
- **Prompt engineering for RAG:** system prompt must explicitly forbid
  knowledge outside the context, require citations on every claim, and
  define the exact output structure
- **Hallucination detection via refusal:** a well-prompted model will say
  "the documents do not contain this information" rather than fabricating
  an answer — test this deliberately with out-of-scope queries
- **Temperature for generation:** 0.1 (not 0.0) — zero is too rigid and
  can cause the model to refuse to synthesise; 0.1 allows natural phrasing
  while keeping answers grounded
- **Token budget:** 1500 tokens truncated the answer mid-sentence; 3000
  tokens is the safe minimum for detailed legal answers
- **Lost in the middle:** LLMs attend better to context placed before the
  question than after — context block always precedes the query in the prompt
- **Parse defensively:** LLMs occasionally deviate from output structure
  instructions; the parser must handle missing sections without crashing

### How generation fits into the pipeline

query
- hybrid retrieval (V3, unchanged)
- top-k chunks with scores and metadata
- context assembly (numbered excerpts)
- grounded prompt + system instruction
- Gemini 3.5 Flash
- structured answer with citations

### Hallucination test results
Query: "What did the Supreme Court rule about right to privacy in 2017?"
Index contains only 1950 cases. Expected behaviour: refusal.
Actual behaviour: answer = "The retrieved documents do not contain
sufficient information to answer this question." confidence = LOW.
No 2017 case law fabricated. Grounding is working correctly.

### API changes

POST /search request — new optional field:
"generate": true <- triggers generation (default: false)

POST /search response — new field when generate=true:
"generated": {
"answer" : "grounded answer with inline citations",
"key_legal_principles" : "bullet points with citations",
"sources_used" : "numbered list of cited cases",
"confidence" : "HIGH / MEDIUM / LOW",
"sources" : [ structured list of chunks used as context ],
"error" : null
}

When generate=false:
"generated": null <- field present but null, V3 behaviour preserved

### Problems encountered and fixed
- max_output_tokens=1500 truncated answers mid-sentence -> increased to 3000
- parse_response() missed sections when model omitted colons or varied
  capitalisation -> switched from startswith() to case-insensitive contains
  matching with colon index detection
- 503 UNAVAILABLE from Gemini during high demand -> wrapped in try/except,
  error surfaced in generated.error field without crashing the response
- KEY LEGAL PRINCIPLES populated even for out-of-scope queries -> acceptable
  behaviour; critical thing is ANSWER section correctly refuses

### Limitations that motivate V5
- Legal arguments and reasoning patterns are not explicitly extracted —
  the answer synthesises but does not identify which party made which argument
- All retrieved chunks treated equally — no distinction between majority
  opinion, dissenting opinion, and headnote sections of a judgment
- Single-pass generation — no verification step to check whether the
  generated answer is actually supported by the cited chunks
- No structured extraction of case outcome, winning arguments, or the
  specific legal test applied

## V5 - Legal Argument Extraction

### What it does
Adds a structured extraction layer between retrieval and generation.
Before the LLM generates an answer, each retrieved chunk is sent to
Qwen 3.6 27B on Groq to identify and label the specific legal components
present in it — petitioner arguments, respondent arguments, court
reasoning, legal principles, decision, and key facts. The generator then
receives structured chunks rather than raw text, producing answers that
are precisely argument-aware rather than undifferentiated summaries.

### New features
- **extractor.py** - new file, sends each retrieved chunk to Groq and
  gets back a six-field structured extraction dict
- **Structured context assembly** - generator.py gains a second context
  builder that formats labelled components instead of raw text
- **Argument-specific generation** - when extract=true, the prompt tells
  Gemini which text is a petitioner argument vs court reasoning vs holding
- **Aggregated summary** - summarise_extractions() collects all non-empty
  values of each component type across all chunks into a consolidated view
- **Groq model migration** - llama-3.3-70b-versatile deprecated June 2026,
  migrated to qwen/qwen3.6-27b across metadata.py and extractor.py
- **Four flag combinations** - extract and generate are independently
  opt-in, enabling V3/V4/V5-extract-only/V5-full modes from one endpoint

### The six extracted components
| Field | What it captures |
|---|---|
| petitioner_arguments | Claims made by the party who brought the case |
| respondent_arguments | Counter-arguments made by the opposing party or state |
| court_reasoning | The judges' own analysis and interpretation |
| legal_principles | Specific legal rules or tests stated or applied |
| decision | What the court actually held in this passage |
| key_facts | Factual background relevant to the legal question |

### Key concepts learned
- **Information retrieval vs information extraction:** retrieval finds
  relevant chunks; extraction finds relevant structures within those chunks
- **Query-time vs index-time extraction:** extraction runs on retrieved
  chunks at search time, not during ingestion. This lets you iterate on
  the extraction prompt without re-indexing the entire dataset. The
  tradeoff is cost paid per query instead of per document
- **Structured context improves generation precision:** labelling text
  as "Petitioner argued" vs "Court held" lets the generator answer
  argument-specific queries without mixing argument types
- **Single chunk rarely contains all six components:** most chunks have
  two or three populated fields. Force-filling empty fields degrades
  extraction quality — the prompt explicitly instructs the model to use
  empty string when a component is genuinely absent
- **Enrichment pattern:** extraction adds an "extraction" key to each
  chunk dict rather than replacing anything. Downstream code can use
  raw text, metadata, or structured extraction depending on need
- **Model migration is a production reality:** llama-3.3-70b-versatile
  deprecated without warning. reasoning_effort="none" on Qwen models
  disables thinking mode for structured JSON tasks — reduces latency
  and token usage without affecting extraction quality

### How extraction fits into the pipeline

query
→ hybrid retrieval (V3, unchanged)
→ top-k chunks
→ extract_chunks() — Groq call per chunk
→ enriched chunks with "extraction" key
→ build_structured_context()
→ Gemini generation
→ argument-aware cited answer


### Flag combinations on POST /search

extract=false, generate=false -> V3: raw chunks only
extract=false, generate=true -> V4: raw generation
extract=true, generate=false -> V5: structured chunks, no generation
extract=true, generate=true -> V5 full: extract then generate


### API changes

POST /search request — new optional field:
"extract": true <- triggers argument extraction (default: false)

POST /search response:
"extract_used": true/false <- confirms whether extraction ran

Per result when extract=true:
"extraction": {
"petitioner_arguments" : "...",
"respondent_arguments" : "...",
"court_reasoning" : "...",
"legal_principles" : "...",
"decision" : "...",
"key_facts" : "..."
}

In generated object when both flags true:
"extraction_used": true <- confirms structured context was used


### Problems encountered and fixed
- llama-3.3-70b-versatile returned 404 — deprecated June 17 2026.
  Migrated to qwen/qwen3.6-27b with reasoning_effort="none" for both
  metadata.py and extractor.py
- PowerShell displays nested extraction dicts as empty — display
  limitation only, data is correctly populated in the actual JSON
- CONFIDENCE: UNKNOWN in V4 mode — model omitting or varying the header
  format. Fixed with fallback inference from answer content length and
  refusal pattern detection in parse_response()

### Limitations that motivate V6
- All retrieved chunks are treated equally regardless of relevance score
  — a chunk with score 0.73 and one with score 0.55 both go into the
  context with equal weight
- The top-k chunks from hybrid search may include marginally relevant
  results that dilute the generated answer
- No re-ranking step to push the most precisely relevant chunks to the
  top before extraction and generation run
- Extraction quality is bounded by chunk quality — a 400-word chunk
  that spans multiple argument types produces mixed extractions

## V6 - Re-Ranking

### What it does
Adds a cross-encoder re-ranking step between hybrid retrieval and
extraction. After hybrid search retrieves 20 candidates, a cross-encoder
model scores each one against the query by reading both together in a
single forward pass. The top 5 by re-rank score go forward to extraction
and generation. This improves precision — the context window passed to
the LLM contains the most genuinely relevant chunks, not just the ones
that scored well on vector similarity or keyword overlap.

### New features
- **reranker.py** - new file, loads cross-encoder/ms-marco-MiniLM-L-6-v2
  locally, scores query-chunk pairs, returns sorted results
- **Lazy model loading** - cross-encoder imported inside retriever.search()
  so the 85MB model only loads on first rerank=true request, not on every
  import of retriever.py
- **rrf_score preserved** - both rrf_score and rerank_score returned per
  chunk so callers can see where each chunk came from and how much it moved
- **rerank flag** - independently opt-in alongside extract and generate,
  same pattern as V5

### Why cross-encoder beats bi-encoder for re-ranking

Bi-encoder (retrieval):
query → embed → vector A
chunk → embed → vector B
score = cosine(A, B)
Query and chunk embedded independently, never see each other.
Fast (pre-computed), less precise.

Cross-encoder (re-ranking):
[query + chunk] → model → single relevance score
Query and chunk processed together in one forward pass.
Model attends to every query word while reading every chunk word.
Slower (must run per query), much more precise.

Standard pattern: retrieve top 20 (recall) → re-rank → take top 5 (precision)


### Re-ranking in practice — your actual results
Query: "What are the constitutional limits on preventive detention?"

Before re-ranking (RRF order):
  #1 chunk 147 (rrf=0.031) — general constitutional structure
  #2 chunk 150 (rrf=0.031) — directly lists Article 22 safeguards
  #9 chunk 264 (rrf=0.016) — three-month limit under clause (7)
  #10 chunk 186 (rrf=0.016) — valid law requirements for detention

After re-ranking (cross-encoder order):
  #1 chunk 150 (rerank=+3.39) — Article 22 safeguards ↑1
  #2 chunk 151 (rerank=+2.26) — right to representation — was not in top 5
  #3 chunk 161 (rerank=+1.89) — maximum period limits ↑5
  #4 chunk 186 (rerank=+1.37) — valid law requirements ↑7
  #5 chunk 264 (rerank=+0.91) — three-month clause (7) ↑5

Chunk 151 was RRF rank 14 — never in any previous answer.
The cross-encoder surfaced it because it directly addresses
the right to representation as a constitutional limit.

### Key concepts learned
- **Cross-encoder vs bi-encoder:** bi-encoder embeds independently and
  compares, cross-encoder processes together. The latter is more precise
  because it can attend to every word in both query and chunk simultaneously
- **Retrieve then re-rank:** generous retrieval (top 20) ensures recall;
  precise re-ranking (top 5) ensures the context window is high quality
- **Lazy imports for heavy models:** importing the cross-encoder at module
  level loads 85MB on every import. Importing inside the function that needs
  it means the model only loads when actually used, and stays cached in
  memory for all subsequent calls in the same server process
- **Score field semantics change with flags:** when rerank=false, score=rrf_score.
  When rerank=true, score=rerank_score. Both scores always returned separately
  so consumers can use whichever they need
- **Local model — no API key:** cross-encoder/ms-marco-MiniLM-L-6-v2 runs
  on CPU, downloads once to HuggingFace cache (~85MB), inference ~50ms per
  chunk. Zero ongoing cost regardless of query volume

### API changes

POST /search request — new optional field:
"rerank": true <- triggers re-ranking (default: false)

POST /search response — new fields per result when rerank=true:
"rerank_score" : 3.3926, <- cross-encoder score (+/- range)
"rerank_rank" : 1, <- position after re-ranking
"rrf_score" : 0.0308, <- original hybrid search score (always present)
"score" : 3.3926 <- primary score field (rerank_score when reranked,
rrf_score when not)

"rerank_used": true/false in response root confirms whether re-ranking ran.


### Full pipeline flag combinations

rerank=false, extract=false, generate=false -> V3: hybrid search only
rerank=false, extract=false, generate=true -> V4: raw generation
rerank=false, extract=true, generate=true -> V5: extract + generate
rerank=true, extract=false, generate=false -> V6: re-ranked chunks only
rerank=true, extract=true, generate=true -> V6 full: best quality,
highest latency (~8-10s)


### Problems encountered and fixed
- max_output_tokens=3000 truncated key_legal_principles mid-sentence
  on complex queries → increased to 4000 in generator.py
- cross-encoder loading at module import level caused 1s delay on every
  retriever.py import → moved import inside search() function, lazy-loaded
  on first rerank=true request

### Limitations that motivate V7
- System retrieves and ranks well but treats all cases as independent —
  no awareness that two cases might reach opposite conclusions on the
  same legal question
- A user asking about precedent gets the most relevant chunks but no
  indication that other cases in the index contradict those findings
- Contradictory judgments are a critical feature of legal research —
  knowing that courts have ruled both ways on a question is often more
  valuable than a single confident answer

## V7 - Contradiction Detection

### What it does
Adds a contradiction detection layer that identifies when cases in the
index reach opposite conclusions on the same legal question. After
retrieval and re-ranking, the retrieved chunks are sent to Qwen 3.8 27B
with a structured prompt asking whether any cases contradict each other.
The result is a structured contradiction report returned alongside the
retrieved chunks and generated answer. When contradictions are found,
the generator is instructed to explicitly acknowledge them in the answer
rather than silently favouring one side.

### New features
- **contradiction.py** - new file, sends retrieved chunks to Groq grouped
  by case, asks for structured contradiction analysis, returns a typed
  report with legal question, case positions, overruled flag, and significance
- **Cross-case comparison** - detection compares holdings across cases,
  not argument components within a case (that is extractor.py's job)
- **Short-circuit logic** - if all retrieved chunks are from the same
  case, the Groq call is skipped entirely and a clean explanatory message
  is returned
- **Generator awareness** - when contradictions are detected, the
  contradiction report is injected into the generation prompt so the
  answer explicitly flags conflicts rather than producing a misleading
  single confident answer
- **New Maneka Gandhi case** - added to index as the canonical contradiction
  partner for A.K. Gopalan — these two Constitution Bench cases directly
  contradict on Article 21 interpretation
- **CONTRADICTIONS section** - generator output gains a new labelled
  section between ANSWER and KEY LEGAL PRINCIPLES

### The canonical contradiction detected
Query: "Does Article 21 apply to preventive detention?"

A.K. Gopalan vs State of Madras (1950 SC 27):
  Held Articles 14, 19, and 21 are silos. Article 21 only requires a
  valid procedure — not a fair, just or reasonable one. Article 22
  clauses (4)-(7) form a complete code for preventive detention.

Maneka Gandhi vs Union of India (AIR 1978 SC 597):
  Explicitly overruled Gopalan. Held Article 21 must be read with
  Articles 14 and 19. Procedure must be just, fair and reasonable —
  not merely technically valid. Established the due process doctrine
  in India. overruled=True.

### Key concepts learned
- **Contradiction detection is semantic not syntactic:** two cases can
  contradict without sharing a keyword. Gopalan and Maneka Gandhi barely
  overlap in exact phrasing but are direct contradictions on Article 21.
  Only an LLM reading both can identify the logical conflict
- **Retrieval gates detection:** the system can only detect contradictions
  between cases it retrieved. If only one side of a contradiction is in
  the top-k, the other side is invisible. Better retrieval directly
  improves contradiction coverage
- **Short-circuit on single-case results:** making a Groq call when all
  chunks are from the same case wastes tokens and time. The has_multiple_cases()
  check avoids this entirely
- **Context grouping improves extraction:** chunks are grouped by case
  before being sent to Groq so the model sees each case's complete position
  before comparing across cases
- **Generator must be told about contradictions explicitly:** passing the
  contradiction report into the prompt as a CONTRADICTION REPORT block
  forces the model to acknowledge it. Without this injection, the model
  ignores the contradiction detection result and produces a single-sided answer
- **Legal research requires uncertainty acknowledgement:** a system that
  returns a confident single answer when courts have ruled both ways is
  actively misleading. The contradictions_found field and CONTRADICTIONS
  section in the answer are the mechanism for surfacing this uncertainty

### New case added to index

Maneka Gandhi vs Union Of India
Citation : AIR 1978 SC 597
Year : 1978
Bench : Constitution Bench (7 judges)
Outcome : Petition allowed
Domain : Constitutional Law
Provisions: Article 14, Article 19, Article 21, Passports Act 1967
Principle : Procedure established by law under Article 21 must be just,
fair and reasonable — not arbitrary, capricious or oppressive


### API changes

POST /search request — new optional field:
"detect_contradictions": true <- triggers contradiction detection

POST /search response — new top-level field:
"contradiction_detection": {
"contradictions_found" : true,
"contradiction_count" : 1,
"contradictions" : [
{
"legal_question" : "...",
"case_1" : "A.K. Gopalan... (1950 SC 27)",
"case_1_position" : "...",
"case_2" : "Maneka Gandhi... (AIR 1978 SC 597)",
"case_2_position" : "...",
"significance" : "...",
"overruled" : true
}
],
"analysis_note" : "...",
"error" : null
}

In generated object when generate=true and contradictions found:
"contradictions": "Note: Maneka Gandhi overruled Gopalan on Article 21..."


### Problems encountered and fixed
- qwen/qwen3.6-27b deprecated September 14 2026 → migrated to
  qwen/qwen3.8-27b across metadata.py, extractor.py, contradiction.py
- Qwen 3.8 has 7000 ITPM limit — Maneka Gandhi header exceeded this
  at 5000 words (7614 tokens). Fixed by reducing HEADER_WORD_COUNT to
  1500 words in patch script (~3500 tokens including prompts)
- patch_maneka.py required direct Groq call bypassing metadata.py's
  HEADER_WORD_COUNT constant which Python had cached from old value
- max_output_tokens=4000 truncated complex multi-opinion answers
  mid-sentence → increased to 5000 in generator.py

### Limitations that motivate V8
- Contradiction detection only fires when both contradicting cases are
  in the retrieved top-k — cases not retrieved are invisible to detection
- The system identifies contradictions but does not synthesise a
  strategic recommendation — which position is stronger, which is more
  recent, which applies to the user's specific fact pattern
- No weighting by precedential authority — a Constitution Bench judgment
  and a Division Bench judgment are treated equally in retrieval

## V8 - Legal Strategy Intelligence

### What it does
Adds a synthesis layer that transforms retrieved evidence into actionable
legal strategy. After retrieval, re-ranking, extraction, contradiction
detection, and generation, a new strategist module analyses everything
produced and generates a structured strategic assessment: strength of
position, winning arguments from precedent, failure patterns to avoid,
key decisive factors courts focus on, recommended primary and fallback
arguments, and risk factors to anticipate.

### New features
- **strategist.py** - new file, synthesises all pipeline outputs into
  a structured strategic assessment using Gemini 3.5 Flash
- **build_strategy_context()** - assembles chunks grouped by case,
  extraction results, contradiction report, and generated answer into
  a single rich context block for the strategist
- **Eight-section structured output** - situation summary, strength
  assessment, winning arguments, failure patterns, key decisive factors,
  recommended strategy, risk factors, confidence
- **Honest weakness assessment** - when precedents don't match the
  situation, strength is assessed as WEAK and the mismatch is explicitly
  flagged (demonstrated with the merchant contract situation)
- **strategize flag** - independently opt-in, runs after generation,
  receives the generated answer as additional context

### The strategic assessment sections
| Section | What it answers |
|---|---|
| situation_summary | Brief restatement of the user's legal situation |
| strength_assessment | STRONG / MODERATE / WEAK with one-line reason |
| winning_arguments | Arguments that succeeded in retrieved precedents |
| failure_patterns | Arguments courts rejected — what NOT to lead with |
| key_decisive_factors | What courts focused on in similar cases |
| recommended_strategy | Primary argument + fallback if primary fails |
| risk_factors | Unfavourable precedents, contradictions, missing facts |
| confidence | HIGH / MEDIUM / LOW based on precedent match quality |

### Generator vs Strategist — the key distinction

Generator (V4): "What has the court held on this topic?"
Descriptive — reports what precedents say
Grounded answer with inline citations

Strategist (V8): "Given my situation, what should I argue?"
Prescriptive — recommends what to do
Actionable strategy based on precedent patterns


### Demonstrated results
Situation: "Client detained without grounds, four months, no advisory board review"
  strength_assessment : STRONG
  Primary argument    : Writ petition on Article 22(4)(a) — four months
                        without advisory board is a direct constitutional violation
  Fallback argument   : Article 22(5) non-communication renders detention
                        void ab initio
  Failure pattern     : Do not demand alternative tribunals — Gopalan
                        explicitly rejected this argument
  Risk factor         : Article 22(6) public interest exception (state
                        may justify non-disclosure) + Article 22(7)
                        special parliamentary laws

Situation: "Merchant — buyer refused to complete purchase after delivery arranged"
  strength_assessment : WEAK
  Reason              : Only available precedent is agency commission
                        on property sale — not sale of goods
  Honest note         : "Severe precedent mismatch — court may reject
                        these arguments entirely"

### Key concepts learned
- **Descriptive vs prescriptive generation:** generating what courts
  held (V4) is different from recommending what to argue (V8). Same
  retrieved evidence, fundamentally different synthesis task
- **Honest weakness assessment:** a strategist that manufactures
  optimism from weak precedents is worse than useless. Strength=WEAK
  with explicit mismatch explanation is more valuable than a confident
  wrong strategy
- **Context richness improves synthesis:** the strategist receives
  extracted argument structure (V5), contradiction report (V7), and
  the generated answer (V4) — not just raw chunks. Each earlier stage
  adds signal that improves strategic reasoning
- **temperature=0.2 for synthesis:** slightly higher than generator's
  0.1 — strategy synthesis benefits from a little more reasoning
  flexibility than factual answer generation
- **Gemini for synthesis, Groq for extraction:** Groq (Qwen) handles
  structured JSON extraction tasks (metadata, arguments, contradictions)
  where speed and JSON precision matter. Gemini handles synthesis tasks
  (generation, strategy) where reasoning quality matters more

### API changes

POST /search request — new optional field:
"strategize": true <- triggers strategy synthesis (default: false)

POST /search response — new top-level field:
"strategy": {
"situation_summary" : "Client detained for four months...",
"strength_assessment" : "STRONG — direct Article 22 violation",
"winning_arguments" : "1. Article 22(4)(a)... 2. Article 22(5)...",
"failure_patterns" : "1. Do not demand alternative tribunals...",
"key_decisive_factors": "1. Three-month temporal barrier...",
"recommended_strategy": "Primary: writ petition... Fallback: void...",
"risk_factors" : "1. Article 22(6) exception...",
"confidence" : "HIGH",
"error" : null
}


### Complete pipeline — all stages

POST /search with all flags:
hybrid search (top 20) [V3]
→ re-rank (top 5) [V6]
→ extract arguments [V5]
→ detect contradictions [V7]
→ generate answer [V4]
→ generate strategy [V8]


### Latency profile when all flags enabled

hybrid search ~200ms
re-ranking ~1s
extraction ~15s (5 chunks × 3s each with sleep)
contradiction ~2s
generation ~5s
strategy ~5s
total ~28-30s


### Problems encountered and fixed
- Qwen OTPM rate limit (1000 tokens/min) hit when extraction calls
  ran back-to-back → added time.sleep(3) between chunks in
  extract_chunks() in extractor.py
- import time missing from extractor.py after adding sleep → added
  to imports

### Limitations that motivate V9
- No way to know if the strategy is actually good — no evaluation
  framework exists yet to measure whether recommendations are
  grounded in retrieved evidence or hallucinated
- Retrieval quality and generation quality are both unmeasured —
  we have no precision, recall, or faithfulness scores
- The system produces confident-sounding output but we have no
  quantitative measure of how often it is actually correct

## Roadmap

| Version | Focus | Status |
|---|---|---|
| V1 | Basic retrieval - embeddings, chunking, vector search | Done |
| V2 | Metadata - extraction, filtering, structured search | Done |
| V3 | Hybrid search - BM25 + vector + RRF fusion | Done |
| V4 | Citation-aware generation - grounded LLM answers | Done |
| V5 | Legal argument extraction - structured reasoning | Done |
| V6 | Re-ranking - cross-encoder for precision | Done |
| V7 | Contradiction detection - conflicting judgments | Done |
| V8 | Legal strategy intelligence - synthesis | Done |
| V9 | Evaluation framework - precision, recall, faithfulness | Next |
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
