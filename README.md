# Kubernetes Hybrid Agentic RAG — Local-First, Evaluated

A local-first Retrieval-Augmented Generation (RAG) application for answering questions about Kubernetes documentation. It combines **BM25 keyword retrieval**, **FAISS vector search**, **Reciprocal Rank Fusion (RRF)**, **CrossEncoder reranking**, and a **LangChain/LangGraph tool-calling agent**. A **Streamlit** UI supports both retrieval-first and agentic modes, with linked documentation sources.

> **Project status:** Working research/demo prototype, tested locally on Kubernetes documentation. This is **not** a production deployment. Retrieval relevance labels and LLM judge scores remain provisional; see [Evaluation caveats](#evaluation-caveats).

## Highlights

- Crawled **586 Kubernetes documentation pages** and created **4,130 chunks** for indexing.
- Built sparse **BM25** and dense **FAISS** indexes using local **Ollama `nomic-embed-text`** embeddings.
- Evaluated **BM25/FAISS fusion** using weighted RRF and reranked candidates with **`cross-encoder/ms-marco-MiniLM-L-6-v2`**.
- Implemented a **LangGraph/LangChain** agent, local **Ollama** inference, and a **Streamlit** chat interface with cited sources.
- Ran a **40-question, page-level retrieval benchmark** and a separate **40-question generated-answer evaluation**.
- Performed a **12-answer AI-assisted qualitative audit** against retrieved source excerpts.

## Architecture

```text
Kubernetes documentation (kubernetes.io/docs)
             │
             ▼
     1-fetch-k8s.py
  parsing + chunking + metadata
             │  4,130 chunks
             ▼
     2-build-index.py
       ┌─────┴─────┐
       ▼           ▼
   BM25 index    Ollama embeddings → FAISS index
       │           │
       └─────┬─────┘
             ▼
      Weighted RRF fusion
             ▼
  Local MiniLM CrossEncoder reranking
             ▼
      Relevant source passages
             │
       ┌─────┴──────────────┐
       ▼                    ▼
 Retrieval-first      LangGraph tool-calling
 question answering   agent (hybrid_search)
       └──────────┬─────────┘
                  ▼
          Local Ollama LLM
                  ▼
       Streamlit chat + source links
```

**Execution modes:** Retrieval-first explicitly calls the retriever before generation; agentic mode allows the LLM to choose a search tool. Both were tested through Streamlit with the kube-proxy question. The UI showed **11.0 s** for retrieval-first and **5.0 s** for agentic execution in these *individual* local trials; they are **not** controlled comparative performance benchmarks.

## Stack

| Layer | Implementation |
|---|---|
| Language | Python |
| Document source | Public Kubernetes documentation |
| Keyword retriever | LangChain BM25Retriever (`rank-bm25`) |
| Dense retrieval | FAISS + Ollama `nomic-embed-text` |
| Fusion | RRF, with 30% BM25 / 70% FAISS in the **weighted benchmark** |
| Reranking | `sentence-transformers` CrossEncoder `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Agent orchestration | LangGraph / LangChain tool calling |
| Generation | Ollama Qwen3:4B in the original evaluation; Streamlit also tested with Llama 3.2:3B |
| Local answer judge | Llama 3.2:3B, exploratory, not calibrated |
| Frontend | Streamlit |

**Configuration note:** The weighted benchmark uses a manually implemented RRF variant. The application's `utils/retrieval.py` uses LangChain `EnsembleRetriever`; verify that its weights and candidate/deduplication settings match the benchmark before treating benchmark numbers as production-retriever scores. The original production retriever used `[0.5, 0.5]`; the intended update is `[0.3, 0.7]`.

## Run locally (Windows PowerShell)

**Prerequisites:** Python, a working virtual environment, Ollama installed and running, and sufficient disk/RAM for FAISS and the CrossEncoder model. The scripts require network access initially to fetch Kubernetes documentation and download model weights.

```powershell
# 1. Create and activate a Python environment (if not already configured)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install project dependencies and UI additions
python -m pip install -r requirements.txt
python -m pip install -r requirements-ui.txt
python -m pip install langchain-classic langchain-ollama sentence-transformers rank-bm25

# 3. Download Ollama models (Ollama must be installed)
ollama pull nomic-embed-text
ollama pull qwen3:4b
ollama pull llama3.2:3b

# 4. Create your local environment configuration
Copy-Item .env.example .env

# 5. Build the corpus and indexes (first run only)
python 1-fetch-k8s.py
python 2-build-index.py

# 6. Start the interface
python -m streamlit run streamlit_app.py
```

Streamlit usually opens at `http://localhost:8501`. Generated `chunks/` and `indexes/` are excluded from Git; **new users must build them locally**. The original repository also includes optional/cloud-oriented scripts; the local flow above uses the locally adapted files instead.

**Note:** Package versions have not been locked in this README, and not every combination of new package releases is guaranteed to work. If setup fails, use the versions installed in the working environment or add a tested lockfile.

## Explore the individual stages

| Script | Purpose |
|---|---|
| `1-fetch-k8s.py` | Download Kubernetes documentation and generate JSON chunks |
| `2-build-index.py` | Create BM25 index, FAISS index, and metadata |
| `3-hybrid-search.py` | Interactively inspect hybrid retrieval |
| `4-agent-local.py` | Run the locally adapted LangChain/LangGraph tool-calling agent |
| `streamlit_app.py` | Run retrieval-first or agentic chat with source links |
| `5-evaluate-local.py` | Initial page-level retrieval evaluation |
| `6-compare-retrievers.py` | Compare FAISS, equal-weight RRF, and reranked retrieval |
| `6-compare-retrievers-weighted.py` | Compare FAISS, 30:70 RRF, and reranked retrieval |
| `validate_kubernetes_eval.py` | Verify that labeled pages exist in the index |
| `7-evaluate-generation-local.py` | Generate and save answers, contexts, sources, and timings |
| `8-evaluate-answer-quality-local.py` | Run exploratory local LLM judging of saved answers |
| `9-audit-evaluation.py` | Prepare and report on a qualitative audit |

### How we built it

1. **Ingestion:** Extracted public Kubernetes docs into 4,130 overlapping, page-tagged chunks.
2. **Indexing:** Persisted BM25 and FAISS indexes with page metadata. Documents are embedded using `nomic-embed-text`.
3. **Retrieval:** Queried BM25 and FAISS for candidate chunks and fused ranked lists with RRF.
4. **Reranking:** Scored query/chunk pairs using a MiniLM CrossEncoder; sorted the fusion candidates accordingly.
5. **Agent:** Connected the retriever through a `hybrid_search` tool; the agent can select and repeat search calls.
6. **Frontend:** Added Streamlit with local model selection, retrieval-first mode, optional agentic execution, and source links.
7. **Evaluation:** Validated page IDs, compared retrieval configurations, generated 40 answers, inspected citations/latencies, ran an exploratory judge, and audited selected failures.

## Experimental results — actual local runs

### 40-question retrieval benchmark (page-level, provisional relevance labels)

| Metric | FAISS only | Hybrid 30:70 RRF | Hybrid + CrossEncoder |
|---|---:|---:|---:|
| Precision@5 | 0.1950 | 0.1950 | 0.1950 |
| Recall@5 | **0.8750** | **0.8750** | 0.8625 |
| MRR | 0.7112 | 0.7162 | **0.7550** |
| NDCG@5 | 0.7341 | 0.7434 | **0.7772** |
| Mean retrieval latency | **0.1209 s** | 0.1287 s | 1.7917 s |

With CrossEncoder, **MRR increased by ~6.2%** and **NDCG@5 increased by ~5.9%** relative to FAISS in this experiment; Recall@5 fell slightly. These are **observed differences on this 40-question provisional dataset**, not generalizable claims or proof of superiority on all workloads.

### Early weight-tuning experiment (4 questions)

| Metric | Hybrid 50:50 | Hybrid 30:70 |
|---|---:|---:|
| Precision@5 | 0.2000 | 0.2000 |
| Recall@5 | 0.7500 | 0.8750 |
| MRR | 0.5000 | 0.7083 |
| NDCG@5 | 0.5811 | 0.7217 |

The four questions helped motivate testing heavier dense-retrieval weighting; the final 40-question results are more informative but remain provisional.

### Answer generation and evaluation (40 questions)

| Measure | Local measured result |
|---|---:|
| Completed generation records | 40 / 40 |
| Mean end-to-end answer pipeline latency | **42.68 s** |
| Answers containing URLs | 39 / 40 (97.5%) |
| Answers passing original URL consistency check | 37 / 40 (92.5%) |
| Exploratory faithfulness (Llama 3.2:3B judge) | **0.7400** |
| Exploratory answer relevance (Llama 3.2:3B judge) | **0.7625** |
| Qualitative audit cases | 12 |
| Judge vs AI-assisted audit disagreement, faithfulness (MAE) | 0.305 |
| Judge vs AI-assisted audit disagreement, relevance (MAE) | 0.171 |

**What these do and don't prove:** Source-URL consistency is **not** citation correctness. The judge scores are **not official RAGAS measurements**, and the 12-case comparison used AI-assisted ratings rather than independent human labels. The 40-question generation timing used the earlier Qwen3:4B pipeline; it should **not** be conflated with the newer single-query Streamlit timings.

### Qualitative failures and lessons

- **Control-plane components:** Retrieved relevant pages but produced an incomplete answer — generation/context selection failure.
- **StatefulSet:** Overstated Headless Service creation and PVC retention behavior — prompting/grounding issue.
- **Secrets vs ConfigMaps:** Overstated default encryption of Secrets — factual correctness issue.
- **Example URLs:** The original citation detector treated `http://localhost:8001` and example service URLs as source citations; this requires a citation-specific parser.
- **Local judge:** Returned highly repetitive faithfulness scores (0.8 on 33/40 records), making calibration and manual review important.

## Evaluation caveats

- The 40 questions use **provisional page-level relevance labels**. Validation confirms the labeled page IDs exist; it **does not** establish that the labels are complete or that all other pages are irrelevant. As a result, the absolute Precision@5 values can be misleading.
- The manual 30:70 RRF experiment and the application's `EnsembleRetriever` are not guaranteed to use identical deduplication, candidate ordering, or ranking logic.
- CrossEncoder scoring may improve ranking and increase latency; it does not guarantee higher recall.
- The local Llama judge is exploratory and its outputs have **not** been calibrated against independently collected human judgments.
- ROUGE, BERTScore, and official RAGAS scores are **not claimed** for the local evaluation shown here; reference answers and/or validated evaluation setups are still needed.
- Production features such as authentication, observability, load testing, deployment, and cost controls are not part of this local prototype.

## Reproduce the evaluations

```powershell
# Ensure indexes were built first
python validate_kubernetes_eval.py
python 6-compare-retrievers-weighted.py

# This step generates fresh answers and may be slow on local hardware
python 7-evaluate-generation-local.py

# Uses saved generation_eval_results.json (does not regenerate answers)
python 8-evaluate-answer-quality-local.py --judge-model llama3.2:3b

# If the audit CSV is present in step10_audit/
python 9-audit-evaluation.py report
```

Note: Generated evaluation outputs and large index files are excluded by `.gitignore` by default. The summary results above are retained here so the repository remains informative without distributing large artifacts. Do not call the AI-assisted 12-answer assessment independent human annotation.

## Next steps

- Test stronger generation and embedding models while retaining the same benchmark questions and properly rebuilding vectors when embeddings change.
- Review and expand relevance labels, create a held-out evaluation set, and run claim-level citation checks.
- Align benchmark and application retrieval implementations.
- Reduce inference latency and add reliable service diagnostics, testing, and deployment practices.
