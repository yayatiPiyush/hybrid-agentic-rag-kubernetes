"""
LangChain-based HybridRetriever: BM25 + FAISS + EnsembleRetriever (RRF) + CohereRerank.

Maps the four-stage pipeline from project 1 to LangChain components:
  Stage 1 — BM25Retriever          (langchain_community)
  Stage 2 — FAISS dense retriever  (langchain_community + langchain_openai)
  Stage 3 — EnsembleRetriever      (RRF with c=60, identical to project 1's _rrf())
  Stage 4 — CohereRerank           (langchain_cohere ContextualCompressionRetriever)

Indexes are loaded from disk (built by 2-build-index.py):
  indexes/bm25.pkl     — pickled BM25Retriever
  indexes/faiss/       — FAISS vector store (index.faiss + index.pkl)
  indexes/meta.json    — chunk_id → full chunk dict (title, url, space_key, text …)
"""

import json
import pickle
from pathlib import Path
from typing import Optional

from sentence_transformers import CrossEncoder
#from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_cohere import CohereRerank
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
#from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

INDEX_DIR = Path(__file__).parents[1] / "indexes"
#EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_MODEL = "nomic-embed-text"
RERANK_MODEL = "rerank-v4.0-fast"
CANDIDATE_K = 50   # candidates fed into each retriever before fusion
RERANK_TOP_N = 20  # Cohere returns this many; we slice to top_k after space filter


def load_meta() -> dict[str, dict]:
    path = INDEX_DIR / "meta.json"
    if not path.exists():
        raise FileNotFoundError(
            "indexes/meta.json not found. Run 2-build-index.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


class HybridRetriever:
    def __init__(self) -> None:
        self._meta = load_meta()

        # Stage 1: BM25 (pickled LangChain BM25Retriever)
        bm25_path = INDEX_DIR / "bm25.pkl"
        if not bm25_path.exists():
            raise FileNotFoundError(
                "indexes/bm25.pkl not found. Run 2-build-index.py first."
            )
        with open(bm25_path, "rb") as f:
            bm25: BM25Retriever = pickle.load(f)
        bm25.k = CANDIDATE_K

        # Stage 2: FAISS dense retriever
        faiss_path = INDEX_DIR / "faiss"
        if not faiss_path.exists():
            raise FileNotFoundError(
                "indexes/faiss/ not found. Run 2-build-index.py first."
            )
        #embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        faiss_store = FAISS.load_local(
            str(faiss_path), embeddings, allow_dangerous_deserialization=True
        )
        faiss_retriever = faiss_store.as_retriever(
            search_kwargs={"k": CANDIDATE_K}
        )

        # Stage 3: Reciprocal Rank Fusion via EnsembleRetriever
        # EnsembleRetriever uses the same RRF formula (c=60) as project 1's _rrf().
        # ensemble = EnsembleRetriever(
        #     retrievers=[bm25, faiss_retriever],
        #     weights=[0.5, 0.5],
        # )
        ensemble = EnsembleRetriever(
        retrievers=[bm25, faiss_retriever],
        weights=[0.3, 0.7],
        )

        # Stage 4: Cohere cross-encoder rerank
        #reranker = CohereRerank(model=RERANK_MODEL, top_n=RERANK_TOP_N)
        # self._chain = ContextualCompressionRetriever(
        #     base_compressor=reranker,
        #     base_retriever=ensemble,
        # )
        #self._chain = ensemble

        self._chain = ensemble

        self._reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        space_key: Optional[str] = None,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Run the full four-stage pipeline and return top_k result dicts.

        Each dict mirrors project 1: chunk_id, page_id, title, space_key,
        url, text, score (Cohere relevance_score).

        Space filtering is applied after reranking (Cohere sees all candidates
        first, then we drop non-matching spaces and truncate to top_k).
        """
        docs = self._chain.invoke(query)

        # Score each retrieved chunk against the query
        pairs = [(query, doc.page_content) for doc in docs]

        if not pairs:
            return []

        scores = self._reranker.predict(pairs)

        # Attach scores and sort from highest to lowest
        ranked_docs = sorted(
            zip(docs, scores),
            key=lambda item: float(item[1]),
            reverse=True
        )

        docs = []

        for doc, score in ranked_docs:
            doc.metadata["relevance_score"] = float(score)
            docs.append(doc)

        results: list[dict] = []
        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id or chunk_id not in self._meta:
                continue
            chunk = self._meta[chunk_id]
            if space_key and chunk.get("space_key") != space_key:
                continue
            results.append(
                {
                    **chunk,
                    "score": doc.metadata.get("relevance_score", 0.0),
                }
            )
            if len(results) >= top_k:
                break

        return results
