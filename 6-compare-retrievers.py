
import json
import math
import pickle
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import CrossEncoder

K = 5
CANDIDATES = 30

# Small initial test set.
# Expand this with independently verified relevance labels.
EVAL_SET = [
    {
        "question": "What is the purpose of kube-proxy?",
        "relevant_pages": ["docs-concepts-architecture"]
    },
    {
        "question": "How does Horizontal Pod Autoscaling work?",
        "relevant_pages": [
            "docs-concepts-workloads-autoscaling-horizontal-pod-autoscale"
        ]
    },
    {
        "question": "How do Kubernetes Services expose Pods?",
        "relevant_pages": [
            "docs-concepts-services-networking-service"
        ]
    },
    {
        "question": "What is the difference between Deployment and StatefulSet?",
        "relevant_pages": [
            "docs-concepts-workloads-controllers-deployment",
            "docs-concepts-workloads-controllers-statefulset"
        ]
    }
]


def load_models():
    print("Loading indexes and reranker...")

    with open("indexes/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    faiss = FAISS.load_local(
        "indexes/faiss",
        embeddings,
        allow_dangerous_deserialization=True
    )

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return bm25, faiss, reranker


def doc_id(doc):
    return doc.metadata["chunk_id"]


def faiss_search(faiss, query):
    return faiss.similarity_search(query, k=CANDIDATES)


def hybrid_search(bm25, faiss_docs, query):
    bm25_docs = bm25.invoke(query)[:CANDIDATES]

    # Reciprocal Rank Fusion
    scores = {}
    documents = {}

    for docs in [bm25_docs, faiss_docs]:
        for rank, doc in enumerate(docs, 1):
            cid = doc_id(doc)

            scores[cid] = (
                scores.get(cid, 0) + 1 / (60 + rank)
            )
            documents[cid] = doc

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return [documents[cid] for cid in ranked_ids]


def rerank(reranker, query, docs):
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda item: float(item[1]),
        reverse=True
    )

    return [doc for doc, score in ranked]


def unique_pages(docs, k=K):
    pages = []
    seen = set()

    for doc in docs:
        page_id = doc.metadata["page_id"]

        if page_id not in seen:
            pages.append(page_id)
            seen.add(page_id)

        if len(pages) >= k:
            break

    return pages


def metrics(pages, relevant_pages):
    relevant = set(relevant_pages)

    hits = [
        1 if p in relevant else 0
        for p in pages[:K]
    ]

    precision = sum(hits) / K
    recall = sum(hits) / len(relevant)

    rr = 0.0
    for rank, hit in enumerate(hits, 1):
        if hit:
            rr = 1 / rank
            break

    dcg = sum(
        hit / math.log2(i + 2)
        for i, hit in enumerate(hits)
    )

    ideal_count = min(K, len(relevant))
    idcg = sum(
        1 / math.log2(i + 2)
        for i in range(ideal_count)
    )

    return {
        "Precision@5": precision,
        "Recall@5": recall,
        "MRR": rr,
        "NDCG@5": dcg / idcg if idcg else 0.0
    }


def main():
    bm25, faiss, reranker = load_models()

    records = []

    for item in EVAL_SET:
        question = item["question"]
        print("\nQuestion:", question)

        # A: FAISS
        start = time.perf_counter()
        dense_docs = faiss_search(faiss, question)
        faiss_time = time.perf_counter() - start

        # B: Hybrid RRF
        start = time.perf_counter()
        hybrid_docs = hybrid_search(
            bm25, dense_docs, question
        )
        hybrid_time = faiss_time + time.perf_counter() - start

        # C: Hybrid + CrossEncoder
        start = time.perf_counter()
        reranked_docs = rerank(
            reranker, question, hybrid_docs
        )
        rerank_time = hybrid_time + time.perf_counter() - start

        experiments = [
            ("FAISS", dense_docs, faiss_time),
            ("Hybrid", hybrid_docs, hybrid_time),
            ("Hybrid+Reranker", reranked_docs, rerank_time)
        ]

        for name, docs, elapsed in experiments:
            pages = unique_pages(docs)
            result = metrics(pages, item["relevant_pages"])

            record = {
                "method": name,
                "question": question,
                **result,
                "Latency(s)": elapsed,
                "retrieved_pages": pages
            }

            records.append(record)

            print(
                f"  {name:16} "
                f"MRR={result['MRR']:.3f} "
                f"NDCG={result['NDCG@5']:.3f} "
                f"Time={elapsed:.2f}s"
            )

    print("\n========== AVERAGE RESULTS ==========")

    metric_names = [
        "Precision@5", "Recall@5",
        "MRR", "NDCG@5", "Latency(s)"
    ]

    for method in ["FAISS", "Hybrid", "Hybrid+Reranker"]:
        subset = [r for r in records if r["method"] == method]

        print(f"\n{method}")

        for metric in metric_names:
            avg = sum(r[metric] for r in subset) / len(subset)
            print(f"  {metric}: {avg:.4f}")

    Path("evaluation_results.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8"
    )

    print("\nSaved evaluation_results.json")


if __name__ == "__main__":
    main()
