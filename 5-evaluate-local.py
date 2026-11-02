
import math
from utils.retrieval import HybridRetriever

# Small manually labeled evaluation dataset
# These are initial known-relevant pages, not exhaustive labels.
EVAL_SET = [
    {
        "question": "What is the purpose of kube-proxy?",
        "relevant_pages": [
            "docs-concepts-architecture"
        ]
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

K = 5


def calculate_metrics(retrieved_ids, relevant_ids):
    # Binary relevance for each retrieved page
    relevance = [
        1 if page_id in relevant_ids else 0
        for page_id in retrieved_ids[:K]
    ]

    # Precision@K
    precision = sum(relevance) / K

    # Recall@K
    recall = sum(relevance) / len(relevant_ids)

    # Reciprocal Rank
    rr = 0.0
    for rank, value in enumerate(relevance, start=1):
        if value == 1:
            rr = 1 / rank
            break

    # Discounted cumulative gain
    dcg = sum(
        value / math.log2(rank + 1)
        for rank, value in enumerate(relevance, start=1)
    )

    # Ideal ranking based on known relevant pages
    ideal_count = min(K, len(relevant_ids))
    idcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    ndcg = dcg / idcg if idcg else 0.0

    return {
        "Precision@5": precision,
        "Recall@5": recall,
        "RR": rr,
        "NDCG@5": ndcg
    }


def main():
    print("Loading Hybrid Retriever...")
    retriever = HybridRetriever()

    all_metrics = []

    for item in EVAL_SET:
        question = item["question"]
        relevant_ids = set(item["relevant_pages"])

        # Retrieve extra chunks because a page can have many chunks
        results = retriever.search(question, top_k=20)

        # Keep the first occurrence of each page
        retrieved_ids = []
        for result in results:
            page_id = result["page_id"]
            if page_id not in retrieved_ids:
                retrieved_ids.append(page_id)

            if len(retrieved_ids) == K:
                break

        metrics = calculate_metrics(
            retrieved_ids,
            relevant_ids
        )

        all_metrics.append(metrics)

        print(f"\nQuestion: {question}")

        for rank, page_id in enumerate(retrieved_ids, 1):
            print(f"  {rank}. {page_id}")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

    print("\n========== AVERAGE RESULTS ==========")

    for name in all_metrics[0]:
        average = sum(
            item[name] for item in all_metrics
        ) / len(all_metrics)

        label = "MRR" if name == "RR" else name
        print(f"{label}: {average:.4f}")


if __name__ == "__main__":
    main()
