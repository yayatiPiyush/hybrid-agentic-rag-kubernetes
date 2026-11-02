"""
Step 3: Interactive hybrid search demo (LangChain retriever).

Shows the four-stage LangChain pipeline output so you can validate retrieval
quality before wiring it into the LangGraph agent.

Good test queries that stress different stages:
  "deployment process"          — likely in multiple spaces, tests BM25+dense
  "on-call rotation"            — paraphrase of "incident response schedule"
  "how to request VPN access"   — multi-word paraphrase, tests dense
  "JIRA-1234" or "prod-db-01"   — exact identifier, tests BM25 precision

LangSmith: if LANGCHAIN_TRACING_V2=true, each invoke() call is traced
automatically — open smith.langchain.com to inspect retrieval stages.

Usage:
  uv run 3-hybrid-search.py
"""

import time

from dotenv import load_dotenv

from utils.retrieval import HybridRetriever

load_dotenv()

SAMPLE_QUERIES = [
    "how to deploy to production",
    "security policy for access requests",
    "incident escalation process",
]


def show_results(query: str, retriever: HybridRetriever, top_k: int = 5) -> None:
    width = 64
    print(f"\n{'─' * width}")
    print(f"  Query: {query}")
    print(f"{'─' * width}")

    start = time.perf_counter()
    results = retriever.search(query, top_k=top_k)
    elapsed = time.perf_counter() - start

    if not results:
        print("  No results found.")
        return

    for i, r in enumerate(results, 1):
        print(
            f"\n  #{i}  {r['title']}\n"
            f"       {r['url']}\n"
            f"       Space: {r['space_key']}  |  page_id: {r['page_id']}  |  score: {r['score']:.4f}\n"
            f"       {r['text'][:280].replace(chr(10), ' ')}..."
        )

    print(f"\n  {len(results)} results in {elapsed:.2f}s")


def main() -> None:
    print("Loading LangChain indexes (BM25 + FAISS)...")
    retriever = HybridRetriever()
    print("Ready.\n")

    for q in SAMPLE_QUERIES:
        show_results(q, retriever)

    print("\n\nInteractive mode — Ctrl+C to quit")
    while True:
        try:
            q = input("\nSearch: ").strip()
            if q:
                show_results(q, retriever)
        except KeyboardInterrupt:
            print()
            break


if __name__ == "__main__":
    main()
