"""Step 8: Generate and audit answers from the local Hybrid Agentic RAG system.

Usage:
    python 7-evaluate-generation-local.py
    python 7-evaluate-generation-local.py kubernetes_eval_40_provisional.json

Dependencies: the same installed packages as working 4-agent-local.py.
No paid APIs are needed. Outputs are *not* automated faithfulness scores.
"""

import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage

from utils.agent_tools import list_spaces, hybrid_search

load_dotenv()

MODEL = "qwen3:4b"
INPUT_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("kubernetes_eval_40_provisional.json")
OUTPUT_PATH = Path("generation_eval_results.json")

SYSTEM_PROMPT = """You are a Kubernetes documentation assistant.
For every Kubernetes question, use hybrid_search to retrieve relevant documentation.
You may use list_spaces to discover available document sources.
Use only the retrieved material to answer. Cite the document URLs actually supplied
by the search tool. Do not invent sources or unsupported claims.
If the context is insufficient, say so explicitly. Keep answers concise.
"""


def as_text(content):
    """Flatten model/tool message content for inspection."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def parse_source_urls(tool_text):
    """Extract URLs supplied to the agent in search tool messages."""
    return sorted(set(re.findall(r"https?://[^\s\]\)>]+", tool_text)))


def evaluate_one(agent, question):
    start = time.perf_counter()
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": 15},
    )
    elapsed = time.perf_counter() - start

    final_answer = as_text(result["messages"][-1].content)
    retrieved_contexts = []
    tool_calls = []
    for message in result["messages"]:
        for call in (getattr(message, "tool_calls", None) or []):
            tool_calls.append({"name": call.get("name"), "args": call.get("args", {})})
        if isinstance(message, ToolMessage) and message.name == "hybrid_search":
            retrieved_contexts.append(as_text(message.content))

    context_text = "\n\n".join(retrieved_contexts)
    available_urls = parse_source_urls(context_text)
    cited_urls = parse_source_urls(final_answer)
    unexpected_citations = sorted(set(cited_urls) - set(available_urls))

    return {
        "question": question,
        "answer": final_answer,
        "retrieved_context": context_text,
        "tool_calls": tool_calls,
        "available_source_urls": available_urls,
        "cited_urls": cited_urls,
        "citation_url_check": {
            "has_citation": bool(cited_urls),
            "all_cited_urls_in_retrieval": not unexpected_citations and bool(cited_urls),
            "unexpected_urls": unexpected_citations,
        },
        "latency_seconds": round(elapsed, 3),
        "human_review": {
            "answer_correct": None,
            "all_claims_supported_by_context": None,
            "citations_support_claims": None,
            "notes": "",
        },
    }


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Evaluation data not found: {INPUT_PATH}")
    items = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError("Expected a nonempty JSON list with 'question' fields")

    llm = ChatOllama(model=MODEL, temperature=0, num_ctx=8192)
    agent = create_agent(
        model=llm,
        tools=[list_spaces, hybrid_search],
        system_prompt=SYSTEM_PROMPT,
    )

    records = []
    for index, item in enumerate(items, start=1):
        question = item["question"]
        print(f"[{index}/{len(items)}] {question}", flush=True)
        try:
            record = evaluate_one(agent, question)
            record["id"] = item.get("id", f"question_{index}")
            record["category"] = item.get("category", "")
            record["relevant_pages_provisional"] = item.get("relevant_pages", [])
            records.append(record)
            print(f"  {record['latency_seconds']:.2f}s; "
                  f"search calls={sum(c['name']=='hybrid_search' for c in record['tool_calls'])}; "
                  f"citations={len(record['cited_urls'])}")
        except Exception as exc:
            records.append({"id": item.get("id", f"question_{index}"),
                            "question": question, "error": str(exc)})
            print(f"  ERROR: {exc}")

        # Persist after each question so interrupted runs retain their outputs.
        OUTPUT_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    valid = [r for r in records if "error" not in r]
    print(f"\nSaved {len(records)} records to {OUTPUT_PATH}")
    if valid:
        mean_latency = sum(r["latency_seconds"] for r in valid) / len(valid)
        citations_present = sum(r["citation_url_check"]["has_citation"] for r in valid)
        supported_urls = sum(r["citation_url_check"]["all_cited_urls_in_retrieval"] for r in valid)
        print(f"Mean generation pipeline latency: {mean_latency:.2f}s")
        print(f"Answers containing URLs: {citations_present}/{len(valid)}")
        print(f"Answers citing only retrieved URLs: {supported_urls}/{len(valid)}")
        print("URL checks do not establish faithfulness or citation correctness; review claims manually.")


if __name__ == "__main__":
    main()
