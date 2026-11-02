"""Streamlit UI for the existing local Hybrid Agentic RAG backend.

Run in the project root: streamlit run streamlit_app.py
Requires the existing utils.agent_tools and previously built indexes.
"""
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama

from utils.agent_tools import hybrid_search, list_spaces

load_dotenv()
ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT = """You are a careful Kubernetes documentation assistant.
Use the hybrid_search tool for each user question. Use space_key='K8S'.
Use list_spaces only if you need to discover a space.
Ground factual claims in the retrieved passages; provide relevant source links from those passages.
Do not invent documentation links. If the retrieved excerpts do not contain enough
information, state what is missing instead of guessing. Be precise and concise.
"""


def flatten(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content or "")


def is_documentation_url(url):
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and parsed.netloc in (
        "kubernetes.io", "www.kubernetes.io"
    )


def extract_source_links(text):
    """Return Kubernetes docs markdown links, not code examples/localhost URLs."""
    pairs = re.findall(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", text)
    links = []
    seen = set()
    for label, url in pairs:
        clean = url.rstrip(".,;:")
        if is_documentation_url(clean) and clean not in seen:
            links.append({"title": label, "url": clean})
            seen.add(clean)
    return links


@st.cache_resource(show_spinner="Initializing the local Kubernetes agent...")
def load_agent(model_name, ctx_size, max_tokens):
    llm = ChatOllama(
        model=model_name,
        temperature=0,
        num_ctx=ctx_size,
        num_predict=max_tokens,
        # Qwen3 supports think=False on recent Ollama/langchain-ollama.
        # If your installed version rejects this, update langchain-ollama.
        think=False,
    )
    return create_agent(
        model=llm,
        tools=[list_spaces, hybrid_search],
        system_prompt=SYSTEM_PROMPT,
    )


def run_question(agent, question, model_name, ctx_size, max_tokens, agentic=True):
    """Run agent, then recover from no-tool/empty-answer with explicit retrieval.

    The fallback is deliberately labeled, not presented as an agent tool call.
    """
    start = time.perf_counter()
    retrieved = []
    calls = []
    answer = ""
    warning = None
    mode = "Agentic" if agentic else "Retrieval-first"

    if agentic:
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=question)]},
                config={"recursion_limit": 12},
            )
            for msg in result.get("messages", []):
                for call in (getattr(msg, "tool_calls", None) or []):
                    calls.append({"name": call.get("name"), "args": call.get("args", {})})
                if isinstance(msg, ToolMessage) and msg.name == "hybrid_search":
                    retrieved.append(flatten(msg.content))
            messages = result.get("messages", [])
            if messages:
                answer = flatten(messages[-1].content).strip()
        except Exception as exc:
            warning = f"Agent pass failed: {type(exc).__name__}: {exc}"

    # Local Qwen can consume output tokens without emitting a final message or tool call.
    # Never show its empty result as a successful RAG response.
    if not answer or not retrieved:
        mode = "Retrieval-first fallback" if agentic else "Retrieval-first"
        if not retrieved:
            try:
                tool_result = hybrid_search.invoke({"query": question, "space_key": "K8S", "top_k": 5})
                retrieved.append(flatten(tool_result))
                calls.append({"name": "hybrid_search", "args": {"query": question, "space_key": "K8S", "top_k": 5}, "source": "explicit retrieval"})
            except Exception as exc:
                return {"answer": "", "sources": [], "cited_sources": [], "citation_warnings": [],
                        "retrieved_context": "", "tool_calls": calls,
                        "elapsed_seconds": round(time.perf_counter()-start, 2),
                        "mode": mode, "warning": f"Retrieval failed: {type(exc).__name__}: {exc}"}
        context = "\n\n".join(retrieved)
        if context.startswith("No results found") and "#1 [" not in context:
            warning = "No matching documents found in the index."
        else:
            try:
                llm = ChatOllama(model=model_name, temperature=0, num_ctx=ctx_size,
                                 num_predict=max(1024, max_tokens), think=False)
                response = llm.invoke([
                    SystemMessage(content=(
                        "Answer the user's Kubernetes question from the retrieved passages only. "
                        "Be concise and clear. Cite one or more relevant URLs from the passages. "
                        "If evidence is insufficient, say so. Do not invent links."
                    )),
                    HumanMessage(content=f"Question: {question}\n\nRetrieved passages:\n{context[:13000]}\n\nAnswer:"),
                ])
                answer = flatten(response.content).strip()
                if not answer:
                    warning = ("Ollama returned an empty final answer even after direct retrieval. "
                               "Try Llama 3.2:3B in the model selector, increase max output tokens, "
                               "or update Ollama/langchain-ollama. "
                               f"Metadata: {getattr(response, 'response_metadata', {})}")
            except Exception as exc:
                warning = f"Direct generation failed: {type(exc).__name__}: {exc}"

    context = "\n\n".join(retrieved)
    sources = extract_source_links(context)
    citations = extract_source_links(answer)
    valid_urls = {x["url"] for x in sources}
    unrecognized = [x for x in citations if x["url"] not in valid_urls]
    return {"answer": answer, "sources": sources, "cited_sources": citations,
            "citation_warnings": unrecognized, "retrieved_context": context,
            "tool_calls": calls, "elapsed_seconds": round(time.perf_counter() - start, 2),
            "mode": mode, "warning": warning}


st.set_page_config(page_title="Hybrid Agentic RAG | Kubernetes", page_icon="🧠", layout="wide")
st.title("🧠 Hybrid Agentic RAG")
st.caption("Kubernetes documentation assistant · BM25 + FAISS + weighted fusion + CrossEncoder + LangGraph + Ollama")

with st.sidebar:
    st.header("Configuration")
    model_name = st.selectbox("Ollama model", ["llama3.2:3b", "qwen3:4b", "qwen3:8b"], index=0)
    agentic = st.toggle("Agentic tool selection (experimental with Qwen3)", value=False, help="Off: always retrieve first and generate a cited answer. On: let the LangGraph agent select tools, falling back if its output is empty.")
    context_size = st.select_slider("Context window", [2048, 4096, 8192], value=4096)
    max_tokens = st.slider("Max answer tokens", 256, 1536, 768, step=128)
    show_context = st.toggle("Show retrieved excerpts", value=False)
    st.divider()
    faiss_ok = (ROOT / "indexes" / "faiss").exists()
    bm25_ok = (ROOT / "indexes" / "bm25.pkl").exists()
    st.caption(f"FAISS index: {'Found' if faiss_ok else 'Missing'}")
    st.caption(f"BM25 index: {'Found' if bm25_ok else 'Missing'}")
    st.caption("Models and embeddings are served by local Ollama.")
    if st.button("New conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

chat_tab, eval_tab = st.tabs(["💬 Chat", "📊 Baseline evaluation"])

with chat_tab:
    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(item["question"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            if item.get("sources"):
                with st.expander("Retrieved sources", expanded=False):
                    for s in item["sources"]:
                        st.markdown(f"- [{s['title']}]({s['url']})")
            st.caption(f"{item['elapsed_seconds']:.1f}s · {sum(c['name'] == 'hybrid_search' for c in item['tool_calls'])} search calls")
            if show_context and item.get("retrieved_context"):
                with st.expander("Retrieval details"):
                    st.text(item["retrieved_context"])
                    st.json(item["tool_calls"])

    question = st.chat_input("Ask about Kubernetes deployments, services, autoscaling...")
    if question:
        if not faiss_ok or not bm25_ok:
            st.error("Indexes are missing. Run your existing 1-fetch-k8s.py and 2-build-index.py first.")
            st.stop()
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching Kubernetes documentation and generating an answer..."):
                try:
                    agent = load_agent(model_name, context_size, max_tokens)
                    record = run_question(agent, question, model_name, context_size, max_tokens, agentic=agentic)
                except Exception as exc:
                    st.error(f"Agent request failed: {type(exc).__name__}: {exc}")
                    st.info("Check that Ollama is running, the model is pulled, and indexes are available.")
                    st.stop()
            st.markdown(record["answer"] or "No usable final answer was returned. See diagnostic below.")
            if record.get("warning"):
                st.warning(record["warning"])
            st.caption(f"Execution mode: {record.get('mode', 'unknown')}")
            if record["sources"]:
                with st.expander("Retrieved sources"):
                    for s in record["sources"]:
                        st.markdown(f"- [{s['title']}]({s['url']})")
            st.caption(f"{record['elapsed_seconds']:.1f}s · {sum(c['name'] == 'hybrid_search' for c in record['tool_calls'])} search calls")
            if record["citation_warnings"]:
                st.warning("Some cited documentation URLs were not found in the tool output; verify citations manually.")
            if show_context:
                with st.expander("Retrieval details"):
                    st.text(record["retrieved_context"])
                    st.json(record["tool_calls"])
            st.session_state.chat_history.append({"question": question, **record})

with eval_tab:
    st.subheader("Reproducible baseline (40 provisional questions)")
    st.info("These are recorded baseline results, not scores recalculated by the Streamlit app. Retrieval labels are provisional; LLM-judge scores are exploratory.")
    st.dataframe([
        {"Pipeline": "FAISS", "Precision@5": 0.1950, "Recall@5": 0.8750, "MRR": 0.7112, "NDCG@5": 0.7341, "Latency (s)": 0.1209},
        {"Pipeline": "Hybrid RRF 30:70", "Precision@5": 0.1950, "Recall@5": 0.8750, "MRR": 0.7162, "NDCG@5": 0.7434, "Latency (s)": 0.1287},
        {"Pipeline": "Hybrid + reranker", "Precision@5": 0.1950, "Recall@5": 0.8625, "MRR": 0.7550, "NDCG@5": 0.7772, "Latency (s)": 1.7917},
    ], hide_index=True, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Generation latency", "42.68 s/question")
    c2.metric("Experimental faithfulness", "0.7400")
    c3.metric("Experimental answer relevance", "0.7625")
    st.caption("Local baseline: Qwen3:4B generator, Llama 3.2:3B judge. No independent accuracy validation yet.")
    if (ROOT / "generation_eval_results.json").exists():
        with st.expander("Saved generation evaluation sample"):
            try:
                records = json.loads((ROOT / "generation_eval_results.json").read_text(encoding="utf-8"))
                st.write(f"{len(records)} saved answer records")
                if records:
                    options = [f"{r.get('id', '?')} — {r.get('question', '')}" for r in records]
                    idx = st.selectbox("Question", range(len(records)), format_func=lambda i: options[i])
                    st.markdown(records[idx].get("answer", "No answer saved"))
            except (OSError, ValueError) as exc:
                st.warning(f"Could not read saved evaluations: {exc}")
