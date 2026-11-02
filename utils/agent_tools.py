"""
Tool functions shared between the LangGraph agent (4-agent.py), the evaluation
runner (5-evaluate.py), and the MCP server (6-mcp-server.py).

The three-tool interface mirrors the agentic-rag tutorial:
  list_spaces    →  discover what spaces/domains exist
  hybrid_search  →  BM25 + dense + RRF + Cohere rerank
  get_page_full  →  read a full Confluence page by ID

Each tool has two forms:
  _impl functions  — plain callables used directly by the MCP server
  @tool wrappers   — LangChain StructuredTool used by the LangGraph agent

The retriever and Confluence client are singletons loaded on first use.
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool

from utils.confluence import ConfluenceClient
from utils.retrieval import HybridRetriever

load_dotenv()

_retriever: HybridRetriever | None = None
_client: ConfluenceClient | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def _get_client() -> ConfluenceClient:
    global _client
    if _client is None:
        _client = ConfluenceClient()
    return _client


# ------------------------------------------------------------------
# Raw implementations (used by MCP server and internally)
# ------------------------------------------------------------------


def _list_spaces_impl() -> str:
    meta_path = Path(__file__).parents[1] / "indexes" / "meta.json"
    if not meta_path.exists():
        return "Index not built yet. Run 2-build-index.py first."

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    seen: dict[str, str] = {}
    for chunk in meta.values():
        key = chunk["space_key"]
        if key not in seen:
            seen[key] = chunk.get("space_name", key)

    spaces = [{"key": k, "name": v} for k, v in seen.items()]
    return json.dumps(spaces, indent=2)


def _hybrid_search_impl(query: str, space_key: str = "", top_k: int = 8) -> str:
    try:
        results = _get_retriever().search(
            query=query,
            space_key=space_key or None,
            top_k=top_k,
        )
    except FileNotFoundError as e:
        return f"Error: {e}"

    if not results:
        return f"No results found for: {query}"

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"#{i} [{r['title']}]({r['url']})\n"
            f"   page_id: {r['page_id']} | space: {r['space_key']} | score: {r['score']:.3f}\n"
            f"   {r['text'][:350]}..."
        )
    return "\n\n".join(lines)


def _get_page_full_impl(page_id: str) -> str:
    try:
        client = _get_client()
        raw = client.get_page(page_id)
        html = raw.get("body", {}).get("storage", {}).get("value", "")
        text = ConfluenceClient.html_to_text(html)
        title = raw.get("title", "")
        space_key = raw.get("space", {}).get("key", "")
        url = client.page_url(space_key, page_id)
        last_modified = raw.get("version", {}).get("when", "")
        header = f"# {title}\nURL: {url}\nLast modified: {last_modified}\n\n"
        return header + text
    except Exception as e:
        return f"Error fetching page {page_id}: {e}"


# ------------------------------------------------------------------
# LangChain @tool wrappers (used by LangGraph agent)
# ------------------------------------------------------------------


@tool
def list_spaces() -> str:
    """
    List the Confluence spaces available in the search index.

    Call this first to understand which knowledge areas are indexed before
    deciding which space_key to filter on in hybrid_search.
    Returns a JSON array of {key, name} objects.
    """
    return _list_spaces_impl()


@tool
def hybrid_search(query: str, space_key: str = "", top_k: int = 8) -> str:
    """
    Search Confluence using hybrid retrieval: BM25 + dense embeddings +
    Reciprocal Rank Fusion + cross-encoder rerank.

    Better than keyword search alone: catches "login configuration" when the
    document says "authentication setup", and still catches exact identifiers
    like ticket IDs or product names that semantic search dilutes.

    Args:
        query:     Natural language question or keyword phrase.
        space_key: Restrict results to one Confluence space (e.g. "ENG").
                   Leave empty to search across all indexed spaces.
        top_k:     Number of chunks to return (default 8).

    Returns a formatted list of results with title, URL, space, and a text
    snippet. Call get_page_full with the page_id for the complete content.
    """
    return _hybrid_search_impl(query=query, space_key=space_key, top_k=top_k)


@tool
def get_page_full(page_id: str) -> str:
    """
    Fetch the complete text of a Confluence page by its page_id.

    Use this after hybrid_search returns a promising result and you need the
    full content rather than just the matched snippet. The page_id is shown
    in every hybrid_search result.

    Args:
        page_id: Confluence numeric page ID (from hybrid_search results).

    Returns the full page text with title and URL header.
    """
    return _get_page_full_impl(page_id)
