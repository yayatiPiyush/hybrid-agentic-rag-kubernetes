"""
Step 2: Build LangChain-native indexes from the saved chunks.

Loads every JSON file from chunks/ then:
  1. Converts chunks to LangChain Documents (page_content + metadata).
  2. Builds a BM25Retriever and pickles it → indexes/bm25.pkl
  3. Embeds each chunk with text-embedding-3-small via OpenAIEmbeddings
     and stores the FAISS vectorstore → indexes/faiss/
  4. Saves metadata mapping → indexes/meta.json  (chunk_id → full chunk dict)

The FAISS and BM25 indexes are consumed by utils/retrieval.py at query time.
The meta.json is used to resolve Documents back to full chunk dicts (title,
url, space_key, text …) since FAISS only stores what you put in the metadata.

Re-run after a fresh 1-fetch-confluence.py to pick up new or changed pages.

Usage:
  uv run 2-build-index.py
"""

import json
import pickle
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
#from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm

load_dotenv()

CHUNKS_DIR = Path("chunks")
INDEX_DIR = Path("indexes")
INDEX_DIR.mkdir(exist_ok=True)

#EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_MODEL = "nomic-embed-text"
EMBED_BATCH = 100   # conservative; OpenAI allows up to 2 048 per call


def load_chunks() -> list[dict]:
    paths = sorted(CHUNKS_DIR.glob("*.json"))
    if not paths:
        return []
    chunks = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    pages = len(set(c["page_id"] for c in chunks))
    print(f"Loaded {len(chunks)} chunks from {pages} pages")
    return chunks


def chunks_to_documents(chunks: list[dict]) -> list[Document]:
    """Convert chunk dicts to LangChain Documents preserving all metadata."""
    return [
        Document(
            page_content=c["text"],
            metadata={
                "chunk_id": c["chunk_id"],
                "page_id": c["page_id"],
                "title": c["title"],
                "space_key": c["space_key"],
                "space_name": c.get("space_name", c["space_key"]),
                "url": c["url"],
            },
        )
        for c in chunks
    ]


def build_bm25(docs: list[Document]) -> None:
    print("\nBuilding BM25 index...")
    retriever = BM25Retriever.from_documents(docs, k=50)
    bm25_path = INDEX_DIR / "bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(retriever, f)
    print(f"  Saved indexes/bm25.pkl ({len(docs)} documents)")


def build_faiss(docs: list[Document]) -> None:
    print("\nBuilding FAISS dense index...")
    #embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Build in batches so we can show progress and stay within API rate limits
    faiss_store = None
    for i in tqdm(range(0, len(docs), EMBED_BATCH), desc="  Batches"):
        batch = docs[i : i + EMBED_BATCH]
        if faiss_store is None:
            faiss_store = FAISS.from_documents(batch, embeddings)
        else:
            faiss_store.add_documents(batch)

    faiss_path = str(INDEX_DIR / "faiss")
    faiss_store.save_local(faiss_path)
    print(f"  Saved indexes/faiss/  ({len(docs)} vectors)")


def build_meta(chunks: list[dict]) -> None:
    meta = {c["chunk_id"]: c for c in chunks}
    (INDEX_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved indexes/meta.json ({len(meta)} entries)")


def main() -> None:
    chunks = load_chunks()
    if not chunks:
        print("No chunks found in chunks/. Run 1-fetch-confluence.py first.")
        return

    docs = chunks_to_documents(chunks)
    build_bm25(docs)
    build_faiss(docs)
    build_meta(chunks)

    print("\nAll indexes built.")
    print("  Run 3-hybrid-search.py to test retrieval interactively.")
    print("  Run 4-agent.py 'your question' to try the full LangGraph agent.")
    print("  Run 5-evaluate.py to benchmark the pipeline.")
    print("  Run 6-mcp-server.py to start the MCP server.")


if __name__ == "__main__":
    main()
