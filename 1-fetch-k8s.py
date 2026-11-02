"""
Alternative to 1-fetch-confluence.py: scrape the Kubernetes public docs.

Reads the English sitemap from kubernetes.io, filters to the sections you
want (concepts, tasks, tutorials, setup by default), scrapes each page,
extracts the main content, chunks it, and saves to chunks/ in the exact
same JSON format that 2-build-index.py expects.

After this runs, the rest of the pipeline is identical to the Confluence flow:
  uv run 2-build-index.py
  uv run 3-hybrid-search.py
  uv run streamlit run 7-chatbot.py

Dependencies beyond the main project:
  pip install beautifulsoup4   (or: uv pip install beautifulsoup4)

Rate limiting: 0.5 s between requests by default — respectful for a public
docs site. Kubernetes.io robots.txt allows crawling documentation pages.
"""

import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# ── Config ───────────────────────────────────────────────────────────────────

CHUNKS_DIR = Path(__file__).parent / "chunks"
CHUNKS_DIR.mkdir(exist_ok=True)

BASE_URL = "https://kubernetes.io"
SITEMAP_URL = f"{BASE_URL}/en/sitemap.xml"

# Sections to crawl (comment out any you don't need).
# Skipping /docs/reference/kubernetes-api/ — it's huge and very low-signal
# for conversational queries. Add it back if you need API field lookups.
INCLUDE_SECTIONS = [
    "/docs/concepts/",
    "/docs/tasks/",
    "/docs/tutorials/",
    "/docs/setup/",
    "/docs/reference/glossary/",
    "/docs/reference/kubectl/",
]

SKIP_PATTERNS = [
    "/docs/reference/kubernetes-api/",  # ~600 pages, mostly spec tables
    "/docs/contribute/",                # meta / contributor guides
]

REQUEST_DELAY = 0.5   # seconds between HTTP requests
MAX_CHUNK_CHARS = 1500
OVERLAP_CHARS = 150

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; k8s-rag-indexer/1.0; "
        "educational project; github.com/your-org/ai-cookbook)"
    )
}

# ── Sitemap ───────────────────────────────────────────────────────────────────


def _sitemap_urls(url: str) -> list[str]:
    """Recursively resolve a sitemap or sitemap-index and return all <loc> URLs."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    # Sitemap index → recurse into children
    children = root.findall("sm:sitemap/sm:loc", ns)
    if children:
        urls: list[str] = []
        for child in children:
            urls.extend(_sitemap_urls(child.text.strip()))
        return urls

    # Regular sitemap → collect <loc> entries
    return [loc.text.strip() for loc in root.findall("sm:url/sm:loc", ns)]


def get_doc_urls() -> list[str]:
    """Return sorted, deduplicated doc URLs matching INCLUDE_SECTIONS."""
    print(f"Fetching sitemap: {SITEMAP_URL}")
    all_urls = _sitemap_urls(SITEMAP_URL)

    kept = []
    for url in all_urls:
        path = urlparse(url).path
        if not any(path.startswith(s) for s in INCLUDE_SECTIONS):
            continue
        if any(path.startswith(s) for s in SKIP_PATTERNS):
            continue
        kept.append(url)

    return sorted(set(kept))


# ── Content extraction ────────────────────────────────────────────────────────


def extract(html: str, url: str) -> tuple[str, str]:
    """
    Parse a Kubernetes docs HTML page and return (title, plain_text).

    kubernetes.io uses the Docsy Hugo theme. The main content is inside
    a <div class="td-content"> element. Navigation, sidebars, feedback
    widgets, and scripts are stripped before text extraction.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Title: prefer the <h1> on the page; fall back to <title> tag
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(" ", strip=True)
    elif soup.title:
        title = soup.title.get_text(" ", strip=True).replace(" | Kubernetes", "").strip()
    else:
        title = urlparse(url).path.rstrip("/").rsplit("/", 1)[-1].replace("-", " ").title()

    # Main content area (Docsy theme)
    content = (
        soup.find(class_="td-content")
        or soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find("article")
    )
    if not content:
        return title, ""

    # Strip noise before text extraction
    for tag in content.find_all(
        ["nav", "script", "style", "aside", "footer", "noscript"]
    ):
        tag.decompose()
    for tag in content.find_all(
        class_=["feedback--widget", "td-sidebar-toc", "td-page-meta"]
    ):
        tag.decompose()

    # Extract text — `\n` separator keeps paragraph/heading breaks
    text = content.get_text(separator="\n", strip=True)
    # Collapse runs of blank lines to at most one blank line
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return title, text


# ── Chunking ──────────────────────────────────────────────────────────────────


def chunk(text: str) -> list[str]:
    """
    Split text at line boundaries into overlapping chunks of ~MAX_CHUNK_CHARS.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    chunks: list[str] = []
    current = ""

    for line in lines:
        candidate = f"{current}\n{line}".strip() if current else line
        if len(candidate) <= MAX_CHUNK_CHARS:
            current = candidate
        else:
            if current:
                chunks.append(current)
                overlap = current[-OVERLAP_CHARS:] if OVERLAP_CHARS else ""
                current = f"{overlap}\n{line}".strip() if overlap else line
            else:
                # Single line longer than limit — hard split
                while len(line) > MAX_CHUNK_CHARS:
                    chunks.append(line[:MAX_CHUNK_CHARS])
                    line = line[MAX_CHUNK_CHARS - OVERLAP_CHARS:]
                current = line

    if current:
        chunks.append(current)

    return chunks or [text]


# ── Page ID ───────────────────────────────────────────────────────────────────


def url_to_id(url: str) -> str:
    """Derive a stable, filesystem-safe page_id from the URL path."""
    path = urlparse(url).path.strip("/")
    return re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    urls = get_doc_urls()
    print(f"Found {len(urls)} pages to index\n")

    total_chunks = ok = skipped = errors = 0

    for i, url in enumerate(urls, 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()

            title, text = extract(resp.text, url)

            if not text.strip():
                skipped += 1
                print(f"[{i:3}/{len(urls)}] SKIP (no content)  {url}")
                time.sleep(REQUEST_DELAY)
                continue

            page_id = url_to_id(url)
            text_chunks = chunk(text)

            for idx, chunk_text in enumerate(text_chunks):
                record = {
                    "chunk_id": f"{page_id}_c{idx}",
                    "page_id": page_id,
                    "title": title,
                    "space_key": "K8S",
                    "space_name": "Kubernetes Docs",
                    "url": url,
                    "text": chunk_text,
                    "chunk_idx": idx,
                    "last_modified": resp.headers.get("Last-Modified", ""),
                }
                out = CHUNKS_DIR / f"{record['chunk_id']}.json"
                out.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

            total_chunks += len(text_chunks)
            ok += 1
            print(
                f"[{i:3}/{len(urls)}] {title[:55]:<55}  {len(text_chunks)} chunk(s)"
            )

        except requests.HTTPError as e:
            errors += 1
            print(f"[{i:3}/{len(urls)}] HTTP {e.response.status_code}  {url}")

        except Exception as e:
            errors += 1
            print(f"[{i:3}/{len(urls)}] ERROR  {url}  —  {e}")

        time.sleep(REQUEST_DELAY)

    print(
        f"\nDone: {ok} pages → {total_chunks} chunks  "
        f"({skipped} empty, {errors} errors)"
    )
    print(f"Chunks saved to: {CHUNKS_DIR.resolve()}")
    print("\nNext step:")
    print("  uv run 2-build-index.py")


if __name__ == "__main__":
    main()
