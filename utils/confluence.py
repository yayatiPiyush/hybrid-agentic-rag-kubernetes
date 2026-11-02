"""
Confluence Cloud REST API client.

Uses the v1 REST API (wiki/rest/api) which is the most widely supported
path for Confluence Cloud. Authentication is email + API token via Basic Auth.

Generate an API token at: https://id.atlassian.com/manage-profile/security/api-tokens
"""

import os
import re
from dataclasses import dataclass
from html import unescape

import requests
from requests.auth import HTTPBasicAuth


@dataclass
class PageChunk:
    chunk_id: str        # "{page_id}_c{idx}" — stable reference
    page_id: str
    title: str
    space_key: str
    space_name: str
    url: str
    text: str            # plain text, HTML stripped
    chunk_idx: int
    last_modified: str   # ISO 8601 from version.when


class ConfluenceClient:
    def __init__(self) -> None:
        base = os.environ["CONFLUENCE_BASE_URL"].rstrip("/")
        self._api = f"{base}/wiki/rest/api"
        self._base = base
        self._auth = HTTPBasicAuth(
            os.environ["CONFLUENCE_EMAIL"],
            os.environ["CONFLUENCE_API_TOKEN"],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_spaces(self) -> list[dict]:
        """Return all global spaces the token can access."""
        results, start = [], 0
        while True:
            data = self._get("/space", params={"start": start, "limit": 50, "type": "global"})
            results.extend(data["results"])
            if data.get("size", 0) < 50:
                break
            start += 50
        return results

    def iter_pages(self, space_key: str):
        """Yield raw page dicts for every page in `space_key` with body.storage."""
        start = 0
        while True:
            data = self._get(
                "/content",
                params={
                    "spaceKey": space_key,
                    "type": "page",
                    "expand": "body.storage,version,space",
                    "start": start,
                    "limit": 50,
                },
            )
            yield from data["results"]
            if data.get("size", 0) < 50:
                break
            start += 50

    def get_page(self, page_id: str) -> dict:
        """Fetch a single page by ID with body.storage expanded."""
        return self._get(
            f"/content/{page_id}",
            params={"expand": "body.storage,version,space"},
        )

    # ------------------------------------------------------------------
    # Text processing
    # ------------------------------------------------------------------

    @staticmethod
    def html_to_text(html: str) -> str:
        """
        Convert Confluence storage-format HTML/XHTML to plain text.

        Block-level tags (p, h1-h6, li, div, br, tr, td) become newlines so
        that paragraph structure survives tag stripping. Inline tags are
        removed and HTML entities are decoded.
        """
        text = re.sub(
            r"<(p|h[1-6]|li|br|div|tr|td)(\s[^>]*)?>",
            "\n",
            html,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"<[^>]+>", "", text)
        text = unescape(text)
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
        return "\n".join(ln for ln in lines if ln)

    @staticmethod
    def chunk(
        text: str,
        max_chars: int = 1500,
        overlap_chars: int = 150,
    ) -> list[str]:
        """
        Split plain text into overlapping chunks at line boundaries.

        1500 chars ≈ 375 tokens, well under the 8 191-token limit for
        text-embedding-3-small.
        """
        lines = [ln for ln in text.splitlines() if ln.strip()]
        chunks: list[str] = []
        current = ""

        for line in lines:
            candidate = f"{current}\n{line}".strip() if current else line
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                    overlap = current[-overlap_chars:] if overlap_chars else ""
                    current = f"{overlap}\n{line}".strip() if overlap else line
                else:
                    while len(line) > max_chars:
                        chunks.append(line[:max_chars])
                        line = line[max_chars - overlap_chars:]
                    current = line

        if current:
            chunks.append(current)

        return chunks or [text]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict = None) -> dict:
        resp = requests.get(
            f"{self._api}{path}",
            auth=self._auth,
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def page_url(self, space_key: str, page_id: str) -> str:
        return f"{self._base}/wiki/spaces/{space_key}/pages/{page_id}"
