"""Web search and URL fetching tool implementations."""

import json
import logging
import os
import re
import urllib.parse
from html.parser import HTMLParser
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from wintermute.core.tool_deps import ToolDeps

logger = logging.getLogger(__name__)

SEARXNG_URL = os.environ.get("WINTERMUTE_SEARXNG_URL", "http://127.0.0.1:8888")


# ---------------------------------------------------------------------------
# HTML-to-text helper (stdlib only, no extra dependencies)
# ---------------------------------------------------------------------------

class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML-to-text converter that strips tags and scripts."""

    _IGNORE_TAGS = frozenset({"script", "style", "noscript", "svg", "head"})

    def __init__(self) -> None:
        super().__init__()
        self._pieces: list[str] = []
        self._ignore_depth = 0

    def handle_starttag(self, tag: str, _attrs: list) -> None:
        if tag in self._IGNORE_TAGS:
            self._ignore_depth += 1
        elif tag in ("br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
            self._pieces.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._IGNORE_TAGS:
            self._ignore_depth = max(0, self._ignore_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._ignore_depth == 0:
            self._pieces.append(data)

    def get_text(self) -> str:
        text = "".join(self._pieces)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _html_to_text(html: str) -> str:
    """Convert HTML to readable plain text."""
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def tool_search_web(inputs: dict, tool_deps: Optional[ToolDeps] = None, **_kw) -> str:
    query = inputs["query"]
    max_results = int(inputs.get("max_results", 5))
    logger.info("search_web: %s", query)
    searxng_url = (tool_deps.searxng_url if tool_deps else "") or SEARXNG_URL

    # --- Try SearXNG first ---
    try:
        params = urllib.parse.urlencode({"q": query, "format": "json", "categories": "general"})
        req = Request(f"{searxng_url}/search?{params}", headers={"User-Agent": "wintermute/0.1"})
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = [
            {"title": item.get("title", ""), "url": item.get("url", ""), "snippet": item.get("content", "")}
            for item in data.get("results", [])[:max_results]
        ]
        return json.dumps({"query": query, "source": "searxng", "results": results, "count": len(results)})
    except URLError as exc:
        reason = str(exc.reason) if hasattr(exc, "reason") else str(exc)
        if not any(s in reason for s in ("Connection refused", "No route to host", "timed out")):
            return json.dumps({"error": f"SearXNG request failed: {reason}"})
        logger.warning("SearXNG unreachable (%s), falling back to curl", reason)
    except Exception as exc:  # noqa: BLE001
        logger.warning("SearXNG error (%s), falling back to curl", exc)

    # --- Fallback: DuckDuckGo Instant Answer API via direct HTTP (no auth required) ---
    try:
        params = urllib.parse.urlencode({
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        })
        req = Request(
            f"https://api.duckduckgo.com/?{params}",
            headers={"User-Agent": "wintermute/0.1"},
        )
        with urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8")
        if not body.strip():
            raise RuntimeError("DuckDuckGo response was empty")
        data = json.loads(body)
        results = []
        if data.get("AbstractText") and data.get("AbstractURL"):
            results.append({"title": data.get("Heading", query), "url": data["AbstractURL"], "snippet": data["AbstractText"]})
        for topic in data.get("RelatedTopics", []):
            if len(results) >= max_results:
                break
            if "FirstURL" in topic:
                results.append({"title": topic.get("Text", "")[:80], "url": topic["FirstURL"], "snippet": topic.get("Text", "")})
            elif "Topics" in topic:
                for sub in topic["Topics"]:
                    if len(results) >= max_results:
                        break
                    if "FirstURL" in sub:
                        results.append({"title": sub.get("Text", "")[:80], "url": sub["FirstURL"], "snippet": sub.get("Text", "")})
        return json.dumps({
            "query": query, "source": "duckduckgo_fallback",
            "warning": "SearXNG unavailable. Start it with: cd ~/searxng-test && ./start-searxng.sh",
            "results": results[:max_results], "count": len(results[:max_results]),
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("search_web fallback failed")
        return json.dumps({"error": f"Both SearXNG and curl fallback failed: {exc}"})


def tool_fetch_url(inputs: dict, **_kw) -> str:
    url = inputs["url"]
    max_chars = int(inputs.get("max_chars", 20000))
    logger.info("fetch_url: %s", url)

    try:
        req = Request(url, headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; wintermute/0.1; "
                "+https://github.com/wintermute)"
            ),
            "Accept": "text/html,application/xhtml+xml,text/plain,*/*",
        })
        with urlopen(req, timeout=20) as resp:
            content_type = resp.headers.get("Content-Type", "")

            # Limit bytes read to avoid unbounded memory usage.
            # Use a generous upper bound to account for multibyte encodings.
            max_bytes = max_chars * 8 + 1024

            content_length = resp.headers.get("Content-Length")
            if content_length is not None:
                try:
                    if int(content_length) > max_bytes:
                        return json.dumps({
                            "error": (
                                f"Content too large to fetch safely "
                                f"({content_length} bytes reported)"
                            )
                        })
                except ValueError:
                    pass  # Invalid Content-Length, fall back to capped read.

            raw_chunks: list[bytes] = []
            bytes_read = 0
            chunk_size = 8192
            while bytes_read < max_bytes:
                chunk = resp.read(min(chunk_size, max_bytes - bytes_read))
                if not chunk:
                    break
                raw_chunks.append(chunk)
                bytes_read += len(chunk)

            raw = b"".join(raw_chunks)

            charset = "utf-8"
            if "charset=" in content_type:
                charset = content_type.split("charset=")[-1].split(";")[0].strip()
            body = raw.decode(charset, errors="replace")

    except URLError as exc:
        reason = str(exc.reason) if hasattr(exc, "reason") else str(exc)
        return json.dumps({"error": f"Failed to fetch URL: {reason}"})
    except Exception as exc:  # noqa: BLE001
        logger.exception("fetch_url failed")
        return json.dumps({"error": str(exc)})

    if "html" in content_type or body.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        body = _html_to_text(body)

    if len(body) > max_chars:
        body = body[:max_chars] + f"\n\n[... truncated at {max_chars} chars]"

    return json.dumps({
        "url": url,
        "content_type": content_type,
        "length": len(body),
        "content": body,
    })
