"""
Text extraction helpers for SecondSelf.

Given a raw capture (meta + content file), extract plain text suitable
for LLM classification.

| Source type | Extraction method                                     |
|------------|--------------------------------------------------------|
| note       | Read ``content.md`` directly                           |
| link       | ``requests`` + ``beautifulsoup4`` strip HTML; fallback to URL |
| file (PDF) | ``pypdf`` text extraction; fallback to filename        |
| file (other)| Read as text if possible; fallback to filename        |
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from .storage import RAW_DIR


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text(capture: Dict[str, Any]) -> str:
    """
    Extract plain text from a raw capture.

    Parameters
    ----------
    capture : dict
        A capture dict as returned by ``storage.read_raw_captures()`` —
        must contain ``id``, ``meta`` (dict), and ``dir`` (Path).

    Returns
    -------
    str
        Extracted text suitable for LLM classification.
    """
    meta = capture["meta"]
    capture_dir: Path = capture["dir"]
    ctype = meta.get("type", "note")
    content_filename = meta.get("content_filename", "")

    if not content_filename:
        # Fallback to type-based default
        if ctype == "note":
            content_filename = "content.md"
        elif ctype == "link":
            content_filename = "content.txt"
        else:
            content_filename = "content.bin"

    content_path = capture_dir / content_filename

    if ctype == "note":
        return _extract_note(content_path)
    elif ctype == "link":
        return _extract_link(content_path)
    elif ctype == "file":
        return _extract_file(content_path, meta)
    else:
        # Unknown type — try reading as text
        return _read_text_safe(content_path)


# ---------------------------------------------------------------------------
# Note extraction
# ---------------------------------------------------------------------------

def _extract_note(content_path: Path) -> str:
    """Read a note's ``content.md`` (or ``content.txt``) directly."""
    return _read_text_safe(content_path)


# ---------------------------------------------------------------------------
# Link extraction
# ---------------------------------------------------------------------------

def _extract_link(content_path: Path) -> str:
    """
    Extract text from a link capture.

    The content file stores the URL on line 1 followed by optional notes.
    We fetch the URL and strip HTML tags; if fetching fails, we fall back
    to the URL string + notes.
    """
    raw = _read_text_safe(content_path)
    lines = raw.strip().split("\n")

    # First non-empty line is the URL
    url = ""
    notes_start = 0
    for i, line in enumerate(lines):
        if line.strip():
            url = line.strip()
            notes_start = i + 1
            break

    # Remaining lines (after a blank line) are notes
    notes = "\n".join(lines[notes_start:]).strip()
    # Skip the blank separator line if present
    if notes.startswith("\n"):
        notes = notes.lstrip("\n")

    # Try to fetch the URL content
    url_text = _fetch_url_text(url)
    if url_text:
        parts = [f"URL: {url}"]
        if notes:
            parts.append(f"Notes: {notes}")
        parts.append(f"Content:\n{url_text}")
        return "\n\n".join(parts)
    else:
        # Fallback to URL + notes
        parts = [f"URL: {url}"]
        if notes:
            parts.append(f"Notes: {notes}")
        return "\n\n".join(parts)


def _fetch_url_text(url: str, timeout: int = 10) -> str:
    """
    Fetch *url* and return visible text (HTML tags stripped).

    Returns an empty string on any error.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (compatible; SecondSelf/1.0)"
        })
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Get text with some structure
        text = soup.get_text(separator="\n", strip=True)

        # Collapse excessive blank lines
        lines = [line for line in text.split("\n") if line.strip()]
        return "\n".join(lines[:200])  # limit to 200 lines

    except Exception:
        return ""


# ---------------------------------------------------------------------------
# File extraction
# ---------------------------------------------------------------------------

def _extract_file(content_path: Path, meta: Dict[str, Any]) -> str:
    """
    Extract text from a file capture.

    - PDF: use ``pypdf`` to extract text
    - Markdown/text: read directly
    - Other: fall back to the original filename
    """
    suffix = content_path.suffix.lower()
    original_filename = meta.get("original_filename", content_path.name)

    if suffix == ".pdf":
        text = _extract_pdf(content_path)
        if text:
            return f"File: {original_filename}\n\n{text}"
        # Fallback to filename
        return f"File: {original_filename}\n\n(PDF text extraction failed)"

    elif suffix in (".md", ".txt", ".markdown", ".rst"):
        text = _read_text_safe(content_path)
        return f"File: {original_filename}\n\n{text}"

    elif suffix in (".json", ".yaml", ".yml", ".csv", ".py", ".js", ".ts",
                    ".html", ".htm", ".xml", ".log", ".ini", ".cfg"):
        text = _read_text_safe(content_path)
        return f"File: {original_filename}\n\n{text}"

    else:
        # Binary or unknown — try text, fall back to filename
        text = _read_text_safe(content_path)
        if text and not text.startswith("[Binary"):
            return f"File: {original_filename}\n\n{text}"
        return f"File: {original_filename}\n\n(Binary file — no text extracted)"


def _extract_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF using ``pypdf``."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        texts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                texts.append(page_text)
        return "\n\n".join(texts)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _read_text_safe(path: Path, max_bytes: int = 100_000) -> str:
    """
    Read *path* as UTF-8 text, returning a placeholder for binary files.

    Reads at most *max_bytes* bytes to avoid loading huge files.
    """
    try:
        raw = path.read_bytes()[:max_bytes]
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return f"[Binary or unreadable file: {path.name}]"