#!/usr/bin/env python3
"""
SecondSelf — Phase 1: The Archivist

Capture any note, link, or file into ``raw/`` with a timestamp and unique ID.

Usage
-----
    python capture.py note "Remember to review embeddings paper"
    python capture.py link "https://arxiv.org/abs/..."
    python capture.py link "https://arxiv.org/abs/..." --notes "Important paper"
    python capture.py file ./documents/resume.pdf
    python capture.py                          # interactive stdin mode
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from lib import models, storage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Current UTC timestamp in ISO-8601 format with ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _check_duplicate(content_hash: str) -> Optional[str]:
    """
    Return the ID of an existing raw capture with the same content hash.

    Returns ``None`` when no duplicate is found.
    """
    for capture in storage.read_raw_captures():
        if capture["meta"].get("content_hash") == content_hash:
            return capture["id"]
    return None


def _warn_duplicate(chash: str) -> None:
    """Print a duplicate-content warning to stderr (does not block capture)."""
    dup_id = _check_duplicate(chash)
    if dup_id:
        print(
            f"Warning: duplicate content already captured as raw/{dup_id}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Core capture functions
# ---------------------------------------------------------------------------

def capture_note(text: str, source: str = "cli") -> models.CaptureResult:
    """
    Capture a plain-text / markdown note.

    Parameters
    ----------
    text : str
        The note content.
    source : str
        Origin of the content — ``"cli"`` or ``"stdin"``.

    Returns
    -------
    CaptureResult
    """
    if not text or not text.strip():
        raise ValueError("note text is empty — nothing to capture")

    content = text.encode("utf-8")
    cid = storage.generate_capture_id()
    chash = storage.content_hash(content)

    _warn_duplicate(chash)

    meta = models.CaptureMeta(
        id=cid,
        timestamp=_now_iso(),
        type="note",
        source=source,
        original_filename="",
        content_hash=chash,
        content_filename="content.md",
    )
    result = storage.write_raw_capture(meta, content)
    print(f"Captured → raw/{cid}")
    return result


def capture_link(url: str, notes: str = "", source: str = "cli") -> models.CaptureResult:
    """
    Capture a URL with optional notes.

    The content file (``content.txt``) stores the URL on line 1 followed by
    any notes on subsequent lines.

    Parameters
    ----------
    url : str
        The URL to capture.
    notes : str
        Optional free-text notes about the link.
    source : str
        Origin of the content — ``"cli"`` or ``"stdin"``.

    Returns
    -------
    CaptureResult
    """
    if not url or not url.strip():
        raise ValueError("URL is empty — nothing to capture")

    # Build content: URL on line 1, optional notes below
    if notes:
        content_text = f"{url.strip()}\n\n{notes}"
    else:
        content_text = url.strip()
    content = content_text.encode("utf-8")

    cid = storage.generate_capture_id()
    chash = storage.content_hash(content)

    _warn_duplicate(chash)

    meta = models.CaptureMeta(
        id=cid,
        timestamp=_now_iso(),
        type="link",
        source=source,
        original_filename="",
        content_hash=chash,
        content_filename="content.txt",
    )
    result = storage.write_raw_capture(meta, content)
    print(f"Captured → raw/{cid}")
    return result


def capture_file(path: str, source: str = "path") -> models.CaptureResult:
    """
    Capture a local file (binary-safe copy).

    The original file is copied as-is into ``raw/{id}/content.{ext}`` and
    ``original_filename`` is recorded in the metadata.

    Parameters
    ----------
    path : str
        Path to the file to capture.
    source : str
        Origin of the content — ``"path"`` or ``"stdin"``.

    Returns
    -------
    CaptureResult

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If *path* is not a regular file.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    if not file_path.is_file():
        raise ValueError(f"not a file: {path}")

    content = file_path.read_bytes()
    cid = storage.generate_capture_id()
    chash = storage.content_hash(content)

    _warn_duplicate(chash)

    ext = file_path.suffix
    content_filename = f"content{ext}" if ext else "content.bin"

    meta = models.CaptureMeta(
        id=cid,
        timestamp=_now_iso(),
        type="file",
        source=source,
        original_filename=file_path.name,
        content_hash=chash,
        content_filename=content_filename,
    )
    result = storage.write_raw_capture(meta, content)
    print(f"Captured → raw/{cid}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SecondSelf — Capture anything into raw/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python capture.py note "Remember to review embeddings paper"
  python capture.py link "https://arxiv.org/abs/..."
  python capture.py link "https://arxiv.org/abs/..." --notes "Important paper"
  python capture.py file ./documents/resume.pdf
  python capture.py                          # interactive stdin mode
""",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- note ---
    note_parser = subparsers.add_parser("note", help="Capture a text note")
    note_parser.add_argument(
        "text", nargs="?", default=None,
        help="Note text (reads from stdin if omitted)",
    )

    # --- link ---
    link_parser = subparsers.add_parser("link", help="Capture a URL")
    link_parser.add_argument("url", help="URL to capture")
    link_parser.add_argument(
        "--notes", default="",
        help="Optional notes about the link",
    )

    # --- file ---
    file_parser = subparsers.add_parser("file", help="Capture a local file")
    file_parser.add_argument("path", help="Path to the file")

    args = parser.parse_args()

    try:
        if args.command == "note":
            text = args.text
            if text is None:
                # Read from stdin
                text = sys.stdin.read()
            capture_note(text, source="cli")

        elif args.command == "link":
            capture_link(args.url, notes=args.notes, source="cli")

        elif args.command == "file":
            capture_file(args.path, source="path")

        else:
            # No subcommand → interactive stdin mode
            text = sys.stdin.read()
            if not text.strip():
                parser.print_help()
                sys.exit(1)
            capture_note(text, source="stdin")

    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
