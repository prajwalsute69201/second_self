#!/usr/bin/env python3
"""
SecondSelf — Phase 2.1: The Librarian (Auto-Classify)

Classify raw captures into PARA categories with tags and summaries,
writing structured wiki notes.

Usage
-----
    python classify.py           # classify all unprocessed captures
    python classify.py --force  # re-classify everything
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from lib import extract, llm, models, storage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Current UTC timestamp in ISO-8601 format with ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _short_id(raw_id: str) -> str:
    """Extract the 8-char UUID portion from a raw capture ID."""
    # raw_id format: 2026-07-06_a1b2c3d4
    parts = raw_id.split("_", 1)
    if len(parts) == 2:
        return parts[1]
    return raw_id


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

def classify_capture(capture: dict) -> models.WikiNote:
    """
    Classify a single raw capture into a WikiNote.

    Parameters
    ----------
    capture : dict
        A capture dict from ``storage.read_raw_captures()``.

    Returns
    -------
    WikiNote
    """
    raw_id = capture["id"]
    meta = capture["meta"]
    note_id = _short_id(raw_id)

    print(f"  Classifying {raw_id} ({meta.get('type', 'unknown')})...")

    # Extract text from the capture
    text = extract.extract_text(capture)
    if not text or not text.strip():
        text = f"[Empty content — raw_id: {raw_id}]"

    # Classify via LLM
    result = llm.classify_content(text)

    para = result["para"]
    tags = result["tags"]
    summary = result["summary"]

    # Clean the body — use the extracted text
    body = text.strip()

    note = models.WikiNote(
        id=note_id,
        raw_id=raw_id,
        para=para,
        tags=tags,
        summary=summary,
        created=_now_iso(),
        links=[],
        body=body,
    )

    # Write the wiki note
    path = storage.write_wiki_note(note)
    print(f"    -> wiki/{para}/{note_id}.md")
    print(f"    para={para}, tags={tags}, summary=\"{summary}\"")

    return note


def classify_all(force: bool = False) -> int:
    """
    Classify all unprocessed raw captures.

    Parameters
    ----------
    force : bool
        If True, re-classify everything (ignore index state).

    Returns
    -------
    int
        Number of captures classified.
    """
    captures = storage.read_raw_captures()
    if not captures:
        print("No raw captures found.")
        return 0

    index = storage.load_index()
    raw_processed = index.get("raw_processed", {})

    count = 0
    for capture in captures:
        raw_id = capture["id"]

        # Skip already-processed unless --force
        if not force and raw_id in raw_processed:
            continue

        try:
            note = classify_capture(capture)

            # Update index
            raw_processed[raw_id] = {
                "note_id": note.id,
                "para": note.para,
                "classified_at": note.created,
            }
            index["raw_processed"] = raw_processed
            storage.save_index(index)  # Save after each capture for robustness
            count += 1

        except Exception as exc:
            print(f"  ERROR classifying {raw_id}: {exc}", file=sys.stderr)
            # Still save what we have so far
            index["raw_processed"] = raw_processed
            storage.save_index(index)

    print(f"\nClassified {count} capture(s).")
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SecondSelf — Classify raw captures into wiki notes",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-classify all captures, even already-processed ones",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SecondSelf — Auto-Classify")
    print("=" * 60)

    classify_all(force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()