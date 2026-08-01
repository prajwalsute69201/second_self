#!/usr/bin/env python3
"""
SecondSelf — Phase 2.2: The Librarian (Auto-Link)

Find semantically related wiki notes and insert wikilinks.

For each wiki note (new or changed):
    embed(summary + body)
    compare vs all existing embeddings
    if similarity >= THRESHOLD:
        add to frontmatter links[]
        append [[other-id]] in body (deduplicated)
    save embedding to embeddings.pkl

Usage
-----
    python link.py                  # link all unlinked notes
    python link.py --force          # re-link everything
    python link.py --threshold 0.65 # custom threshold
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from lib import embeddings, models, storage

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_embed_text(note: Dict) -> str:
    """
    Build the text to embed for a wiki note.

    Combines summary + body (truncated to keep things fast).
    """
    summary = note.get("summary", "")
    body = note.get("body", "")
    # Truncate body to first 2000 chars for embedding
    body_trunc = body[:2000] if len(body) > 2000 else body
    return f"{summary}\n\n{body_trunc}".strip()


def _extract_wikilinks(body: str) -> List[str]:
    """Extract all [[note-id]] wikilinks from *body*."""
    return re.findall(r'\[\[([a-f0-9]{8})\]\]', body)


def _add_wikilink_to_body(body: str, note_id: str) -> str:
    """
    Append a [[note-id]] wikilink to the body if not already present.
    """
    link = f"[[{note_id}]]"
    if link in body:
        return body
    # Add a "Related" section at the end
    if "## Related" in body:
        # Append to existing Related section
        body = body.rstrip() + f"\n- {link}\n"
    else:
        body = body.rstrip() + f"\n\n## Related\n\n- {link}\n"
    return body


def _update_wiki_note(note: Dict, links: List[str], body: str) -> None:
    """
    Update a wiki note file with new links and body.
    """
    note_path = Path(note["path"])
    wiki_note = models.WikiNote(
        id=note["id"],
        raw_id=note["raw_id"],
        para=note["para"],
        tags=note.get("tags", []),
        summary=note.get("summary", ""),
        created=note.get("created", ""),
        links=links,
        body=body,
    )
    storage.write_wiki_note(wiki_note)


# ---------------------------------------------------------------------------
# Core linking
# ---------------------------------------------------------------------------

def link_note(
    note: Dict,
    all_embeddings: Dict[str, np.ndarray],
    threshold: float = DEFAULT_THRESHOLD,
) -> List[Tuple[str, float]]:
    """
    Find and add links for a single wiki note.

    Parameters
    ----------
    note : dict
        A note dict from ``storage.read_wiki_notes()``.
    all_embeddings : dict
        ``{note_id: np.ndarray}`` — all existing embeddings.
    threshold : float
        Minimum cosine similarity to create a link.

    Returns
    -------
    list[tuple]
        List of ``(other_id, similarity_score)`` for linked notes.
    """
    note_id = note["id"]

    # Build text and embed
    text = _build_embed_text(note)
    vec = embeddings.embed_text(text)

    # Find similar notes
    linked: List[Tuple[str, float]] = []

    for other_id, other_vec in all_embeddings.items():
        if other_id == note_id:
            continue
        sim = embeddings.cosine_similarity(vec, other_vec)
        if sim >= threshold:
            linked.append((other_id, sim))

    # Sort by similarity (highest first)
    linked.sort(key=lambda x: x[1], reverse=True)

    # Get existing links
    existing_links = set(note.get("links", []))
    existing_body_links = set(_extract_wikilinks(note.get("body", "")))
    all_existing = existing_links | existing_body_links

    # Add new links
    new_links = []
    for other_id, sim in linked:
        if other_id not in all_existing:
            new_links.append(other_id)

    # Update note
    updated_links = list(existing_links | set(new_links))
    updated_body = note.get("body", "")
    for other_id in new_links:
        updated_body = _add_wikilink_to_body(updated_body, other_id)

    if new_links:
        _update_wiki_note(note, updated_links, updated_body)

    return linked


def link_all(force: bool = False, threshold: float = DEFAULT_THRESHOLD) -> int:
    """
    Link all wiki notes.

    Parameters
    ----------
    force : bool
        If True, re-embed and re-link all notes.
    threshold : float
        Minimum cosine similarity to create a link.

    Returns
    -------
    int
        Number of notes processed.
    """
    notes = storage.read_wiki_notes()
    if not notes:
        print("No wiki notes found.")
        return 0

    # Load existing embeddings
    all_embeddings = embeddings.load_embeddings()

    # Determine which notes need processing
    to_process = []
    for note in notes:
        note_id = note["id"]
        if force or note_id not in all_embeddings:
            to_process.append(note)

    if not to_process:
        print("All notes already have embeddings. Use --force to re-link.")
        return 0

    print(f"Processing {len(to_process)} note(s)...")

    # Batch embed all notes to process
    texts = [_build_embed_text(note) for note in to_process]
    print("  Embedding notes...")
    vecs = embeddings.embed_texts(texts)

    # Store new embeddings
    for i, note in enumerate(to_process):
        all_embeddings[note["id"]] = vecs[i]

    # Save embeddings after computing
    embeddings.save_embeddings(all_embeddings)
    print(f"  Saved {len(all_embeddings)} embeddings to data/embeddings.pkl")

    # Now link each note
    total_links = 0
    for note in to_process:
        note_id = note["id"]
        vec = all_embeddings[note_id]

        # Find similar notes
        linked: List[Tuple[str, float]] = []
        for other_id, other_vec in all_embeddings.items():
            if other_id == note_id:
                continue
            sim = embeddings.cosine_similarity(vec, other_vec)
            if sim >= threshold:
                linked.append((other_id, sim))

        linked.sort(key=lambda x: x[1], reverse=True)

        # Get existing links
        existing_links = set(note.get("links", []))
        existing_body_links = set(_extract_wikilinks(note.get("body", "")))
        all_existing = existing_links | existing_body_links

        # Add new links
        new_links = []
        for other_id, sim in linked:
            if other_id not in all_existing:
                new_links.append(other_id)

        # Update note
        updated_links = list(existing_links | set(new_links))
        updated_body = note.get("body", "")
        for other_id in new_links:
            updated_body = _add_wikilink_to_body(updated_body, other_id)

        if new_links:
            _update_wiki_note(note, updated_links, updated_body)

        total_links += len(new_links)
        if linked:
            print(f"  {note_id}: {len(linked)} similar, {len(new_links)} new links")
            for other_id, sim in linked[:3]:
                print(f"    -> {other_id} (sim={sim:.3f})")

    print(f"\nLinked {total_links} connection(s) across {len(to_process)} note(s).")
    return len(to_process)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SecondSelf — Link wiki notes via embedding similarity",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-embed and re-link all notes",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Similarity threshold (default: {DEFAULT_THRESHOLD})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SecondSelf — Auto-Link")
    print("=" * 60)
    print(f"Threshold: {args.threshold}")

    link_all(force=args.force, threshold=args.threshold)

    print("\nDone.")


if __name__ == "__main__":
    main()