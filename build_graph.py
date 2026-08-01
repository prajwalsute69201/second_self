#!/usr/bin/env python3
"""
SecondSelf — Graph Builder (Sub-Phase 3.1)

Converts the linked wiki into a graph data structure suitable for
force-directed visualisation.

Algorithm
---------
1. Parse every ``wiki/**/*.md`` → one ``GraphNode`` per note.
2. Extract edges from two sources:
   a. ``links[]`` frontmatter list (written by ``link.py``)
   b. ``[[id]]`` wikilink patterns in the body text
3. Deduplicate edges using the canonical key
   ``(min(source, target), max(source, target))``.
4. Enrich each node:
   - ``label``   = summary (falling back to id)
   - ``group``   = para category
   - ``content_preview`` = first 200 chars of body
5. Export ``data/graph.json`` and update ``data/index.json``.

Usage
-----
    python build_graph.py
    python build_graph.py --output data/graph.json   # explicit path
    python build_graph.py --pretty                   # indent=4 (default: 2)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

from lib.models import GraphEdge, GraphNode
from lib.storage import DATA_DIR, load_index, read_wiki_notes, save_index

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WIKILINK_RE = re.compile(r"\[\[([a-f0-9]{8})\]\]")  # [[abcd1234]]
DEFAULT_OUTPUT = DATA_DIR / "graph.json"
CONTENT_PREVIEW_CHARS = 200


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _content_preview(body: str) -> str:
    """Return the first ``CONTENT_PREVIEW_CHARS`` characters of *body*."""
    preview = body.strip()
    if len(preview) > CONTENT_PREVIEW_CHARS:
        preview = preview[:CONTENT_PREVIEW_CHARS].rstrip() + "…"
    return preview


def _canonical_edge_key(source: str, target: str) -> Tuple[str, str]:
    """Deterministic edge key that treats (A→B) and (B→A) as the same edge."""
    return (min(source, target), max(source, target))


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_graph(output_path: Path = DEFAULT_OUTPUT, indent: int = 2) -> Dict:
    """
    Parse wiki notes and build ``graph.json``.

    Parameters
    ----------
    output_path : Path
        Destination file for the exported JSON.
    indent : int
        JSON indentation level.

    Returns
    -------
    dict
        The graph dict (``nodes``, ``edges``, ``metadata``).
    """
    raw_notes = read_wiki_notes()
    if not raw_notes:
        print("No wiki notes found — run `python pipeline.py process` first.")
        return {"nodes": [], "edges": [], "metadata": {}}

    # Deduplicate notes by ID before processing — storage.py already warns,
    # but build_graph adds a second safety net so graph.json is always clean.
    seen: Set[str] = set()
    deduped_notes = []
    for note in raw_notes:
        if note["id"] in seen:
            print(
                f"[build_graph] WARNING: duplicate id '{note['id']}' "
                f"at '{note.get('path', '?')}' — skipped."
            )
            continue
        seen.add(note["id"])
        deduped_notes.append(note)
    raw_notes = deduped_notes

    # Build a set of valid IDs for edge validation
    valid_ids: Set[str] = {n["id"] for n in raw_notes}

    # ------------------------------------------------------------------ nodes
    nodes: List[GraphNode] = []
    for note in raw_notes:
        node = GraphNode(
            id=note["id"],
            label=note["summary"] or note["id"],
            para=note["para"],
            tags=note["tags"],
            summary=note["summary"],
            content_preview=_content_preview(note["body"]),
            group=note["para"],  # group mirrors para for vis-network colouring
        )
        nodes.append(node)

    # ------------------------------------------------------------------ edges
    # Use a dict keyed by canonical tuple to deduplicate
    edge_map: Dict[Tuple[str, str], GraphEdge] = {}

    def _add_edge(source: str, target: str, edge_type: str = "semantic") -> None:
        """Add or update an edge; skip self-loops and unknown IDs."""
        if source == target:
            return
        if source not in valid_ids or target not in valid_ids:
            return
        key = _canonical_edge_key(source, target)
        if key not in edge_map:
            edge_map[key] = GraphEdge(
                source=key[0],
                target=key[1],
                weight=1.0,
                type=edge_type,
            )
        else:
            # Bump weight for each additional reference
            edge_map[key].weight = round(edge_map[key].weight + 0.5, 2)

    for note in raw_notes:
        source_id = note["id"]

        # Source 1 — links[] frontmatter
        for linked_id in note.get("links", []):
            _add_edge(source_id, linked_id, edge_type="semantic")

        # Source 2 — [[id]] wikilink patterns in body
        for match in WIKILINK_RE.finditer(note.get("body", "")):
            linked_id = match.group(1)
            _add_edge(source_id, linked_id, edge_type="wikilink")

    edges = list(edge_map.values())

    # ---------------------------------------------------------------- export
    generated_at = datetime.now(timezone.utc).isoformat()
    graph = {
        "nodes": [n.to_dict() for n in nodes],
        "edges": [e.to_dict() for e in edges],
        "metadata": {
            "generated_at": generated_at,
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=indent, ensure_ascii=False)

    # Update index.json timestamp
    index = load_index()
    index["last_graph_build"] = generated_at
    save_index(index)

    return graph


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SecondSelf — Build knowledge graph from wiki notes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python build_graph.py
  python build_graph.py --output data/graph.json
  python build_graph.py --pretty
""",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path for graph.json (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Use indent=4 instead of the default indent=2",
    )
    args = parser.parse_args()

    indent = 4 if args.pretty else 2
    graph = build_graph(output_path=args.output, indent=indent)

    meta = graph.get("metadata", {})
    if meta:
        print(f"Graph built → {args.output}")
        print(f"  Nodes : {meta['node_count']}")
        print(f"  Edges : {meta['edge_count']}")
        print(f"  At    : {meta['generated_at']}")
    else:
        print("Graph is empty — nothing to export.")
        sys.exit(1)


if __name__ == "__main__":
    main()
