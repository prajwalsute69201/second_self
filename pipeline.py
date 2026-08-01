#!/usr/bin/env python3
"""
SecondSelf — Pipeline Orchestrator

Combines classify and link into a single pipeline.

Usage
-----
    python pipeline.py classify   # classify only
    python pipeline.py link       # link only
    python pipeline.py process    # classify + link
"""

from __future__ import annotations

import argparse
import sys


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def run_classify(force: bool = False) -> int:
    """Run the classification step."""
    from classify import classify_all
    print("\n--- Step 1: Classify ---\n")
    return classify_all(force=force)


def run_link(force: bool = False, threshold: float = 0.75) -> int:
    """Run the linking step."""
    from link import link_all
    print("\n--- Step 2: Link ---\n")
    return link_all(force=force, threshold=threshold)


def run_graph() -> None:
    """Run the graph build step."""
    from build_graph import build_graph
    print("\n--- Step 3: Build Graph ---\n")
    build_graph()


def run_process(force: bool = False, threshold: float = 0.75) -> None:
    """Run the full pipeline: classify + link + graph."""
    run_classify(force=force)
    run_link(force=force, threshold=threshold)
    run_graph()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SecondSelf — Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python pipeline.py classify   # classify only
  python pipeline.py link       # link only
  python pipeline.py graph      # build graph.json only
  python pipeline.py process    # classify + link + graph
  python pipeline.py process --force  # re-process everything
""",
    )
    subparsers = parser.add_subparsers(dest="command")

    # classify
    cls_parser = subparsers.add_parser("classify", help="Classify raw captures")
    cls_parser.add_argument(
        "--force", action="store_true",
        help="Re-classify all captures",
    )

    # link
    link_parser = subparsers.add_parser("link", help="Link wiki notes")
    link_parser.add_argument(
        "--force", action="store_true",
        help="Re-embed and re-link all notes",
    )
    link_parser.add_argument(
        "--threshold", type=float, default=0.75,
        help="Similarity threshold (default: 0.75)",
    )

    # graph
    subparsers.add_parser("graph", help="Build graph.json from wiki notes")

    # process
    proc_parser = subparsers.add_parser("process", help="Classify + Link + Graph")
    proc_parser.add_argument(
        "--force", action="store_true",
        help="Re-process everything",
    )
    proc_parser.add_argument(
        "--threshold", type=float, default=0.75,
        help="Similarity threshold (default: 0.75)",
    )

    args = parser.parse_args()

    if args.command == "classify":
        run_classify(force=args.force)
    elif args.command == "link":
        run_link(force=args.force, threshold=args.threshold)
    elif args.command == "graph":
        run_graph()
    elif args.command == "process":
        run_process(force=args.force, threshold=args.threshold)
    else:
        parser.print_help()
        sys.exit(1)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()