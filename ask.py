#!/usr/bin/env python3
"""
SecondSelf — RAG Q&A Engine (Sub-Phase 4.1)

Ask questions in plain English and get answers synthesized from your
personal knowledge base.

Pipeline
--------
1. Embed *question* with ``lib/embeddings.embed_text()``
2. Load all embeddings from ``data/embeddings.pkl``
3. Score every note via cosine similarity → pick top-K
4. Load full wiki bodies for retrieved IDs
5. Build RAG context string (truncated to MAX_CONTEXT_CHARS)
6. Call ``lib/llm.synthesize_answer()`` → get cited answer
7. Return ``AskResult(answer, sources)``

Usage
-----
    python ask.py "What are my career goals?"
    python ask.py "What ML resources have I saved?"
    python ask.py "Summarize my active projects"
    python ask.py --top-k 3 "What do I know about embeddings?"
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List

import numpy as np

from lib.embeddings import (
    cosine_similarity_matrix,
    embed_text,
    load_embeddings,
)
from lib.llm import synthesize_answer
from lib.models import AskResult
from lib.storage import read_wiki_notes

# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 5
MIN_RELEVANCE = 0.20        # notes below this threshold are never used
MAX_CONTEXT_CHARS = 6000    # hard cap for the context block sent to the LLM
NOTE_CHAR_LIMIT = 1200      # per-note truncation before assembling context


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _retrieve(question: str, top_k: int) -> List[Dict]:
    """
    Embed *question* and return up to *top_k* notes ranked by similarity.

    Each returned dict has keys:
        id, para, summary, tags, body, relevance_score
    """
    embeddings: Dict[str, np.ndarray] = load_embeddings()
    if not embeddings:
        return []

    # Stable ordering for matrix construction
    ids = list(embeddings.keys())
    matrix = np.stack([embeddings[i] for i in ids])   # (n, 384)

    q_vec = embed_text(question)
    scores = cosine_similarity_matrix(q_vec, matrix)  # (n,)

    # Sort descending
    ranked_indices = np.argsort(scores)[::-1]

    # Load wiki notes once and index by id
    all_notes: Dict[str, Dict] = {n["id"]: n for n in read_wiki_notes()}

    results = []
    for idx in ranked_indices:
        note_id = ids[idx]
        score = float(scores[idx])

        if score < MIN_RELEVANCE:
            break                    # remaining scores are all lower — stop

        note = all_notes.get(note_id)
        if note is None:
            continue                 # embedding exists but wiki note was deleted

        results.append({
            "id": note_id,
            "para": note.get("para", ""),
            "summary": note.get("summary", ""),
            "tags": note.get("tags", []),
            "body": note.get("body", ""),
            "relevance_score": round(score, 4),
        })

        if len(results) >= top_k:
            break

    return results


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(notes: List[Dict]) -> str:
    """
    Assemble the RAG context block from retrieved notes.

    Each note contributes:
        [note-id] (para · summary)
        <body, truncated to NOTE_CHAR_LIMIT chars>

    The entire block is hard-capped at MAX_CONTEXT_CHARS.
    """
    parts = []
    total = 0

    for note in notes:
        body = note["body"].strip()
        if len(body) > NOTE_CHAR_LIMIT:
            body = body[:NOTE_CHAR_LIMIT].rstrip() + "…"

        block = (
            f"[{note['id']}] ({note['para']} · {note['summary']})\n"
            f"{body}"
        )

        if total + len(block) > MAX_CONTEXT_CHARS:
            # Fit as much of this note as possible
            remaining = MAX_CONTEXT_CHARS - total - 60   # 60-char header budget
            if remaining > 100:
                body = body[:remaining].rstrip() + "…"
                block = (
                    f"[{note['id']}] ({note['para']} · {note['summary']})\n"
                    f"{body}"
                )
                parts.append(block)
            break

        parts.append(block)
        total += len(block) + 2   # +2 for the "\n\n" separator

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask(question: str, top_k: int = DEFAULT_TOP_K) -> AskResult:
    """
    Answer *question* from the personal knowledge base.

    Parameters
    ----------
    question : str
        Plain-English question.
    top_k : int
        Maximum number of notes to retrieve (default 5).

    Returns
    -------
    AskResult
        ``.answer``   — synthesized answer with [note-id] citations
        ``.sources``  — list of ``{id, summary, relevance_score, para}``
    """
    question = question.strip()
    if not question:
        return AskResult(
            answer="Please provide a question.",
            sources=[],
        )

    # 1. Retrieve
    retrieved = _retrieve(question, top_k)

    # 2. No relevant notes → early return
    if not retrieved:
        return AskResult(
            answer="I don't have notes about that.",
            sources=[],
        )

    # 3. Build context
    context = _build_context(retrieved)

    # 4. Synthesize answer
    answer = synthesize_answer(context=context, question=question)

    # 5. Build sources list (id, summary, relevance_score, para)
    sources = [
        {
            "id": n["id"],
            "summary": n["summary"],
            "relevance_score": n["relevance_score"],
            "para": n["para"],
        }
        for n in retrieved
    ]

    return AskResult(answer=answer, sources=sources)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_result(result: AskResult, verbose: bool = False) -> None:
    """Pretty-print an AskResult to stdout."""
    print("\n" + "─" * 60)
    print("Answer\n")
    print(result.answer)

    if result.sources:
        print("\n" + "─" * 60)
        print("Sources\n")
        for s in result.sources:
            score_bar = "█" * int(s["relevance_score"] * 10)
            print(
                f"  [{s['id']}]  {s['para']:10}  "
                f"{score_bar:<10} {s['relevance_score']:.3f}"
            )
            if verbose:
                print(f"             {s['summary']}")
    else:
        print("\n(No sources — answer based on general knowledge or fallback)")

    print("─" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SecondSelf — Ask your personal knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python ask.py "What are my career goals?"
  python ask.py "What ML resources have I saved?"
  python ask.py "Summarize my active projects"
  python ask.py --top-k 3 "What do I know about embeddings?"
  python ask.py --verbose "What is SecondSelf?"
""",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask (omit to enter interactive mode)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        metavar="K",
        help=f"Number of notes to retrieve (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show note summaries alongside sources",
    )

    args = parser.parse_args()

    if args.question:
        # Single-shot mode
        questions = [args.question]
    else:
        # Interactive mode
        print("SecondSelf — Ask mode  (type 'quit' or Ctrl-C to exit)\n")
        questions = []
        try:
            while True:
                q = input("Question: ").strip()
                if not q:
                    continue
                if q.lower() in {"quit", "exit", "q"}:
                    break
                result = ask(q, top_k=args.top_k)
                _print_result(result, verbose=args.verbose)
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
        return

    for q in questions:
        print(f"\nQ: {q}")
        result = ask(q, top_k=args.top_k)
        _print_result(result, verbose=args.verbose)


if __name__ == "__main__":
    main()
