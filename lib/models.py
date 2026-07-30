"""
Shared dataclasses for SecondSelf.

These models are the canonical schemas used across all phases:
  - CaptureMeta / CaptureResult  — Phase 1 (capture)
  - WikiNote                     — Phase 2 (classify / link)
  - GraphNode / GraphEdge        — Phase 3 (graph builder)
  - AskResult                    — Phase 4 (RAG Q&A)
"""

from dataclasses import dataclass, field, asdict, fields
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_dict(data: Dict[str, Any], cls) -> Dict[str, Any]:
    """Return only keys that are valid fields of *cls*."""
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid}


# ---------------------------------------------------------------------------
# Phase 1 — Capture
# ---------------------------------------------------------------------------

@dataclass
class CaptureMeta:
    """Metadata stored as ``raw/{id}/meta.json`` for every capture."""

    id: str
    timestamp: str          # ISO-8601 UTC
    type: str               # "note" | "link" | "file"
    source: str             # "cli" | "stdin" | "path"
    original_filename: str  # original name (for files) or ""
    content_hash: str       # "sha256:..."
    content_filename: str = ""  # e.g. "content.md", "content.txt", "content.pdf"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaptureMeta":
        return cls(**_filter_dict(data, cls))


@dataclass
class CaptureResult:
    """Lightweight return value from capture functions."""

    id: str
    path: str   # absolute path to the capture folder
    type: str   # "note" | "link" | "file"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaptureResult":
        return cls(**_filter_dict(data, cls))


# ---------------------------------------------------------------------------
# Phase 2 — Wiki
# ---------------------------------------------------------------------------

@dataclass
class WikiNote:
    """A classified, linked note stored as ``wiki/{para}/{id}.md``."""

    id: str
    raw_id: str
    para: str                       # Projects | Areas | Resources | Archives
    tags: List[str] = field(default_factory=list)
    summary: str = ""
    created: str = ""               # ISO-8601 UTC
    links: List[str] = field(default_factory=list)
    body: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WikiNote":
        return cls(**_filter_dict(data, cls))


# ---------------------------------------------------------------------------
# Phase 3 — Graph
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    """A node in ``data/graph.json``."""

    id: str
    label: str
    para: str
    tags: List[str] = field(default_factory=list)
    summary: str = ""
    content_preview: str = ""
    group: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphNode":
        return cls(**_filter_dict(data, cls))


@dataclass
class GraphEdge:
    """An edge in ``data/graph.json``."""

    source: str
    target: str
    weight: float = 0.0
    type: str = "semantic"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphEdge":
        return cls(**_filter_dict(data, cls))


# ---------------------------------------------------------------------------
# Phase 4 — Ask
# ---------------------------------------------------------------------------

@dataclass
class AskResult:
    """Return value from ``ask.ask()``."""

    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AskResult":
        return cls(**_filter_dict(data, cls))
