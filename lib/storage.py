"""
Filesystem helpers for SecondSelf.

All read/write operations against ``raw/``, ``wiki/`` and ``data/`` live here
so that every later phase has a single, consistent I/O layer.
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .models import CaptureMeta, CaptureResult, WikiNote

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw"
WIKI_DIR = BASE_DIR / "wiki"
DATA_DIR = BASE_DIR / "data"
INDEX_FILE = DATA_DIR / "index.json"

PARA_CATEGORIES = ["Projects", "Areas", "Resources", "Archives"]


# ---------------------------------------------------------------------------
# ID / hash helpers
# ---------------------------------------------------------------------------

def generate_capture_id() -> str:
    """Return ``{YYYY-MM-DD}_{uuid8}`` — e.g. ``2026-07-06_a1b2c3d4``."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    uuid8 = uuid.uuid4().hex[:8]
    return f"{date_str}_{uuid8}"


def content_hash(data: bytes) -> str:
    """SHA-256 hash of *data*, prefixed with ``sha256:`` for clarity."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _content_filename(meta: CaptureMeta) -> str:
    """Determine the on-disk filename for a capture's content."""
    if meta.content_filename:
        return meta.content_filename
    if meta.type == "note":
        return "content.md"
    if meta.type == "link":
        return "content.txt"
    if meta.type == "file":
        ext = Path(meta.original_filename).suffix
        return f"content{ext}" if ext else "content.bin"
    return "content.bin"


# ---------------------------------------------------------------------------
# Raw captures
# ---------------------------------------------------------------------------

def write_raw_capture(meta: CaptureMeta, content: bytes) -> CaptureResult:
    """
    Create ``raw/{id}/`` with ``meta.json`` and the content file.

    Parameters
    ----------
    meta : CaptureMeta
        Metadata for the capture (id, timestamp, type, …).
    content : bytes
        Raw content bytes (text encoded or binary).
    """
    capture_dir = RAW_DIR / meta.id
    capture_dir.mkdir(parents=True, exist_ok=True)

    # meta.json
    meta_path = capture_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta.to_dict(), f, indent=2, ensure_ascii=False)

    # content file
    fname = _content_filename(meta)
    content_path = capture_dir / fname
    with open(content_path, "wb") as f:
        f.write(content)

    return CaptureResult(id=meta.id, path=str(capture_dir), type=meta.type)


def read_raw_captures() -> List[Dict]:
    """
    List every capture in ``raw/``.

    Returns a list of dicts, each with keys:
        ``id``, ``meta`` (dict), ``dir`` (Path)
    """
    captures: List[Dict] = []
    if not RAW_DIR.exists():
        return captures

    for capture_dir in sorted(RAW_DIR.iterdir()):
        if not capture_dir.is_dir():
            continue
        meta_path = capture_dir / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        captures.append({
            "id": capture_dir.name,
            "meta": meta,
            "dir": capture_dir,
        })
    return captures


# ---------------------------------------------------------------------------
# Wiki notes
# ---------------------------------------------------------------------------

def write_wiki_note(note: WikiNote) -> Path:
    """
    Write ``wiki/{para}/{id}.md`` with YAML frontmatter + body.

    The ``para`` directory is created if it does not exist.
    """
    para_dir = WIKI_DIR / note.para
    para_dir.mkdir(parents=True, exist_ok=True)

    note_path = para_dir / f"{note.id}.md"

    frontmatter = {
        "id": note.id,
        "raw_id": note.raw_id,
        "para": note.para,
        "tags": note.tags,
        "summary": note.summary,
        "created": note.created,
        "links": note.links,
    }

    with open(note_path, "w", encoding="utf-8") as f:
        f.write("---\n")
        yaml.dump(frontmatter, f, default_flow_style=False, allow_unicode=True)
        f.write("---\n\n")
        f.write(note.body)

    return note_path


def read_wiki_notes() -> List[Dict]:
    """
    Parse every ``wiki/**/*.md`` file.

    Returns a list of dicts with keys:
        ``id``, ``raw_id``, ``para``, ``tags``, ``summary``,
        ``created``, ``links``, ``body``, ``path``

    Duplicate IDs (same note filed under two PARA folders) are
    deduplicated: the first occurrence wins and a warning is printed.
    """
    notes: List[Dict] = []
    seen_ids: set = set()

    if not WIKI_DIR.exists():
        return notes

    for para_dir in sorted(WIKI_DIR.iterdir()):
        if not para_dir.is_dir():
            continue
        for note_path in sorted(para_dir.glob("*.md")):
            with open(note_path, "r", encoding="utf-8") as f:
                content = f.read()

            frontmatter: Dict = {}
            body = content

            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    try:
                        frontmatter = yaml.safe_load(parts[1]) or {}
                    except yaml.YAMLError:
                        frontmatter = {}
                    body = parts[2].strip()

            note_id = frontmatter.get("id", note_path.stem)

            # Skip duplicates — same ID already loaded from another PARA folder
            if note_id in seen_ids:
                print(
                    f"[storage] WARNING: duplicate id '{note_id}' at "
                    f"'{note_path}' — skipped (first occurrence kept)."
                )
                continue
            seen_ids.add(note_id)

            notes.append({
                "id": note_id,
                "raw_id": frontmatter.get("raw_id", ""),
                "para": frontmatter.get("para", para_dir.name),
                "tags": frontmatter.get("tags", []) or [],
                "summary": frontmatter.get("summary", ""),
                "created": frontmatter.get("created", ""),
                "links": frontmatter.get("links", []) or [],
                "body": body,
                "path": str(note_path),
            })
    return notes


# ---------------------------------------------------------------------------
# Index (data/index.json)
# ---------------------------------------------------------------------------

def load_index() -> Dict:
    """Read ``data/index.json``; return a default skeleton if missing."""
    if not INDEX_FILE.exists():
        return {
            "raw_processed": {},
            "embeddings_version": "all-MiniLM-L6-v2",
            "last_graph_build": None,
        }
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_index(index: Dict) -> None:
    """Write *index* to ``data/index.json``."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
