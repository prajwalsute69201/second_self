# SecondSelf — Edge Cases & Corner Scenarios

A comprehensive catalog of all corner scenarios, failure modes, and edge cases for the SecondSelf personal knowledge brain. Derived from `architecture.md`, `Implementationplan.md`, and `ProblemStatement.md`, organized by component and phase.

---

## Table of Contents

1. [Capture Phase (`capture.py`)](#1-capture-phase-capturepy)
2. [Classification Phase (`classify.py`, `lib/llm.py`)](#2-classification-phase-classifypy-libllmpy)
3. [Linking Phase (`link.py`, `lib/embeddings.py`)](#3-linking-phase-linkpy-libembeddingspy)
4. [Graph Building (`build_graph.py`)](#4-graph-building-build_graphpy)
5. [Ask / RAG Engine (`ask.py`)](#5-ask--rag-engine-askpy)
6. [Streamlit App (`app.py`)](#6-streamlit-app-apppy)
7. [Pipeline & Orchestration (`pipeline.py`)](#7-pipeline--orchestration-pipelinepy)
8. [Storage & Data Integrity (`lib/storage.py`)](#8-storage--data-integrity-libstoragesy)
9. [Deployment](#9-deployment)
10. [Cross-Cutting Concerns](#10-cross-cutting-concerns)

---

## 1. Capture Phase (`capture.py`)

### 1.1 File Does Not Exist

| Field | Detail |
|-------|--------|
| **Scenario** | `python capture.py file /nonexistent/path.pdf` |
| **Impact** | Capture fails silently or crashes if not handled |
| **Detection** | `Path.exists()` check before processing |
| **Handling** | Print clear error message (`File not found: /nonexistent/path.pdf`), exit with code 1 |

### 1.2 Empty Note Text

| Field | Detail |
|-------|--------|
| **Scenario** | `python capture.py note ""` or `python capture.py note` (no argument) |
| **Impact** | Creates a meaningless empty capture; wastes storage |
| **Detection** | Check `text.strip()` is non-empty before proceeding |
| **Handling** | Reject with message: `Error: note text cannot be empty` |

### 1.3 Binary File Capture

| Field | Detail |
|-------|--------|
| **Scenario** | Capturing a binary file (PDF, image, executable) via `capture_file` |
| **Impact** | File must be stored byte-for-byte; text extraction deferred to classify stage |
| **Detection** | File extension and/or content sniffing |
| **Handling** | Copy file as-is to `raw/{id}/content.{ext}`; record `original_filename` in `meta.json`; defer text extraction to `classify.py` |

### 1.4 Duplicate Content (Same URL or Same Text)

| Field | Detail |
|-------|--------|
| **Scenario** | User captures the same URL or identical note text twice |
| **Impact** | Duplicate entries in `raw/`; redundant processing later |
| **Detection** | SHA-256 hash of content compared against `index.json["raw_processed"]` |
| **Handling** | Warn the user (`Warning: content with hash sha256:... was already captured at raw/{existing_id}`); still allow capture (user may want a duplicate with different context) |

### 1.5 Duplicate Capture (Same URL)

| Field | Detail |
|-------|--------|
| **Scenario** | Same URL captured multiple times |
| **Impact** | Redundant URL fetches and classification |
| **Detection** | Hash check in `index.json` |
| **Handling** | Skip or warn (per architecture Section 11); allow user to override |

### 1.6 File With No Extension

| Field | Detail |
|-------|--------|
| **Scenario** | `python capture.py file ./README` (no extension) |
| **Impact** | Cannot determine content type or storage filename |
| **Detection** | Check file extension via `Path.suffix` |
| **Handling** | Store as `content` (no extension) or detect MIME type via `mimetypes` module; record detected type in `meta.json` |

### 1.7 Symlink as Input File

| Field | Detail |
|-------|--------|
| **Scenario** | `python capture.py file ./symlink_to_pdf` where the path is a symlink |
| **Impact** | Copying a symlink may copy the link, not the target; or may follow it unexpectedly |
| **Detection** | `Path.is_symlink()` check |
| **Handling** | Resolve symlink to real path (`Path.resolve()`) before copying; store the resolved path in `meta.json` |

### 1.8 Path Traversal / Malicious File Path

| Field | Detail |
|-------|--------|
| **Scenario** | `python capture.py file ../../etc/passwd` or paths with `..` segments |
| **Impact** | Could read/write outside intended `raw/` directory |
| **Detection** | Validate resolved path is within expected bounds |
| **Handling** | Reject paths containing `..` or that resolve outside the project directory; sanitize filename |

### 1.9 Disk Full During Capture

| Field | Detail |
|-------|--------|
| **Scenario** | Writing `meta.json` or content file fails because disk is full |
| **Impact** | Partial capture; corrupted `raw/{id}/` folder |
| **Detection** | `OSError` / `IOError` during write |
| **Handling** | Clean up partial `raw/{id}/` directory on failure; print error message; exit with code 1 |

### 1.10 Permission Denied

| Field | Detail |
|-------|--------|
| **Scenario** | `raw/` directory is read-only or user lacks write permissions |
| **Impact** | Cannot create capture folder or write files |
| **Detection** | `PermissionError` during file operations |
| **Handling** | Print clear error (`Permission denied: cannot write to raw/`); exit with code 1 |

### 1.11 Interactive Stdin Mode With No Input

| Field | Detail |
|-------|--------|
| **Scenario** | `python capture.py` (interactive mode) but stdin is empty or closed (e.g., piped from `/dev/null`) |
| **Impact** | Script hangs waiting for input or crashes |
| **Detection** | Check `sys.stdin.isatty()` or handle `EOFError` |
| **Handling** | If stdin is not a TTY and is empty, print usage/help and exit; if EOF received, exit gracefully |

### 1.12 Very Long Note Text

| Field | Detail |
|-------|--------|
| **Scenario** | Note text is extremely long (e.g., pasted entire book chapter) |
| **Impact** | Large `content.md` file; downstream LLM classification may hit token limits |
| **Detection** | Check character/token count |
| **Handling** | Store full text in `raw/`; truncation happens at classify stage (architecture Section 11: "Truncate for LLM; full text kept in wiki file") |

### 1.13 Unicode / Encoding Issues in Captured Content

| Field | Detail |
|-------|--------|
| **Scenario** | Note text or file content contains non-UTF-8 bytes or unusual Unicode |
| **Impact** | `UnicodeDecodeError` when reading/writing content |
| **Detection** | Exception during encoding/decoding |
| **Handling** | Read/write with `encoding="utf-8"` and `errors="replace"` or `"surrogateescape"`; log encoding issues |

---

## 2. Classification Phase (`classify.py`, `lib/llm.py`)

### 2.1 LLM Returns Invalid JSON

| Field | Detail |
|-------|--------|
| **Scenario** | Groq LLM response is not valid JSON (e.g., contains explanatory text, trailing commas, or is truncated) |
| **Impact** | `json.loads()` fails; classification cannot proceed |
| **Detection** | `json.JSONDecodeError` during parsing |
| **Handling** | Retry once with a stricter prompt; if still invalid, fallback to `para: Resources`, `tags: []`, `summary: <first 100 chars of content>` (architecture Section 11) |

### 2.2 LLM Returns Valid JSON But Missing Required Keys

| Field | Detail |
|-------|--------|
| **Scenario** | LLM returns `{"para": "Projects"}` but omits `tags` and `summary` |
| **Impact** | `KeyError` when accessing missing fields; incomplete wiki note |
| **Detection** | Schema validation after JSON parse |
| **Handling** | Validate all required keys (`para`, `tags`, `summary`); if missing, retry or apply defaults (`tags: []`, `summary: <truncated content>`) |

### 2.3 LLM Returns Invalid PARA Category

| Field | Detail |
|-------|--------|
| **Scenario** | LLM returns `para: "Research"` which is not one of `Projects | Areas | Resources | Archives` |
| **Impact** | Wiki note written to non-standard folder; graph grouping breaks |
| **Detection** | Check `para` value against allowed set |
| **Handling** | Map unknown categories to closest valid one; if no match, default to `Resources` |

### 2.4 URL Fetch Fails (Network Error)

| Field | Detail |
|-------|--------|
| **Scenario** | `requests.get(url)` raises `ConnectionError`, `Timeout`, or returns HTTP 4xx/5xx |
| **Impact** | Cannot extract page text for classification |
| **Detection** | `requests.exceptions.RequestException` or non-200 status code |
| **Handling** | Store the URL only; classify from the URL string itself (architecture Section 11: "Store URL only; classify from URL string") |

### 2.5 URL Fetch Returns Non-HTML Content

| Field | Detail |
|-------|--------|
| **Scenario** | URL points to a PDF, image, or binary file (e.g., `https://example.com/doc.pdf`) |
| **Impact** | `beautifulsoup4` HTML stripping produces garbage |
| **Detection** | Check `Content-Type` header; check file extension in URL |
| **Handling** | If content is not HTML, store URL only and classify from URL string; optionally download and defer to file extraction |

### 2.6 PDF Text Extraction Fails

| Field | Detail |
|-------|--------|
| **Scenario** | `pypdf` cannot extract text from a scanned PDF or corrupted PDF |
| **Impact** | No text available for classification; summary is empty |
| **Detection** | `pypdf.PdfReadError` or empty extracted text |
| **Handling** | Store the file; set `summary` to the original filename; classify from filename only (architecture Section 11: "Store file; summary = filename only") |

### 2.7 Very Long Note Exceeds LLM Token Limit

| Field | Detail |
|-------|--------|
| **Scenario** | Captured note text is 50,000+ tokens; LLM context window is ~8,192 tokens |
| **Impact** | LLM call fails or produces poor results due to truncation |
| **Detection** | Token count exceeds model limit (e.g., 4,000 tokens for safety) |
| **Handling** | Truncate to first ~4,000 tokens before sending to LLM (architecture Section 4.2: "Token truncation for long captures (keep first ~4000 tokens)"); keep full text in wiki file |

### 2.8 Groq API Key Missing or Invalid

| Field | Detail |
|-------|--------|
| **Scenario** | `GROQ_API_KEY` not set in environment or `.env` file, or key is expired/revoked |
| **Impact** | All LLM calls fail; classification and RAG are blocked |
| **Detection** | `AuthenticationError` from Groq client; `KeyError` or `None` when reading env var |
| **Handling** | Print clear error (`GROQ_API_KEY not found. Copy .env.example to .env and add your key.`); exit with code 1 |

### 2.9 Groq API Rate Limit Exceeded

| Field | Detail |
|-------|--------|
| **Scenario** | Too many LLM calls in a short period; Groq returns HTTP 429 |
| **Impact** | Classification or answer synthesis fails temporarily |
| **Detection** | HTTP 429 status code or `RateLimitError` from Groq client |
| **Handling** | Retry with exponential backoff (architecture Section 4.2: "Retry with exponential backoff"); batch classify calls to reduce request frequency (Implementationplan Risk Register) |

### 2.10 LLM Returns Empty or Whitespace-Only Summary

| Field | Detail |
|-------|--------|
| **Scenario** | LLM returns `{"para": "Resources", "tags": [], "summary": "   "}` |
| **Impact** | Wiki note has no meaningful summary; graph node label is blank |
| **Detection** | Check `summary.strip()` is non-empty |
| **Handling** | If summary is empty/whitespace, generate one from the first sentence or first 100 characters of the content |

### 2.11 LLM Hallucinates Tags or Summary

| Field | Detail |
|-------|--------|
| **Scenario** | LLM invents tags or a summary that doesn't match the content |
| **Impact** | Misleading wiki notes; poor search results |
| **Detection** | Manual review (no automated detection) |
| **Handling** | Document as known limitation; user can manually edit wiki notes; consider prompt refinement |

### 2.12 Non-English Content

| Field | Detail |
|-------|--------|
| **Scenario** | Captured note is in a language other than English |
| **Impact** | LLM classification quality may degrade; embeddings may be less accurate |
| **Detection** | Language detection (e.g., `langdetect` library) or user-specified |
| **Handling** | Document as known limitation; Llama 3.1 8B supports multiple languages but quality varies |

---

## 3. Linking Phase (`link.py`, `lib/embeddings.py`)

### 3.1 No Similar Notes Found

| Field | Detail |
|-------|--------|
| **Scenario** | A new note has no semantic similarity above threshold to any existing note |
| **Impact** | Note stands alone in the graph; no links inserted |
| **Detection** | Max similarity score below threshold (e.g., 0.75) |
| **Handling** | Note is stored with empty `links: []`; appears as isolated node in graph (architecture Section 11: "Note stands alone; graph is a single node") |

### 3.2 Embedding Model Fails to Load

| Field | Detail |
|-------|--------|
| **Scenario** | `sentence-transformers/all-MiniLM-L6-v2` cannot be downloaded (no internet) or cached model is corrupted |
| **Impact** | Cannot compute embeddings; linking and RAG are blocked |
| **Detection** | `OSError` or `FileNotFoundError` from `SentenceTransformer()` |
| **Handling** | Print clear error (`Failed to load embedding model. Check internet connection or cache.`); exit with code 1; consider pre-downloading model in CI |

### 3.3 All-Zero Embeddings (Division by Zero in Cosine Similarity)

| Field | Detail |
|-------|--------|
| **Scenario** | Embedding vector is all zeros (e.g., empty text input to model) |
| **Impact** | `cosine_similarity` divides by zero (norm = 0); `NaN` or `inf` result |
| **Detection** | Check `np.linalg.norm(a) == 0` before division |
| **Handling** | Return similarity score of `0.0` if either vector has zero norm; skip linking for empty-content notes |

### 3.4 NaN in Embeddings

| Field | Detail |
|-------|--------|
| **Scenario** | Numerical instability produces `NaN` values in embedding vectors |
| **Impact** | Cosine similarity with `NaN` produces `NaN`; sorting/retrieval breaks |
| **Detection** | `np.isnan(embedding).any()` check |
| **Handling** | Re-embed the note; if persists, skip the note and log a warning |

### 3.5 Embedding Dimension Mismatch

| Field | Detail |
|-------|--------|
| **Scenario** | Existing `embeddings.pkl` was generated with a different model (e.g., 768-dim) but current model produces 384-dim vectors |
| **Impact** | Cosine similarity fails or produces garbage; `index.json["embeddings_version"]` mismatch |
| **Detection** | Compare vector shape against expected dimension (384 for all-MiniLM-L6-v2) |
| **Handling** | Check `index.json["embeddings_version"]`; if mismatch, delete `embeddings.pkl` and recompute all embeddings |

### 3.6 Self-Referencing Link

| Field | Detail |
|-------|--------|
| **Scenario** | A note's embedding is most similar to itself (e.g., exact duplicate content) |
| **Impact** | Note links to itself: `[[self-id]]` in body and `links: [self-id]` in frontmatter |
| **Detection** | Check `source_id == target_id` before inserting link |
| **Handling** | Skip self-links; never insert a link where source and target are the same note |

### 3.7 Circular Wikilinks

| Field | Detail |
|-------|--------|
| **Scenario** | Note A links to Note B, and Note B links to Note A |
| **Impact** | Graph has bidirectional edges; potential infinite loops in traversal algorithms |
| **Detection** | During graph traversal, track visited nodes |
| **Handling** | Allow circular links (they represent real relationships); ensure graph traversal algorithms use visited-set to prevent infinite loops |

### 3.8 Invalid Wikilink Target

| Field | Detail |
|-------|--------|
| **Scenario** | Body contains `[[nonexistent-id]]` where no wiki note with that ID exists |
| **Impact** | Broken link in graph; `build_graph.py` may create a node for a non-existent note |
| **Detection** | Check if target ID exists in wiki notes during graph building |
| **Handling** | In `build_graph.py`, skip edges to non-existent nodes; optionally warn about broken links |

### 3.9 Duplicate Wikilinks in Body

| Field | Detail |
|-------|--------|
| **Scenario** | `[[other-id]]` appears multiple times in the same note body |
| **Impact** | Redundant links; graph may have duplicate edges |
| **Detection** | Check for existing `[[id]]` before appending |
| **Handling** | Deduplicate: only insert `[[id]]` once per note body (architecture Section 4.3: "deduplicated") |

### 3.10 Duplicate Links in Frontmatter

| Field | Detail |
|-------|--------|
| **Scenario** | `links: [b2c3d4e5, b2c3d4e5]` — same ID appears twice in frontmatter |
| **Impact** | Redundant edges in graph |
| **Detection** | Check for existing ID in `links` list before appending |
| **Handling** | Deduplicate `links` list before writing frontmatter |

### 3.11 Similarity Threshold Too Low (Too Many Links)

| Field | Detail |
|-------|--------|
| **Scenario** | Threshold set to 0.65; every note links to 10+ others |
| **Impact** | Graph is overly dense; links are noisy and not meaningful |
| **Detection** | Average links per note is very high; user reports noise |
| **Handling** | Raise threshold to 0.75 or 0.80 (architecture Section 4.3: "raise to 0.80 if too noisy") |

### 3.12 Similarity Threshold Too High (Too Few Links)

| Field | Detail |
|-------|--------|
| **Scenario** | Threshold set to 0.90; most notes have no links |
| **Impact** | Graph is sparse; few relationships detected |
| **Detection** | Most notes have 0 links; graph has many isolated nodes |
| **Handling** | Lower threshold to 0.75 or 0.65 (architecture Section 4.3: "lower to 0.65 if graph feels sparse") |

### 3.13 Corrupted `embeddings.pkl`

| Field | Detail |
|-------|--------|
| **Scenario** | `data/embeddings.pkl` is corrupted (e.g., interrupted write, disk error) |
| **Impact** | `pickle.load()` fails; linking and RAG cannot retrieve embeddings |
| **Detection** | `pickle.UnpicklingError` or `EOFError` during load |
| **Handling** | Delete corrupted `embeddings.pkl`; recompute all embeddings from wiki notes |

### 3.14 Embedding Model Cold Start (First Load Slow)

| Field | Detail |
|-------|--------|
| **Scenario** | First call to `SentenceTransformer()` takes 10+ seconds to download/load model |
| **Impact** | Slow first classification or RAG query |
| **Detection** | Timing the model load |
| **Handling** | Use `@st.cache_resource` in Streamlit to load model once (architecture Section 8: "first query slow (~10s)") |

---

## 4. Graph Building (`build_graph.py`)

### 4.1 Empty Wiki Directory

| Field | Detail |
|-------|--------|
| **Scenario** | `wiki/` has no markdown files (no notes have been classified yet) |
| **Impact** | `graph.json` has zero nodes and zero edges |
| **Detection** | `read_wiki_notes()` returns empty list |
| **Handling** | Write `graph.json` with empty `nodes` and `edges` arrays; include metadata with zero counts; Streamlit app should display "No notes yet" message |

### 4.2 Single Node Graph

| Field | Detail |
|-------|--------|
| **Scenario** | Only one note exists in `wiki/` |
| **Impact** | Graph has one node and zero edges |
| **Detection** | `node_count == 1` |
| **Handling** | Render single node in vis-network; ensure physics doesn't cause issues with single node |

### 4.3 Graph Too Cluttered (Too Many Nodes/Edges)

| Field | Detail |
|-------|--------|
| **Scenario** | Hundreds of notes with many links; graph is visually overwhelming |
| **Impact** | vis-network performance