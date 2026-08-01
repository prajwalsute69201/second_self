#!/usr/bin/env python3
"""
SecondSelf — Streamlit App (Sub-Phase 4.2)

Layout
------
┌────────────────────────────────────────────────────┐
│  🧠 SecondSelf                    [Refresh Graph]  │
├────────────────────────────────────────────────────┤
│  Ask your brain: [________________________] [Ask]  │
│  Answer panel + source citations                   │
├────────────────────────────────────────────────────┤
│  Interactive Knowledge Graph (vis-network)         │
├────────────────────────────────────────────────────┤
│  Sidebar: Capture | Process | Stats                │
└────────────────────────────────────────────────────┘

Usage
-----
    streamlit run app.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import os

import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SecondSelf",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Deployment: surface missing GROQ_API_KEY early as a friendly banner
# instead of letting the app crash on the first LLM call.
# Streamlit Cloud injects secrets as env vars; local dev uses .env.
# ---------------------------------------------------------------------------

def _check_api_key() -> bool:
    """Return True if GROQ_API_KEY is available, else show an error banner."""
    # Streamlit Secrets take precedence; fall back to environment / .env
    key = st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None
    if not key:
        key = os.environ.get("GROQ_API_KEY")
    if not key:
        st.error(
            "**GROQ_API_KEY is not set.**\n\n"
            "- **Streamlit Cloud:** add it under *App settings → Secrets* as "
            "`GROQ_API_KEY = \"gsk_…\"`\n"
            "- **Local:** copy `.env.example` to `.env` and fill in your key.\n\n"
            "The graph view still works, but Ask and pipeline features are disabled.",
            icon="🔑",
        )
        return False
    # Ensure it is visible to sub-modules that call os.environ.get()
    os.environ.setdefault("GROQ_API_KEY", key)
    return True

_GROQ_AVAILABLE = _check_api_key()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
GRAPH_JSON = BASE_DIR / "data" / "graph.json"
GRAPH_HTML = BASE_DIR / "static" / "graph.html"

# ---------------------------------------------------------------------------
# Custom CSS — theme-aware (respects Streamlit light / dark / system mode)
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* ---- main content top padding ---- */
    section[data-testid="stMain"] > div { padding-top: 1rem; }

    /* ---- answer panel: adapts to active theme via CSS vars ---- */
    .ss-answer-box {
        background: var(--secondary-background-color);
        border: 1px solid var(--border-color, rgba(128,128,128,0.3));
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        color: var(--text-color);
        line-height: 1.65;
    }

    /* ---- source chip ---- */
    .ss-source-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--secondary-background-color);
        border: 1px solid var(--border-color, rgba(128,128,128,0.3));
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 0.78rem;
        color: var(--text-color);
        margin: 3px 4px 3px 0;
    }

    /* ---- PARA accent score label ---- */
    .ss-score-label {
        color: #7c3aed;
        font-weight: 600;
    }

    /* ---- PARA colour dot ---- */
    .ss-para-dot {
        width: 8px; height: 8px; border-radius: 50%;
        display: inline-block; flex-shrink: 0;
    }

    /* hide Streamlit default footer ---- */
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# PARA colour map (mirrors graph.html)
# ---------------------------------------------------------------------------

PARA_COLORS = {
    "Projects":  "#7c3aed",
    "Areas":     "#0369a1",
    "Resources": "#065f46",
    "Archives":  "#78350f",
}


def para_dot(para: str) -> str:
    color = PARA_COLORS.get(para, "#374151")
    return f'<span class="ss-para-dot" style="background:{color}"></span>'


# ---------------------------------------------------------------------------
# Cached resource / data loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model…")
def _load_embeddings_cached():
    """Load and cache the embeddings dict (survives reruns)."""
    from lib.embeddings import load_embeddings
    return load_embeddings()


@st.cache_resource(show_spinner="Warming up embedding model…")
def _prewarm_model():
    """
    Load the sentence-transformers model at startup so the first Ask
    query doesn't stall for 30–60 s on a cold Streamlit Cloud instance.
    Called once at module import time; result is cached for the server lifetime.
    """
    from lib.embeddings import load_model
    load_model()

# Pre-warm immediately when the module loads (non-blocking for the UI
# because Streamlit renders the spinner while this runs).
_prewarm_model()


@st.cache_data(show_spinner=False, ttl=None)
def _load_graph_cached() -> dict:
    """
    Load graph.json from disk.

    Cache is busted manually via ``st.cache_data.clear()`` after a
    pipeline run so the graph always reflects the latest state.
    """
    if not GRAPH_JSON.exists():
        return {"nodes": [], "edges": [], "metadata": {}}
    with open(GRAPH_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False, ttl=None)
def _read_graph_html() -> str:
    """Read the vis-network HTML template once."""
    if not GRAPH_HTML.exists():
        return "<p style='color:#f87171'>graph.html not found.</p>"
    with open(GRAPH_HTML, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wiki_stats() -> dict:
    """Return counts of notes by PARA category."""
    from lib.storage import read_wiki_notes
    notes = read_wiki_notes()
    counts: dict[str, int] = {}
    for n in notes:
        counts[n["para"]] = counts.get(n["para"], 0) + 1
    return {"total": len(notes), "by_para": counts}


def _raw_stats() -> int:
    """Return number of raw captures on disk."""
    from lib.storage import read_raw_captures
    return len(read_raw_captures())


def _graph_metadata() -> dict:
    g = _load_graph_cached()
    return g.get("metadata", {})


def _inline_graph_html() -> str:
    """
    Return a self-contained vis-network HTML where graph data is inlined
    so it works inside a Streamlit iframe (blob: URL context where
    relative fetch() and new URL() both fail).

    Strategy: replace the entire async init() body with a version that
    reads from window.__INLINE_GRAPH__ first and never calls new URL() or
    fetch() when that data is present.
    """
    graph_data = _load_graph_cached()
    graph_json_str = json.dumps(graph_data)

    base_html = _read_graph_html()

    # ------------------------------------------------------------------
    # 1. Inject the data payload right before </head>
    # ------------------------------------------------------------------
    inline_data_script = textwrap.dedent(f"""\
        <script>
          // Graph data inlined by Streamlit — avoids fetch() in blob: context
          window.__INLINE_GRAPH__ = {graph_json_str};
        </script>
    """)
    patched = base_html.replace("</head>", inline_data_script + "</head>", 1)

    # ------------------------------------------------------------------
    # 2. Replace the fetch block with an inline-first loader.
    #    The original block is:
    #
    #      const graphUrl = new URL("../data/graph.json", window.location.href).href;
    #      let graphData;
    #      try {
    #        const res = await fetch(graphUrl);
    #        ...
    #        graphData = await res.json();
    #      } catch (err) {
    #        ... show error ...
    #        return;
    #      }
    #
    #    We replace from the graphUrl line through the closing `}` of the
    #    catch block with a version that uses inline data exclusively.
    # ------------------------------------------------------------------
    OLD_FETCH_BLOCK = (
        '    // Resolve path to graph.json relative to this file\n'
        '    const graphUrl = new URL("../data/graph.json", window.location.href).href;\n'
        '\n'
        '    let graphData;\n'
        '    try {\n'
        '      const res = await fetch(graphUrl);\n'
        '      if (!res.ok) throw new Error(`HTTP ${res.status}`);\n'
        '      graphData = await res.json();\n'
        '    } catch (err) {\n'
        '      document.getElementById("loading").style.display = "none";\n'
        '      const errEl = document.getElementById("error-msg");\n'
        '      errEl.style.display = "flex";\n'
        '      document.getElementById("error-text").textContent =\n'
        '        `Could not load graph.json: ${err.message}`;\n'
        '      return;\n'
        '    }'
    )

    NEW_FETCH_BLOCK = textwrap.dedent("""\
        // Use inlined graph data (injected by Streamlit to avoid fetch in blob: context)
        let graphData = window.__INLINE_GRAPH__ || null;
        if (!graphData) {
          document.getElementById("loading").style.display = "none";
          const errEl = document.getElementById("error-msg");
          errEl.style.display = "flex";
          document.getElementById("error-text").textContent =
            "Graph data not available. Run the pipeline and refresh.";
          return;
        }""")

    if OLD_FETCH_BLOCK in patched:
        patched = patched.replace(OLD_FETCH_BLOCK, NEW_FETCH_BLOCK, 1)
    else:
        # Fallback: looser replacement targeting just the URL construction line
        patched = patched.replace(
            'const graphUrl = new URL("../data/graph.json", window.location.href).href;',
            '// graphUrl skipped — using inline data',
        ).replace(
            "let graphData;\n    try {\n      const res = await fetch(graphUrl);\n"
            "      if (!res.ok) throw new Error(`HTTP ${res.status}`);\n"
            "      graphData = await res.json();\n    } catch (err) {\n"
            "      document.getElementById(\"loading\").style.display = \"none\";\n"
            "      const errEl = document.getElementById(\"error-msg\");\n"
            "      errEl.style.display = \"flex\";\n"
            "      document.getElementById(\"error-text\").textContent =\n"
            "        `Could not load graph.json: ${err.message}`;\n"
            "      return;\n    }",
            "let graphData = window.__INLINE_GRAPH__ || null;\n"
            "    if (!graphData) { return; }",
        )

    return patched


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🧠 SecondSelf")
        st.divider()

        # ── Capture ──────────────────────────────────────────────────────
        st.markdown("##### 📥 Capture")

        capture_tab, link_tab, file_tab = st.tabs(["Note", "Link", "File"])

        with capture_tab:
            note_text = st.text_area(
                "Note text",
                placeholder="Type your note here…",
                height=100,
                label_visibility="collapsed",
            )
            if st.button("Capture Note", use_container_width=True, key="btn_capture_note"):
                if note_text.strip():
                    with st.spinner("Capturing…"):
                        from capture import capture_note
                        result = capture_note(note_text.strip())
                    st.success(f"Saved → `{result.id}`")
                else:
                    st.warning("Note text is empty.")

        with link_tab:
            link_url = st.text_input(
                "URL",
                placeholder="https://…",
                label_visibility="collapsed",
                key="input_link_url",
            )
            link_notes = st.text_input(
                "Notes (optional)",
                placeholder="Brief context…",
                label_visibility="collapsed",
                key="input_link_notes",
            )
            if st.button("Capture Link", use_container_width=True, key="btn_capture_link"):
                if link_url.strip():
                    with st.spinner("Capturing…"):
                        from capture import capture_link
                        result = capture_link(link_url.strip(), notes=link_notes.strip())
                    st.success(f"Saved → `{result.id}`")
                else:
                    st.warning("URL is required.")

        with file_tab:
            uploaded = st.file_uploader(
                "Upload file",
                label_visibility="collapsed",
                key="file_uploader",
            )
            if st.button("Capture File", use_container_width=True, key="btn_capture_file"):
                if uploaded is not None:
                    with st.spinner("Capturing…"):
                        # Write to a temp location then capture
                        import tempfile, os
                        suffix = Path(uploaded.name).suffix or ".bin"
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=suffix
                        ) as tmp:
                            tmp.write(uploaded.read())
                            tmp_path = tmp.name
                        try:
                            from capture import capture_file
                            result = capture_file(tmp_path, original_name=uploaded.name)
                            st.success(f"Saved → `{result.id}`")
                        finally:
                            os.unlink(tmp_path)
                else:
                    st.warning("No file selected.")

        st.divider()

        # ── Process ──────────────────────────────────────────────────────
        st.markdown("##### ⚙️ Process")

        col_a, col_b = st.columns(2)
        with col_a:
            force_reprocess = st.checkbox("Force re-run", value=False, key="chk_force")
        with col_b:
            threshold = st.number_input(
                "Link threshold",
                min_value=0.50,
                max_value=0.99,
                value=0.75,
                step=0.05,
                format="%.2f",
                label_visibility="visible",
                key="input_threshold",
            )

        if st.button("▶ Run Full Pipeline", use_container_width=True, key="btn_process"):
            if not _GROQ_AVAILABLE:
                st.warning("Pipeline disabled — GROQ_API_KEY is not configured.", icon="🔑")
            else:
                with st.spinner("Running classify → link → graph…"):
                    from pipeline import run_process
                    run_process(force=force_reprocess, threshold=threshold)
                    # Bust caches so UI reflects new data
                    st.cache_data.clear()
                st.success("Pipeline complete — graph refreshed.")
                st.rerun()

        proc_col1, proc_col2 = st.columns(2)
        with proc_col1:
            if st.button("Classify only", use_container_width=True, key="btn_classify"):
                if not _GROQ_AVAILABLE:
                    st.warning("Disabled — GROQ_API_KEY not configured.", icon="🔑")
                else:
                    with st.spinner("Classifying…"):
                        from pipeline import run_classify
                        run_classify(force=force_reprocess)
                    st.success("Classify done.")

        with proc_col2:
            if st.button("Link only", use_container_width=True, key="btn_link"):
                if not _GROQ_AVAILABLE:
                    st.warning("Disabled — GROQ_API_KEY not configured.", icon="🔑")
                else:
                    with st.spinner("Linking…"):
                        from pipeline import run_link
                        run_link(force=force_reprocess, threshold=threshold)
                    st.cache_data.clear()
                    st.success("Link done.")

        st.divider()

        # ── Stats ─────────────────────────────────────────────────────────
        st.markdown("##### 📊 Stats")

        try:
            wiki     = _wiki_stats()
            raw_count = _raw_stats()
            meta     = _graph_metadata()

            # Top-level counts as native metrics (always readable in any theme)
            c1, c2 = st.columns(2)
            c1.metric("Raw captures", raw_count)
            c2.metric("Wiki notes",   wiki["total"])

            c3, c4 = st.columns(2)
            c3.metric("Graph nodes", meta.get("node_count", "—"))
            c4.metric("Graph edges", meta.get("edge_count", "—"))

            # Per-PARA breakdown
            if wiki["by_para"]:
                st.markdown("**By category**")
                for para, count in sorted(wiki["by_para"].items()):
                    color = PARA_COLORS.get(para, "#374151")
                    st.markdown(
                        f'<span style="display:inline-flex;align-items:center;'
                        f'gap:6px;font-size:0.85rem">'
                        f'<span style="width:10px;height:10px;border-radius:50%;'
                        f'background:{color};display:inline-block;flex-shrink:0"></span>'
                        f"<strong>{para}</strong> — {count}"
                        f"</span>",
                        unsafe_allow_html=True,
                    )

        except Exception as exc:
            st.caption(f"Stats unavailable: {exc}")

        st.divider()
        st.caption("SecondSelf · Phase 4")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def _render_header() -> None:
    col_brand, col_btn = st.columns([5, 1])
    with col_brand:
        st.title("🧠 SecondSelf")
        st.caption("Your personal knowledge brain")
    with col_btn:
        # Vertical spacer so button aligns with the title
        st.markdown("<div style='margin-top:1.1rem'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Graph", key="btn_refresh_graph"):
            with st.spinner("Rebuilding graph…"):
                from build_graph import build_graph
                build_graph()
                st.cache_data.clear()
            st.rerun()

    st.divider()


def _render_ask_section() -> None:
    st.markdown("### Ask your brain")

    ask_col, btn_col = st.columns([6, 1])
    with ask_col:
        question = st.text_input(
            "Question",
            placeholder='e.g. "Ask Your Question?"',
            label_visibility="collapsed",
            key="ask_input",
        )
    with btn_col:
        ask_clicked = st.button("Ask →", use_container_width=True, key="btn_ask")

    # Also trigger on Enter (question changes + button or Enter)
    if ask_clicked and question.strip():
        if not _GROQ_AVAILABLE:
            st.warning("Ask is disabled — GROQ_API_KEY is not configured.", icon="🔑")
        else:
            with st.spinner("Searching your notes…"):
                from ask import ask
                result = ask(question.strip())
            st.session_state["last_answer"] = result
            st.session_state["last_question"] = question.strip()

    # Display persisted answer
    if "last_answer" in st.session_state:
        _render_answer(
            st.session_state["last_question"],
            st.session_state["last_answer"],
        )


def _render_answer(question: str, result) -> None:
    st.markdown(
        f'<div class="ss-answer-box"><strong>Q:</strong> {question}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="ss-answer-box">{result.answer}</div>',
        unsafe_allow_html=True,
    )

    if result.sources:
        chips_html = ""
        for src in result.sources:
            score_pct = int(src["relevance_score"] * 100)
            dot = para_dot(src["para"])
            summary_short = (src["summary"] or src["id"])[:55]
            if len(src["summary"] or "") > 55:
                summary_short += "…"
            chips_html += (
                f'<span class="ss-source-chip">'
                f"  {dot}"
                f"  <strong>[{src['id']}]</strong>"
                f"  {summary_short}"
                f'  <span class="ss-score-label">{score_pct}%</span>'
                f"</span>"
            )
        st.markdown(
            f"<div style='margin-top:0.5rem'>{chips_html}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("No source notes matched your query.")


def _render_graph_section() -> None:
    st.markdown("### Knowledge Graph")

    graph_html = _inline_graph_html()
    graph_data = _load_graph_cached()
    node_count = graph_data.get("metadata", {}).get("node_count", 0)

    if node_count == 0:
        st.info(
            "No graph data yet. Capture some notes, then click "
            "**▶ Run Full Pipeline** in the sidebar (or **🔄 Refresh Graph** above)."
        )
    else:
        components.html(graph_html, height=620, scrolling=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _render_sidebar()
    _render_header()
    _render_ask_section()

    st.divider()

    _render_graph_section()


if __name__ == "__main__":
    main()
