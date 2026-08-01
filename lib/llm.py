"""
LLM helpers for SecondSelf.

Wraps the Groq API for content classification and RAG answer synthesis.
Model: ``llama-3.1-8b-instant``
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "llama-3.1-8b-instant"
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds

# Load .env once at import time
load_dotenv()

_client: Optional[Groq] = None


def _get_client() -> Groq:
    """Return a cached Groq client (created lazily on first use)."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to .env or export it as an "
                "environment variable."
            )
        _client = Groq(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Core API wrapper
# ---------------------------------------------------------------------------

def call_llm(prompt: str, system: str = "") -> str:
    """
    Call the Groq LLM with *prompt* and optional *system* message.

    Retries up to ``MAX_RETRIES`` times with exponential backoff on
    rate-limit or transient errors.

    Parameters
    ----------
    prompt : str
        The user prompt.
    system : str
        Optional system message to set behaviour.

    Returns
    -------
    str
        The raw text response from the model.
    """
    client = _get_client()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                print(f"  [llm] retry {attempt}/{MAX_RETRIES} after error: {exc}")
                time.sleep(wait)

    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} retries: {last_exc}")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM = (
    "You are a personal-knowledge-base classifier. "
    "Given a piece of content, classify it into exactly one PARA category "
    "(Projects, Areas, Resources, Archives), assign 1-5 lowercase tags, "
    "and write a one-line summary (max 120 chars). "
    "Respond ONLY with a JSON object, no markdown fences."
)

CLASSIFY_PROMPT_TEMPLATE = """Classify this content.

Content:
\"\"\"
{content}
\"\"\"

Respond with JSON in this exact format:
{{"para": "Projects|Areas|Resources|Archives", "tags": ["tag1", "tag2"], "summary": "One-line summary"}}"""


def classify_content(text: str) -> Dict[str, Any]:
    """
    Classify *text* into a PARA category with tags and a summary.

    Parameters
    ----------
    text : str
        The content to classify (truncated to ~4000 chars to stay within
        context limits).

    Returns
    -------
    dict
        ``{"para": str, "tags": list[str], "summary": str}``
    """
    # Truncate very long content
    truncated = text[:4000] if len(text) > 4000 else text

    prompt = CLASSIFY_PROMPT_TEMPLATE.format(content=truncated)
    raw = call_llm(prompt, system=CLASSIFY_SYSTEM)

    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove first line (```json or ```) and last ```
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON from the text
        import re
        match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
            except json.JSONDecodeError:
                result = {
                    "para": "Resources",
                    "tags": ["unclassified"],
                    "summary": "Classification failed — see raw content",
                }
        else:
            result = {
                "para": "Resources",
                "tags": ["unclassified"],
                "summary": "Classification failed — see raw content",
            }

    # Validate para
    valid_para = {"Projects", "Areas", "Resources", "Archives"}
    if result.get("para") not in valid_para:
        result["para"] = "Resources"

    # Ensure tags is a list
    if not isinstance(result.get("tags"), list):
        result["tags"] = []

    # Ensure summary is a string
    if not isinstance(result.get("summary"), str):
        result["summary"] = ""

    return result


# ---------------------------------------------------------------------------
# RAG answer synthesis (used in Phase 4)
# ---------------------------------------------------------------------------

SYNTHESIZE_SYSTEM = (
    "You are SecondSelf, answering from the user's personal knowledge base. "
    "Use ONLY the provided notes. If the answer isn't in the notes, say so. "
    "Cite sources as [note-id]."
)

SYNTHESIZE_PROMPT_TEMPLATE = """Notes:
{context}

Question: {question}

Answer (cite sources as [note-id]):"""


def synthesize_answer(context: str, question: str) -> str:
    """
    Synthesize an answer from *context* notes for *question*.

    Parameters
    ----------
    context : str
        Concatenated note bodies with IDs.
    question : str
        The user's question.

    Returns
    -------
    str
        The synthesized answer.
    """
    prompt = SYNTHESIZE_PROMPT_TEMPLATE.format(
        context=context[:6000],  # truncate to ~6000 tokens worth
        question=question,
    )
    return call_llm(prompt, system=SYNTHESIZE_SYSTEM)