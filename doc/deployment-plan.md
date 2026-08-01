# SecondSelf — Streamlit Deployment Plan

## 1. Overview

SecondSelf is a local-first knowledge system with a Streamlit front-end (`app.py`). The primary deployment target is **Streamlit Community Cloud** — a free, git-push-based platform that gives you a public `*.streamlit.app` URL with zero infrastructure to manage.

An alternative path via **Hugging Face Spaces** is included in Section 8.

---

## 2. Prerequisites

| Requirement | Detail |
|-------------|--------|
| Python version | 3.11 or higher |
| GitHub account | Public or private repo (Community Cloud supports both) |
| Groq API key | Free at [console.groq.com](https://console.groq.com) |
| `sentence-transformers` model | `all-MiniLM-L6-v2` (~80 MB) — downloaded on first run |

---

## 3. Pre-Deployment Checklist

Work through these before pushing to the deployment branch.

### 3.1 Build data artifacts locally

The app reads pre-built files from `data/` and `wiki/`. These must be committed (or generated on startup) because Streamlit Community Cloud has no persistent disk between deploys.

```bash
# 1. Classify all raw captures → wiki/
python pipeline.py classify

# 2. Compute embeddings + semantic links
python pipeline.py link

# 3. Build graph.json
python pipeline.py graph
```

Confirm these files exist and are non-empty:

```
data/
  embeddings.pkl    ← must be present for the Ask feature
  graph.json        ← must be present for the graph view
  index.json        ← tracks processing state
wiki/
  Projects/...
  Areas/...
  Resources/...
  Archives/...
```

### 3.2 Verify the app runs locally

```bash
streamlit run app.py
```

Open `http://localhost:8501` and check:
- Knowledge graph renders (vis-network iframe)
- Ask bar returns answers with source chips
- Sidebar "Run Full Pipeline" button completes without error

### 3.3 Pin dependency versions

Streamlit Cloud rebuilds the environment from `requirements.txt` on every deploy. Use pinned versions to avoid unexpected breakage:

```
streamlit==1.35.0
groq==0.9.0
sentence-transformers==3.0.1
numpy==1.26.4
pyyaml==6.0.1
pypdf==4.2.0
requests==2.32.3
beautifulsoup4==4.12.3
python-dotenv==1.0.1
```

Check the latest stable versions of each package before locking.

### 3.4 Create `.streamlit/config.toml`

Create the file at `.streamlit/config.toml` in the repo root:

```toml
[server]
headless = true
port = 8501

[browser]
gatherUsageStats = false

[theme]
base = "dark"
primaryColor = "#7c3aed"
backgroundColor = "#0f1117"
secondaryBackgroundColor = "#1a1d27"
textColor = "#e2e8f0"
font = "sans serif"
```

This ensures the dark theme matches the vis-network graph styling already baked into `static/graph.html`.

---

## 4. Repository Hygiene

### 4.1 What to commit

| Path | Commit? | Reason |
|------|---------|--------|
| `app.py`, `*.py` source files | Yes | Application code |
| `lib/`, `static/` | Yes | Shared modules and graph HTML |
| `requirements.txt` | Yes | Dependency manifest |
| `.streamlit/config.toml` | Yes | UI configuration |
| `wiki/` | Yes | Pre-processed notes (demo data) |
| `data/graph.json` | Yes | Pre-built graph for first render |
| `data/index.json` | Yes | Pipeline state |
| `data/embeddings.pkl` | **Yes** | Required for Ask to work at startup |
| `raw/` | Optional | Raw captures — omit to keep repo lean |
| `.env` | **No** | Never commit secrets |
| `.venv/` | No | Already in `.gitignore` |

> **Privacy note:** `wiki/` notes and `data/embeddings.pkl` will be public if your GitHub repo is public. Either make the repo private, strip personal content before deploying, or add authentication (see Section 7).

### 4.2 `.gitignore` additions

Append these lines to `.gitignore` if not already present:

```
.env
.venv/
__pycache__/
*.pyc
.DS_Store
```

The existing `.gitignore` already covers these, so no changes should be needed.

---

## 5. Deploying to Streamlit Community Cloud

### Step 1 — Push to GitHub

```bash
git add .
git commit -m "chore: prepare for Streamlit Cloud deploy"
git push origin main
```

### Step 2 — Connect to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**.
3. Select your repository, branch (`main`), and set the main file path to `app.py`.
4. Click **Advanced settings** before deploying (see Step 3).

### Step 3 — Add the API key secret

In **Advanced settings → Secrets**, paste:

```toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"
```

Streamlit injects this as an environment variable at runtime. The `load_dotenv()` call in `lib/llm.py` falls back gracefully when `.env` is absent because `os.environ.get("GROQ_API_KEY")` will still find the value injected by Streamlit Secrets.

### Step 4 — Deploy

Click **Deploy**. Community Cloud will:
1. Clone the repo into a sandbox.
2. Install packages from `requirements.txt` (including downloading the `all-MiniLM-L6-v2` model on first boot — allow ~2 min).
3. Run `streamlit run app.py`.

Your app will be live at:
```
https://<your-github-username>-secondself-app-<hash>.streamlit.app
```

---

## 6. Cold-Start Behaviour

The `sentence-transformers` model is loaded lazily via `@st.cache_resource` in `lib/embeddings.py`. On the very first request after a deploy:

- Model download: ~80 MB, ~30–60 seconds on Community Cloud.
- Subsequent requests: model stays in memory for the lifetime of the server.

If cold starts are unacceptable, pre-warm the model by calling `load_model()` inside an `@st.cache_resource` function that runs at import time in `app.py`.

---

## 7. Handling Private Notes

By default, any notes committed to a public repo are publicly readable. Choose one of:

| Option | Effort | Notes |
|--------|--------|-------|
| Make the GitHub repo **private** | Low | Community Cloud supports private repos on the free tier |
| Use `streamlit-authenticator` | Medium | Password-gates the UI; notes stay in repo |
| Move to Hugging Face Spaces (private) | Low | Mark the Space as private |
| Strip personal content; use demo data only | Low | Best for public showcases |

---

## 8. Alternative: Hugging Face Spaces

If you prefer Hugging Face Spaces:

1. Create a new Space → SDK: **Streamlit**.
2. Push the same repo content.
3. Add `GROQ_API_KEY` under **Settings → Repository secrets**.
4. Hugging Face will auto-detect `requirements.txt` and run `streamlit run app.py`.

No extra configuration files are needed. The free tier provides 2 vCPUs and 16 GB RAM — enough for the `all-MiniLM-L6-v2` model.

---

## 9. Keeping Data Fresh

Because Streamlit Community Cloud has no persistent disk, any captures made through the UI's sidebar **will not survive a redeploy**. There are two practical strategies:

**Option A — Pre-build and redeploy (recommended for v1)**  
Run the pipeline locally, commit the updated `wiki/`, `data/`, and push. The next deploy reflects the new notes.

```bash
python pipeline.py process
git add wiki/ data/
git commit -m "data: add new captures"
git push
```

Streamlit Cloud detects the push and auto-redeploys in ~1 minute.

**Option B — External storage (future)**  
Swap `lib/storage.py` to read/write from a hosted store (e.g. GitHub API, S3, or a small SQLite on Turso). This is out of scope for v1 but the storage abstraction in `lib/storage.py` makes it a single-file change.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `GROQ_API_KEY not set` error | Secret not added | Re-check Streamlit Secrets (Step 3) |
| Graph shows "No graph data yet" | `data/graph.json` not committed or empty | Run `python pipeline.py graph` locally and commit |
| Ask returns no results | `data/embeddings.pkl` missing | Run `python pipeline.py link` locally and commit |
| App crashes on startup with `ModuleNotFoundError` | Dependency missing from `requirements.txt` | Add the missing package and redeploy |
| Cold start takes > 3 minutes | Model download slow on first boot | Normal; subsequent loads use cache |
| vis-network graph is blank | `graph.json` has 0 nodes | Process at least one capture through the pipeline |
| `sentence-transformers` install fails | C++ build tools missing in sandbox | Add `torch==2.3.0` (CPU-only) to `requirements.txt` to force a pre-built wheel |

---

## 11. File Checklist Before First Deploy

```
d:\New folder (2)\
├── app.py                    ✅ entry point
├── requirements.txt          ✅ pinned versions
├── .streamlit/
│   └── config.toml           ✅ create this
├── lib/                      ✅ commit as-is
├── static/graph.html         ✅ commit as-is
├── wiki/                     ✅ pre-built notes
├── data/
│   ├── graph.json            ✅ pre-built
│   ├── embeddings.pkl        ✅ pre-built
│   └── index.json            ✅ pre-built
├── .env                      ❌ never commit — add via Secrets UI
└── raw/                      optional (can omit to keep repo lean)
```
