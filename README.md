# Finn.py - AEC Signal Intelligence Dashboard

Finn.py is a research assistant for Architecture, Engineering, and Construction (AEC) teams. It ingests curated RSS feeds and uploaded documents, asks an LLM to summarize the findings, stores the resulting "signals" in Qdrant, and exposes a Tailwind-powered Flask dashboard that lets analysts explore insights, run refresh jobs, and chat with an embedded assistant.
 
## Highlights
- **Dashboard & workflows:** `/` shows ranked signals, `/assistant` exposes the LLM, `/add_source` validates RSS feeds, `/refresh` controls ingestion schedules, and `/visuals` renders simple analytics.
- **Automated ingestion:** `ingest.py` fetches articles from `sources.txt`, enriches them with the StackIT-hosted Llama 3.3 model, and uploads vectors plus metadata to Qdrant.
- **Document and RSS uploads:** Users can upload PDF, DOCX, TXT, or Markdown files whose contents are auto-summarized and indexed next to RSS entries.
- **Searchable vector store:** Qdrant stores dense embeddings (StackIT or sentence-transformers), enabling semantic search and deduplication.
- **Configurable refresher:** A JSON schedule in `data/refresh_schedule.json` keeps track of manual or scheduled refreshes.
  
## Repository Layout 
- `app.py` - Flask application serving the dashboard, assistant, upload, refresh, and API endpoints.  
- `ingest.py` - CLI plus library that ingests RSS feeds or uploaded documents and syncs them to Qdrant.
- `client.py` - Minimal script to confirm StackIT LLM access. 
- `templates/` - Jinja templates for dashboard, assistant, add-source, refresh, and visuals pages.
- `static/` - Tailwind-ready assets, icons, and helper JS.
- `sources.txt` - Pipe-delimited list of RSS feeds (`Name|URL|Category`).
- `data/` - Generated artifacts (`*.jsonl` batches, `refresh_schedule.json`, etc.).
- `requirements.txt` - Python dependencies; install inside the project-specific virtual environment.

## Prerequisites 
- Python 3.11 (any recent 3.10+ build with venv support works).
- Access tokens for StackIT Model Serving (chat plus embeddings) and a Qdrant instance.
- Optional: a GPU-backed environment if you plan to swap in local embedding models.
 
## Required Environment Variables
Create a `.env` file (already git-ignored). Copy the keys below with your own values:

| Variable | Purpose |
| --- | --- |
| `STACKIT_MODEL_SERVING_MODEL` | LLM name used for summarization and the assistant (for example `cortecs/Llama-3.3-70B-Instruct-FP8-Dynamic`). |
| `STACKIT_MODEL_SERVING_BASE_URL` | Base URL for the StackIT OpenAI-compatible endpoint. |
| `STACKIT_MODEL_SERVING_AUTH_TOKEN` | API token for StackIT. |
| `STACKIT_EMBED_MODEL` | Embedding model to request from StackIT (falls back to `LOCAL_EMBED_MODEL`). |
| `LOCAL_EMBED_MODEL` | Hugging Face or sentence-transformers model to use when StackIT embeddings are unavailable. |
| `QDRANT_URL` | HTTPS endpoint of your Qdrant cluster. |
| `QDRANT_API_KEY` | Qdrant API key with write permissions. |
| `QDRANT_COLLECTION` | Collection name for signals (defaults to `aec_signals`). |
| `DASHBOARD_LIMIT` | Default number of records shown on the dashboard. |
| `INGEST_SOURCES_FILE` | Path to the RSS definition list (`sources.txt`). |
| `INGEST_OUTPUT_DIR` | Directory where JSONL batches and schedule files are written (`data`). |
| `INGEST_MAX_ITEMS` | Default RSS article limit per run (can be overridden with CLI flags). |

Tip: never commit the `.env` file. Keep production and local credentials separate.

`client.py` simply reuses the StackIT variables above to run a "ping" request against the LLM endpoint.

## Quickstart

### 1. Clone and enter the repo
```powershell
git clone <repo-url> hackathon
cd hackathon
```

### 2. Create and activate a virtual environment
On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\activate
```

On macOS or Linux:
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Provide environment variables
Copy `.env.example` from your secrets store (or edit `.env`) with the keys outlined above.

### 5. Run the Flask dashboard
```bash
python app.py
```
Visit `http://127.0.0.1:8000` (or `http://localhost:8000`). The app runs with `debug=True` by default; override via `FLASK_ENV=production` or run behind a WSGI server in production.

## Managing RSS Sources
`sources.txt` contains one feed per line:
```
Source Name | https://example.com/rss | Optional Category
```
- Use the `/add_source` UI to validate and append feeds interactively.
- Ensure categories match the values expected by `ingest.py` (see `CATEGORY_CHOICES` for canonical names).

## Running the Ingestion Pipeline
With the virtual environment active:
```bash
python ingest.py --max-items 25        # fetch up to 25 new items per feed
python ingest.py --dry-run             # write JSONL but skip Qdrant upserts
```
The script:
1. Loads feeds from `sources.txt`.
2. Fetches and parses RSS entries (Feedparser, Trafilatura).
3. Calls the StackIT Llama model to summarize, score impact, and extract metadata.
4. Generates embeddings (StackIT or local sentence-transformers).
5. Writes a JSONL batch in `data/` and optionally upserts to Qdrant.

`data/refresh_schedule.json` tracks manual or automatic refreshes that the `/refresh` page exposes. When you trigger **Run refresh now**, the server calls `run_ingest()` with the provided limits and updates this schedule file.

## Uploading Individual Documents
- `POST /api/upload_ingest` is wired to the "Upload and ingest" form inside the dashboard.
- Supported extensions: `.pdf`, `.txt`, `.md`, `.docx`.
- Each upload is chunked, summarized, and pushed through the same ingestion flow as RSS items.

## Assistant Endpoint
The `/assistant` page streams responses from the configured StackIT chat model via LangChain. Use the `client.py` script to verify credentials:
```bash
python client.py
```
You should receive a short greeting from the model.

## Troubleshooting
- **Missing modules:** Ensure the virtual environment is active (prompt prefixed by `(venv)`), then rerun `pip install -r requirements.txt`.
- **Qdrant errors:** Double-check `QDRANT_URL`, `QDRANT_API_KEY`, and that the collection exists. The app creates clients lazily and will raise `RuntimeError` if credentials are absent.
- **StackIT 401 or 403:** Tokens expire; generate a fresh `STACKIT_MODEL_SERVING_AUTH_TOKEN` and restart the app.
- **Windows path issues:** Keep paths in `.env` wrapped in quotes if they include spaces (for example `INGEST_OUTPUT_DIR="C:/data/aec"`).

## Next Steps
- Containerize the service (Gunicorn plus Nginx, background ingest via APScheduler or Celery).
- Hook `/api/refresh/schedule` to a cron or worker so refreshes run automatically.
- Add telemetry (OpenTelemetry, Sentry) before production deployment.

Happy building!
