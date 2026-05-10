from __future__ import annotations

import os
import json
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import feedparser
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from zoneinfo import ZoneInfo
from docx import Document
from pypdf import PdfReader
from ingest import ingest_documents, ingest_single_source, run_ingest
from werkzeug.utils import secure_filename

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "aec_signals")
DEFAULT_SIGNAL_LIMIT = int(os.getenv("DASHBOARD_LIMIT", "24"))
STACKIT_MODEL = os.getenv("STACKIT_MODEL_SERVING_MODEL")
STACKIT_BASE_URL = os.getenv("STACKIT_MODEL_SERVING_BASE_URL")
STACKIT_TOKEN = os.getenv("STACKIT_MODEL_SERVING_AUTH_TOKEN")
SOURCES_FILE = Path(os.getenv("INGEST_SOURCES_FILE", "sources.txt"))
SCHEDULE_FILE = Path("data/refresh_schedule.json")
BERLIN_TZ = ZoneInfo("Europe/Berlin")
UPLOAD_ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".docx"}
LEVEL_GROUPS = {
    "high": {"label": "High", "levels": [5, 4]},
    "medium": {"label": "Medium", "levels": [3]},
    "low": {"label": "Low", "levels": [2, 1]},
}

LEVEL_LABELS: Dict[int, str] = {}
LEVEL_KEY_BY_VALUE: Dict[int, str] = {}
for key, meta in LEVEL_GROUPS.items():
    for value in meta["levels"]:
        LEVEL_LABELS[value] = meta["label"]
        LEVEL_KEY_BY_VALUE[value] = key

UPLOAD_ARCHIVE_DIR = Path(os.getenv("UPLOAD_ARCHIVE_DIR", "data/uploads"))
UPLOAD_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder="templates")


@app.context_processor
def inject_globals():
    return {"level_labels": LEVEL_LABELS, "level_groups": LEVEL_GROUPS}


def _get_qdrant_client() -> QdrantClient:
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError("Qdrant credentials are missing in .env")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def _parse_published(value: Optional[str]) -> Tuple[str, datetime]:
    if not value:
        normalized = datetime.min.replace(tzinfo=timezone.utc)
        return ("-", normalized)
    dt: Optional[datetime] = None
    try:
        dt = parsedate_to_datetime(value)
    except Exception:
        try:
            dt = datetime.fromisoformat(value)
        except Exception:
            normalized = datetime.min.replace(tzinfo=timezone.utc)
            return (value, normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return (dt.strftime("%d %b %Y %H:%M"), dt)


def _parse_filters(args) -> Tuple[Optional[List[int]], Optional[str], Optional[str], Optional[int]]:
    level_filter: Optional[List[int]] = None
    selected_impact: Optional[str] = None
    primary_level: Optional[int] = None

    impact_param = args.get("impact")
    if impact_param:
        impact_key = impact_param.lower()
        meta = LEVEL_GROUPS.get(impact_key)
        if meta:
            selected_impact = impact_key
            level_filter = list(meta["levels"])
            primary_level = level_filter[0] if level_filter else None

    level_param = args.get("level")
    if level_param and not level_filter:
        try:
            level_candidate = int(level_param)
            if 1 <= level_candidate <= 5:
                level_filter = [level_candidate]
                primary_level = level_candidate
                selected_impact = LEVEL_KEY_BY_VALUE.get(level_candidate)
        except ValueError:
            pass

    category = args.get("category")
    category_filter = category.strip() if category else None
    if category_filter == "":
        category_filter = None
    return level_filter, category_filter, selected_impact, primary_level


def _archive_uploaded_file(original_name: str, data: bytes) -> Path:
    safe_name = secure_filename(original_name or "document")
    suffix = Path(safe_name).suffix or ""
    unique_id = uuid4().hex[:8]
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    stored_name = f"{timestamp}-{unique_id}{suffix}"
    destination = UPLOAD_ARCHIVE_DIR / stored_name
    destination.write_bytes(data)
    return destination


def _score_from_label(label) -> int:
    if not label:
        return 2
    if isinstance(label, (int, float)):
        try:
            return int(label)
        except (TypeError, ValueError):
            return 2
    normalized = str(label).strip().lower()
    for key, meta in LEVEL_GROUPS.items():
        if normalized == key or normalized == meta["label"].lower():
            return max(meta["levels"])
    if "high" in normalized or "critical" in normalized:
        return 5
    if "med" in normalized:
        return 3
    return 2


def fetch_signals(
    limit: int = DEFAULT_SIGNAL_LIMIT,
    level_filter: Optional[List[int]] = None,
    category_filter: Optional[str] = None,
    collect_categories: bool = False,
    search_query: Optional[str] = None,
    search_field: str = "title",
    priority_impact: Optional[str] = None,
) -> Any:
    client = _get_qdrant_client()
    points: List[Any] = []
    offset = None
    # grab slightly more in case we drop empty payloads
    fetch_target = max(limit * 2, limit + 5)
    while len(points) < fetch_target:
        batch, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=fetch_target - len(points),
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        points.extend(batch)
        if offset is None:
            break

    records: List[Dict[str, Any]] = []
    categories_seen: set[str] = set()
    for point in points:
        payload = point.payload or {}
        published_display, published_dt = _parse_published(payload.get("published"))
        category_value = payload.get("category", "general")
        if isinstance(category_value, str):
            categories_seen.add(category_value)
        raw_numeric = payload.get("signal_level_numeric")
        try:
            level_value = int(raw_numeric)
        except (TypeError, ValueError):
            level_value = _score_from_label(payload.get("signal_level"))
        level_label_raw = payload.get("signal_level")
        if isinstance(level_label_raw, (int, float)) or (isinstance(level_label_raw, str) and level_label_raw.isdigit()):
            level_label = LEVEL_LABELS.get(int(level_label_raw), LEVEL_LABELS.get(level_value, "Medium"))
        else:
            level_label = level_label_raw or LEVEL_LABELS.get(level_value, "Medium")
        impact_key = payload.get("signal_impact") or LEVEL_KEY_BY_VALUE.get(level_value)
        if not impact_key and level_label:
            impact_key = level_label.lower()
        records.append(
            {
                "source": payload.get("source", "Unknown source"),
                "source_category": payload.get("source_category", "Uncategorized"),
                "link": payload.get("link"),
                "title": payload.get("title", "Untitled insight"),
                "summary": payload.get("summary", ""),
                "category": category_value,
                "keywords": payload.get("keywords") or [],
                "entities": payload.get("entities") or [],
                "signal_level_score": level_value,
                "signal_level_label": level_label,
                "signal_impact": impact_key,
                "published": published_display,
                "published_dt": published_dt,
                "collected_at": payload.get("collected_at"),
            }
        )

    records.sort(key=lambda r: (r["signal_level_score"], r["published_dt"]), reverse=True)
    if priority_impact and priority_impact in LEVEL_GROUPS:
        records.sort(
            key=lambda r: 0 if (r.get("signal_impact") == priority_impact) else 1
        )

    if level_filter:
        allowed = set(level_filter)
        records = [r for r in records if r["signal_level_score"] in allowed]
    if category_filter:
        cat_lower = category_filter.lower()
        records = [
            r for r in records if (r["category"] or "").lower() == cat_lower
        ]
    if search_query:
        terms = [term.lower() for term in search_query.split() if term.strip()]
        if terms:
            field = (
                "summary"
                if search_field == "summary"
                else "source"
                if search_field == "source"
                else "title"
            )
            records = [
                r
                for r in records
                if all(term in (r.get(field) or "").lower() for term in terms)
            ]

    sliced = records[:limit]
    if collect_categories:
        options = sorted({c for c in categories_seen if c})
        return sliced, options
    return sliced


def _get_llm() -> ChatOpenAI:
    if not STACKIT_MODEL or not STACKIT_TOKEN or not STACKIT_BASE_URL:
        raise RuntimeError("STACKIT model settings are missing in .env")
    return ChatOpenAI(
        model=STACKIT_MODEL,
        base_url=STACKIT_BASE_URL,
        api_key=STACKIT_TOKEN,
        temperature=0.2,
        max_retries=2,
    )


def _read_sources() -> List[Tuple[str, str, str]]:
    if not SOURCES_FILE.exists():
        return []
    rows: List[Tuple[str, str, str]] = []
    with SOURCES_FILE.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                name = parts[0]
                url = parts[1]
                category = parts[2] if len(parts) > 2 else "General"
                rows.append((name, url, category))
    return rows


def _append_source(name: str, url: str, category: str) -> None:
    SOURCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SOURCES_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"{name} | {url} | {category}\n")


def _validate_rss_url(url: str) -> str:
    feed = feedparser.parse(url)
    if getattr(feed, "bozo", False):
        raise ValueError("Feed appears invalid or unreachable.")
    entries = getattr(feed, "entries", None)
    if not entries:
        raise ValueError("Feed returned no entries.")
    feed_meta = getattr(feed, "feed", {}) or {}
    title = feed_meta.get("title") if isinstance(feed_meta, dict) else getattr(feed_meta, "title", None)
    return title or url


def _load_schedule() -> Dict[str, Any]:
    if SCHEDULE_FILE.exists():
        try:
            return json.loads(SCHEDULE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"period_days": 7, "last_run": None}


def _save_schedule(data: Dict[str, Any]) -> None:
    SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCHEDULE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _parse_schedule_dt(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _format_schedule_dt(raw: Optional[str]) -> Optional[str]:
    dt = _parse_schedule_dt(raw)
    if not dt:
        return None
    return dt.astimezone(BERLIN_TZ).strftime("%d %b %Y %H:%M %Z")


def _format_schedule_next(schedule: Dict[str, Any]) -> Optional[str]:
    last_run_dt = _parse_schedule_dt(schedule.get("last_run"))
    if not last_run_dt:
        return None
    try:
        period_days = int(schedule.get("period_days", 7))
        if period_days <= 0:
            return None
    except (TypeError, ValueError):
        return None
    next_run_dt = last_run_dt + timedelta(days=period_days)
    return next_run_dt.astimezone(BERLIN_TZ).strftime("%d %b %Y %H:%M %Z")


def _extract_text_from_upload(filename: str, data: bytes) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in UPLOAD_ALLOWED_SUFFIXES:
        raise ValueError("Unsupported file type. Please upload PDF, DOCX, or text files.")
    if suffix in {".txt", ".md"}:
        return data.decode("utf-8", errors="ignore").strip()
    stream = BytesIO(data)
    if suffix == ".pdf":
        reader = PdfReader(stream)
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        return text.strip()
    if suffix == ".docx":
        document = Document(stream)
        return "\n".join(p.text for p in document.paragraphs if p.text).strip()
    raise ValueError("Unsupported file type.")


@app.route("/")
def dashboard():
    level_filter, category_filter, impact_key, primary_level = _parse_filters(request.args)
    search_query = (request.args.get("search") or "").strip()
    search_type = request.args.get("search_type", "title")
    if search_type not in {"title", "summary", "source"}:
        search_type = "title"
    sort_impact = request.args.get("sort_impact", "").lower() or None
    if sort_impact not in LEVEL_GROUPS:
        sort_impact = None
    data = fetch_signals(
        level_filter=level_filter,
        category_filter=category_filter,
        collect_categories=True,
        search_query=search_query,
        search_field=search_type,
        priority_impact=sort_impact,
    )
    signals, category_options = data
    high_count = sum(1 for s in signals if s["signal_level_label"] == "High")
    medium_count = sum(1 for s in signals if s["signal_level_label"] == "Medium")
    low_count = sum(1 for s in signals if s["signal_level_label"] == "Low")
    unique_sources = len({s["source"] for s in signals if s.get("source")})
    latest_dt = max((s["published_dt"] for s in signals if s["published_dt"] != datetime.min), default=None)
    stats = {
        "total": len(signals),
        "high": high_count,
        "medium": medium_count,
        "low": low_count,
        "sources": unique_sources,
        "latest": latest_dt.strftime("%d %b %Y %H:%M") if latest_dt else "-",
    }
    return render_template(
        "index.html",
        signals=signals,
        collection=QDRANT_COLLECTION,
        last_refreshed=datetime.utcnow().strftime("%d %b %Y %H:%M UTC"),
        selected_level=primary_level,
        selected_category=category_filter,
        category_options=category_options,
        selected_impact=impact_key,
        search_query=search_query,
        search_type=search_type,
        selected_sort_impact=sort_impact,
        stats=stats,
    )


@app.route("/api/signals")
def api_signals():
    level_filter, category_filter, impact_key, primary_level = _parse_filters(request.args)
    search_query = (request.args.get("search") or "").strip()
    search_type = request.args.get("search_type", "title")
    if search_type not in {"title", "summary", "source"}:
        search_type = "title"
    sort_impact = request.args.get("sort_impact", "").lower() or None
    if sort_impact not in LEVEL_GROUPS:
        sort_impact = None
    data = fetch_signals(
        level_filter=level_filter,
        category_filter=category_filter,
        search_query=search_query,
        search_field=search_type,
        priority_impact=sort_impact,
    )
    for record in data:
        record.pop("published_dt", None)
    return jsonify(
        {
            "collection": QDRANT_COLLECTION,
            "count": len(data),
            "filters": {
                "impact": impact_key,
                "level": primary_level,
                "category": category_filter,
            },
            "signals": data,
        }
    )


@app.route("/assistant", methods=["GET", "POST"])
def assistant():
    user_question = ""
    answer = None
    error = None
    if request.method == "POST":
        user_question = (request.form.get("question") or "").strip()
        if not user_question:
            error = "Please enter a question before submitting."
        else:
            try:
                llm = _get_llm()
                context_records = fetch_signals(limit=6)
                context_text = "\n".join(
                    f"- {item['title']} ({item['category']}, signal {item['signal_level_label']}): {item['summary']}"
                    for item in context_records
                )
                prompt = (
                    "You are Finn, an AI analyst for the Finn.py market intelligence team. "
                    "Use the provided context to answer user questions. "
                    "When referencing data, cite the source title and keep responses under 180 words.\n\n"
                    f"Context:\n{context_text or 'No recent signals available.'}\n\n"
                    f"Question: {user_question}"
                )
                response = llm.invoke(
                    [
                        {"role": "system", "content": "Provide concise, insight-focused answers."},
                        {"role": "user", "content": prompt},
                    ]
                )
                answer = response.content.strip()
            except Exception as exc:  # noqa: BLE001
                error = f"Assistant request failed: {exc}"

    return render_template(
        "assistant.html",
        question=user_question,
        answer=answer,
        error=error,
        collection=QDRANT_COLLECTION,
        last_refreshed=datetime.utcnow().strftime("%d %b %Y %H:%M UTC"),
    )


@app.route("/sources/new", methods=["GET", "POST"])
def add_source():
    existing_sources = _read_sources()

    def _category_options():
        return sorted({row[2] for row in existing_sources if row[2]}) or ["General"]

    known_categories = _category_options()
    form_data = {
        "name": "",
        "rss_url": "",
        "category": known_categories[0],
    }
    message = None
    error = None

    if request.method == "POST":
        form_data["name"] = (request.form.get("name") or "").strip()
        form_data["rss_url"] = (request.form.get("rss_url") or "").strip()
        form_data["category"] = (request.form.get("category") or "").strip() or known_categories[0]

        if not form_data["rss_url"]:
            error = "RSS URL is required."
        else:
            try:
                feed_title = _validate_rss_url(form_data["rss_url"])
                display_name = form_data["name"] or feed_title
                existing_urls = {row[1] for row in existing_sources}
                if form_data["rss_url"] in existing_urls:
                    raise ValueError("This RSS URL already exists in sources.txt.")
                _append_source(display_name, form_data["rss_url"], form_data["category"])
                message = f"Added '{display_name}' to sources.txt"
                form_data = {"name": "", "rss_url": "", "category": form_data["category"]}
                existing_sources = _read_sources()
                known_categories = _category_options()
            except Exception as exc:  # noqa: BLE001
                error = str(exc)

    return render_template(
        "add_source.html",
        collection=QDRANT_COLLECTION,
        last_refreshed=datetime.utcnow().strftime("%d %b %Y %H:%M UTC"),
        form=form_data,
        categories=known_categories,
        message=message,
        error=error,
        sources_count=len(existing_sources),
    )


@app.route("/visuals")
def visuals_page():
    signals, _ = fetch_signals(limit=300, collect_categories=True)
    category_counts: Dict[str, int] = {}
    for signal in signals:
        category = signal.get("category") or signal.get("source_category") or "Uncategorized"
        category_counts[category] = category_counts.get(category, 0) + 1
    if not category_counts:
        category_counts = {"No data": 1}
    category_pairs = sorted(category_counts.items(), key=lambda item: item[1], reverse=True)
    impact_order = ["High", "Medium", "Low"]
    impact_counts = {label: 0 for label in impact_order}
    for signal in signals:
        label = (signal.get("signal_level_label") or "Medium").title()
        if label not in impact_counts:
            impact_counts[label] = 0
        impact_counts[label] += 1
    top_categories = category_pairs[:5]
    return render_template(
        "visuals.html",
        collection=QDRANT_COLLECTION,
        last_refreshed=datetime.utcnow().strftime("%d %b %Y %H:%M UTC"),
        total_signals=len(signals),
        total_categories=len(category_counts),
        top_categories=top_categories,
        category_labels=[label for label, _ in category_pairs],
        category_values=[count for _, count in category_pairs],
        impact_labels=impact_order,
        impact_values=[impact_counts.get(label, 0) for label in impact_order],
    )


@app.route("/api/source/ingest", methods=["POST"])
def api_source_ingest():
    payload = request.get_json(silent=True) or {}
    rss_url = (payload.get("rss_url") or "").strip()
    if not rss_url:
        return jsonify({"status": "error", "message": "RSS URL is required."}), 400
    name = (payload.get("name") or "").strip()
    category = (payload.get("category") or "General").strip() or "General"
    max_items = payload.get("max_items")
    try:
        max_items_int = int(max_items) if max_items not in (None, "") else None
        if max_items_int is not None and max_items_int <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Max items must be a positive integer."}), 400
    try:
        validated_name = _validate_rss_url(rss_url)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "message": str(exc)}), 400
    display_name = name or validated_name
    try:
        result = ingest_single_source(
            name=display_name,
            url=rss_url,
            category=category,
            max_items=max_items_int,
            skip_existing=True,
            dry_run=False,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "message": str(exc)}), 500
    return jsonify({"status": "success", "result": result})


@app.route("/uploads/<path:filename>")
def serve_uploaded_file(filename: str):
    return send_from_directory(UPLOAD_ARCHIVE_DIR, filename, as_attachment=False)


@app.route("/api/upload_ingest", methods=["POST"])
def api_upload_ingest():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "Please attach a file to ingest."}), 400
    file = request.files["file"]
    filename = file.filename or "document"
    data = file.read()
    if not data:
        return jsonify({"status": "error", "message": "Uploaded file is empty."}), 400
    try:
        text = _extract_text_from_upload(filename, data)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "message": str(exc)}), 400
    if not text.strip():
        return jsonify({"status": "error", "message": "Could not extract text from the uploaded file."}), 400
    try:
        archived_path = _archive_uploaded_file(filename, data)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "message": f"Could not store uploaded file: {exc}"}), 500
    file_url = url_for("serve_uploaded_file", filename=archived_path.name)
    doc_title = (request.form.get("title") or Path(filename).stem or "Uploaded document").strip()
    category = (request.form.get("category") or "Imported Documents").strip() or "Imported Documents"
    source_name = (request.form.get("source") or "Manual upload").strip() or "Manual upload"
    summary = (request.form.get("summary") or "").strip()
    try:
        result = ingest_documents(
            [
                {
                    "title": doc_title,
                    "summary": summary,
                    "content": text,
                    "category": category,
                    "source": source_name,
                    "link": file_url,
                    "local_file": archived_path.name,
                    "original_filename": filename,
                }
            ],
            skip_existing=False,
            dry_run=False,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "message": str(exc)}), 500
    result["stored_file"] = archived_path.name
    result["file_url"] = file_url
    return jsonify({"status": "success", "result": result})


@app.route("/refresh")
def refresh_page():
    schedule = _load_schedule()
    return render_template(
        "refresh.html",
        collection=QDRANT_COLLECTION,
        last_refreshed=datetime.utcnow().strftime("%d %b %Y %H:%M UTC"),
        schedule=schedule,
        schedule_last_run=_format_schedule_dt(schedule.get("last_run")),
        schedule_next_run=_format_schedule_next(schedule),
    )


@app.route("/api/refresh/run", methods=["POST"])
def api_refresh_run():
    payload = request.get_json(silent=True) or {}
    max_items = payload.get("max_items")
    try:
        max_items_int = int(max_items) if max_items not in (None, "") else None
    except (TypeError, ValueError):
        max_items_int = None
    try:
        result = run_ingest(max_items=max_items_int, skip_existing=True, dry_run=False)
        schedule = _load_schedule()
        schedule["last_run"] = datetime.now(timezone.utc).isoformat()
        _save_schedule(schedule)
        return jsonify({"status": "success", "result": result})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/refresh/schedule", methods=["POST"])
def api_refresh_schedule():
    payload = request.get_json(silent=True) or {}
    period = payload.get("period_days")
    try:
        period = int(period)
        if period <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Period must be a positive integer."}), 400
    schedule = _load_schedule()
    schedule["period_days"] = period
    _save_schedule(schedule)
    response_schedule = {
        **schedule,
        "next_run": _format_schedule_next(schedule),
        "last_run_display": _format_schedule_dt(schedule.get("last_run")),
    }
    return jsonify({"status": "success", "schedule": response_schedule})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

