from __future__ import annotations

import argparse
import json
import logging
import os
import re
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from functools import lru_cache
from typing import Dict, Iterable, List, Optional

import feedparser
import requests
import trafilatura
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("rss_ingest")

DATA_DIR = Path(os.getenv("INGEST_OUTPUT_DIR", "data"))
DATA_DIR.mkdir(exist_ok=True)
SOURCES_FILE = Path(os.getenv("INGEST_SOURCES_FILE", "sources.txt"))
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "aec_signals")
STACKIT_MODEL = os.getenv("STACKIT_MODEL_SERVING_MODEL")
STACKIT_BASE_URL = os.getenv("STACKIT_MODEL_SERVING_BASE_URL")
STACKIT_TOKEN = os.getenv("STACKIT_MODEL_SERVING_AUTH_TOKEN")
STACKIT_EMBED_MODEL = os.getenv("STACKIT_EMBED_MODEL", "intfloat/e5-mistral-7b-instruct")
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

USER_AGENT = "AEC-RSS-Qdrant-Agent/1.0"

CATEGORY_CHOICES: List[str] = [
    "AI-Driven Project Planning in AEC trends",
    "Smart Materials & Robotics in AEC",
    "XR and Digital Twin in AEC",
    "Smart Materials & Robotics",
    "XR & Digital Twin",
    "Site Process Automation",
    "Modular Construction Systems",
    "Sustainable Concrete & Materials",
    "Logistics & Supply Chain Optimization",
    "Research Update",
    "News Feed",
]

LEVEL_BUCKETS = {
    "high": {"label": "High", "scores": {5, 4}},
    "medium": {"label": "Medium", "scores": {3}},
    "low": {"label": "Low", "scores": {2, 1}},
}


@dataclass(frozen=True)
class RssSource:
    name: str
    url: str
    category: str


def _init_llm() -> ChatOpenAI:
    if not STACKIT_MODEL:
        raise RuntimeError("STACKIT_MODEL_SERVING_MODEL is not configured.")
    return ChatOpenAI(
        model=STACKIT_MODEL,
        base_url=STACKIT_BASE_URL,
        api_key=STACKIT_TOKEN,
        temperature=0.2,
        max_retries=3,
    )


def _init_embeddings() -> Optional[OpenAIEmbeddings]:
    if not STACKIT_TOKEN or not STACKIT_BASE_URL:
        logger.warning("STACKIT embedding credentials missing; defaulting to local embeddings.")
        return None
    return OpenAIEmbeddings(
        model=STACKIT_EMBED_MODEL,
        base_url=STACKIT_BASE_URL,
        api_key=STACKIT_TOKEN,
    )


def _init_qdrant() -> Optional[QdrantClient]:
    if not QDRANT_URL or not QDRANT_API_KEY:
        logger.warning("Qdrant credentials missing; skipping vector upload.")
        return None
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def _load_sources(path: Path) -> List[RssSource]:
    if not path.exists():
        raise FileNotFoundError(f"Sources file not found: {path}")
    sources: List[RssSource] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue
            name, url = parts[0], parts[1]
            category = parts[2] if len(parts) > 2 else "General"
            sources.append(RssSource(name=name, url=url, category=category))
    if not sources:
        raise ValueError(f"No sources defined in {path}")
    return sources


def _fetch_article_text(url: str) -> str:
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        extracted = trafilatura.extract(resp.text, url=url, include_links=False)
        return (extracted or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to pull article body for %s (%s)", url, exc)
        return ""


def _safe_json_parse(value: str) -> dict:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]+\}", value)
        if not match:
            raise
        return json.loads(match.group(0))


def _normalize_category(raw: Optional[str]) -> str:
    if not raw:
        return CATEGORY_CHOICES[0]
    target = raw.strip().lower()
    for choice in CATEGORY_CHOICES:
        if target == choice.lower():
            return choice
    for choice in CATEGORY_CHOICES:
        if target in choice.lower() or choice.lower() in target:
            return choice
    return CATEGORY_CHOICES[0]


def _canonical_signal_level(raw_level) -> tuple[str, int]:
    if raw_level is None:
        return ("Low", 2)
    try:
        num = int(raw_level)
        for meta in LEVEL_BUCKETS.values():
            if num in meta["scores"]:
                return (meta["label"], num)
    except (ValueError, TypeError):
        pass
    label = str(raw_level).strip().lower()
    for key, meta in LEVEL_BUCKETS.items():
        if label == key or label == meta["label"].lower():
            # pick representative score (max of bucket)
            score = max(meta["scores"])
            return (meta["label"], score)
    if "med" in label:
        return ("Medium", 3)
    if "high" in label or "critical" in label:
        return ("High", 5)
    return ("Low", 2)


def _analyze_article(
    llm: ChatOpenAI, title: str, summary: str, article_text: str
) -> dict:
    categories_clause = ", ".join(CATEGORY_CHOICES)
    instruction = (
        "You are an AEC (Architecture, Engineering, Construction) market intelligence analyst. "
        "Given the article details, produce JSON with these fields: "
        "title (<=120 chars), summary (<=80 words), "
        f"category (choose EXACTLY one from: {categories_clause}), "
        "keywords (array of <=6 lowercase keywords), entities (array of entity names), "
        "signal_level (string: High, Medium, or Low describing market urgency). "
        "Only output JSON, no prose."
    )
    content = (
        f"TITLE:\n{title}\n\nSUMMARY SNIPPET:\n{summary}\n\n"
        f"FULL TEXT (truncated):\n{article_text[:4000]}"
    )
    response = llm.invoke(
        [
            {"role": "system", "content": instruction},
            {"role": "user", "content": content},
        ]
    )
    data = _safe_json_parse(response.content)
    # Normalize shapes
    data["title"] = data.get("title") or title
    data["summary"] = data.get("summary") or summary
    data["category"] = _normalize_category(data.get("category"))
    data["keywords"] = [str(k).strip().lower() for k in data.get("keywords", []) if str(k).strip()]
    data["entities"] = [str(e).strip() for e in data.get("entities", []) if str(e).strip()]
    level_label, level_score = _canonical_signal_level(data.get("signal_level"))
    data["signal_level"] = level_label
    data["signal_level_numeric"] = level_score
    return data


def _sanitize_for_embedding(raw: str) -> str:
    normalized = unicodedata.normalize("NFKD", raw or "")
    return normalized.encode("ascii", "ignore").decode("ascii", errors="ignore")


def _hash_fallback_embedding(text: str, dim: int = 384) -> List[float]:
    import hashlib

    base = text or "untitled signal"
    vec: List[float] = []
    for i in range(dim):
        digest = hashlib.sha256(f"{base}|{i}".encode("utf-8", "ignore")).digest()
        val = int.from_bytes(digest[:4], "big") / 2**32
        vec.append((val * 2) - 1)
    return vec


@lru_cache(maxsize=1)
def _local_embedder() -> SentenceTransformer:
    logger.info("Loading local embedding model: %s", LOCAL_EMBED_MODEL)
    return SentenceTransformer(LOCAL_EMBED_MODEL)


def _local_embed(text: str) -> List[float]:
    try:
        model = _local_embedder()
        vector = model.encode(text or "empty signal", normalize_embeddings=True)
        return vector.tolist()
    except Exception as exc:  # noqa: BLE001
        logger.error("Local embedding failed (%s); falling back to hash vector.", exc)
        return _hash_fallback_embedding(text)


def _embed_text(embedding_client: Optional[OpenAIEmbeddings], text: str) -> List[float]:
    payload = _sanitize_for_embedding(text[:2000])
    if not payload.strip():
        payload = "empty signal"
    if embedding_client is None:
        return _local_embed(payload)
    try:
        return embedding_client.embed_query(payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Remote embedding failed (%s); switching to local model.", exc)
        return _local_embed(payload)


def _ensure_collection(client: QdrantClient, vector_size: int) -> None:
    if client.collection_exists(QDRANT_COLLECTION):
        return
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
    )
    logger.info("Created Qdrant collection %s (size=%s)", QDRANT_COLLECTION, vector_size)


def _upsert_records(client: QdrantClient, records: List[dict]) -> int:
    if not records:
        return 0
    vectors: List[rest.PointStruct] = []
    for record in records:
        vector = record.pop("vector", None)
        if not vector:
            continue
        vectors.append(
            rest.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=record,
            )
        )
    if not vectors:
        return 0
    _ensure_collection(client, len(vectors[0].vector))
    client.upsert(collection_name=QDRANT_COLLECTION, points=vectors)
    return len(vectors)


def _load_existing_links(client: QdrantClient, chunk: int = 256) -> set[str]:
    links: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=chunk,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for point in points:
            link = (point.payload or {}).get("link")
            if link:
                links.add(link)
        if offset is None:
            break
    return links


def collect_signals(
    sources: Iterable[RssSource],
    max_items: Optional[int],
    llm: ChatOpenAI,
    embedding_client: Optional[OpenAIEmbeddings],
    existing_links: Optional[set[str]] = None,
) -> tuple[List[dict], int]:
    records: List[dict] = []
    skipped_existing = 0
    known_links = existing_links if existing_links is not None else set()
    for source in sources:
        logger.info("Fetching feed: %s (%s)", source.name, source.url)
        feed = feedparser.parse(source.url)
        entries = list(getattr(feed, "entries", []) or [])
        if max_items is not None:
            entries = entries[:max_items]
        for entry in entries:
            link = entry.get("link")
            if existing_links is not None and link and link in known_links:
                skipped_existing += 1
                continue
            title = entry.get("title", source.name)
            summary = entry.get("summary", "") or entry.get("description", "")
            published = entry.get("published") or entry.get("updated")
            article_text = _fetch_article_text(link) if link else ""
            analysis = _analyze_article(llm, title, summary, article_text)
            payload = {
                "source": source.name,
                "source_category": source.category,
                "link": link,
                "published": published,
                "title": analysis["title"],
                "summary": analysis["summary"],
                "category": analysis["category"],
                "keywords": analysis["keywords"],
                "entities": analysis["entities"],
                "signal_level": analysis["signal_level"],
                "signal_level_numeric": analysis["signal_level_numeric"],
                "signal_impact": analysis["signal_level"].lower(),
                "collected_at": datetime.utcnow().isoformat(),
            }
            embed_text = f"{analysis['title']}\n\n{analysis['summary']}\n\n{article_text}"
            payload["vector"] = _embed_text(embedding_client, embed_text)
            records.append(payload)
            if existing_links is not None and link:
                known_links.add(link)
    return records, skipped_existing


def _write_jsonl(records: List[dict], batch_name: str) -> Path:
    path = DATA_DIR / f"signals_{batch_name}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            safe = dict(row)
            safe.pop("vector", None)
            handle.write(json.dumps(safe, ensure_ascii=False) + "\n")
    return path


def _complete_ingest(records: List[dict], skipped_existing: int, qdrant: Optional[QdrantClient]) -> dict:
    if not records:
        return {
            "saved": 0,
            "skipped_existing": skipped_existing,
            "file": None,
            "qdrant_upserts": 0,
            "collection": QDRANT_COLLECTION,
        }
    batch_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_file = _write_jsonl(records, batch_id)
    qdrant_upserts = 0
    if qdrant:
        qdrant_upserts = _upsert_records(qdrant, records)
    return {
        "saved": len(records),
        "skipped_existing": skipped_existing,
        "file": str(out_file),
        "qdrant_upserts": qdrant_upserts,
        "collection": QDRANT_COLLECTION,
        "batch_id": batch_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest RSS feeds and push structured signals to Qdrant.")
    parser.add_argument("--max-items", type=int, default=int(os.getenv("INGEST_MAX_ITEMS", "3")))
    parser.add_argument("--dry-run", action="store_true", help="Skip Qdrant upload, just write JSONL.")
    args = parser.parse_args()

    result = run_ingest(max_items=args.max_items, skip_existing=True, dry_run=args.dry_run)
    logger.info("Ingest complete: %s", json.dumps(result))

def _ingest_sources(
    sources: List[RssSource],
    max_items: Optional[int],
    skip_existing: bool,
    dry_run: bool,
) -> dict:
    llm = _init_llm()
    embeddings = _init_embeddings()
    qdrant = None if dry_run else _init_qdrant()

    existing_links: Optional[set[str]] = None
    if skip_existing and qdrant:
        existing_links = _load_existing_links(qdrant)
        logger.info("Loaded %s existing links from Qdrant", len(existing_links))

    records, skipped_existing = collect_signals(sources, max_items, llm, embeddings, existing_links)
    return _complete_ingest(records, skipped_existing, qdrant)


def run_ingest(
    max_items: Optional[int] = None,
    skip_existing: bool = True,
    dry_run: bool = False,
) -> dict:
    sources = _load_sources(SOURCES_FILE)
    return _ingest_sources(list(sources), max_items, skip_existing, dry_run)


def ingest_single_source(
    name: str,
    url: str,
    category: str,
    max_items: Optional[int] = None,
    skip_existing: bool = True,
    dry_run: bool = False,
) -> dict:
    source = RssSource(name=name or "New source", url=url, category=category or "General")
    return _ingest_sources([source], max_items, skip_existing, dry_run)


def ingest_documents(
    documents: List[dict],
    skip_existing: bool = True,
    dry_run: bool = False,
) -> dict:
    if not documents:
        raise ValueError("No documents provided for ingestion.")
    llm = _init_llm()
    embeddings = _init_embeddings()
    qdrant = None if dry_run else _init_qdrant()
    records: List[dict] = []
    skipped_existing = 0
    for doc in documents:
        content = (doc.get("content") or "").strip()
        if not content:
            continue
        title = doc.get("title") or "Untitled document"
        summary = doc.get("summary") or ""
        category = doc.get("category") or "General"
        source = doc.get("source") or "Manual upload"
        link = doc.get("link")
        analysis = _analyze_article(llm, title, summary, content)
        payload = {
            "source": source,
            "source_category": category,
            "published": doc.get("published") or datetime.utcnow().isoformat(),
            "title": analysis["title"],
            "summary": analysis["summary"],
            "category": analysis["category"],
            "keywords": analysis["keywords"],
            "entities": analysis["entities"],
            "signal_level": analysis["signal_level"],
            "signal_level_numeric": analysis["signal_level_numeric"],
            "signal_impact": analysis["signal_level"].lower(),
            "collected_at": datetime.utcnow().isoformat(),
        }
        embed_text = f"{analysis['title']}\n\n{analysis['summary']}\n\n{content}"
        payload["vector"] = _embed_text(embeddings, embed_text)
        if link:
            payload["link"] = link
        if doc.get("local_file"):
            payload["local_file"] = doc["local_file"]
        if doc.get("original_filename"):
            payload["original_filename"] = doc["original_filename"]
        records.append(payload)
    return _complete_ingest(records, skipped_existing, qdrant)

if __name__ == "__main__":
    main()
