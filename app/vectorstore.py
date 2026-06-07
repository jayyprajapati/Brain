"""Qdrant Cloud wrapper. One collection per app (`app_name`); scoped within by
`doc_id` (one document) and optionally `namespace` (a tenant/user grouping)."""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .config import settings

_client: QdrantClient | None = None
# Collections we've already created/verified this process — skips redundant round-trips.
_ensured: set[str] = set()

_NAME_RE = re.compile(r"[^a-z0-9_-]+")


@dataclass
class Hit:
    text: str
    heading: str
    urls: list[str]
    doc_id: str
    score: float


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    return _client


def collection_name(app_name: str) -> str:
    """Map an `app_name` to its dedicated Qdrant collection name."""
    slug = _NAME_RE.sub("_", (app_name or "").strip().lower()).strip("_")
    if not slug:
        raise ValueError("app_name is required to resolve a collection")
    return f"{settings.qdrant_collection_prefix}{slug}"


def ensure_collection(app_name: str, dim: int) -> str:
    """Create the app's collection (+ tenant indexes) once. Returns the collection name."""
    name = collection_name(app_name)
    if name in _ensured:
        return name
    client = _get_client()
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        # Index the in-collection tenant fields so filtered search/delete stay fast.
        for field in ("doc_id", "namespace"):
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
    _ensured.add(name)
    return name


def _exists(name: str) -> bool:
    return name in _ensured or _get_client().collection_exists(name)


def _filter(doc_ids: list[str] | None, namespace: str | None) -> qm.Filter | None:
    must: list = []
    if doc_ids:
        must.append(qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=list(doc_ids))))
    if namespace:
        must.append(qm.FieldCondition(key="namespace", match=qm.MatchValue(value=namespace)))
    return qm.Filter(must=must) if must else None


def upsert(
    app_name: str,
    doc_id: str,
    vectors: list,
    payloads: list[dict],
    namespace: str | None = None,
) -> None:
    name = collection_name(app_name)
    base = {"doc_id": doc_id}
    if namespace:
        base["namespace"] = namespace
    points = [
        qm.PointStruct(
            id=str(uuid.uuid4()),
            vector=vec.tolist() if hasattr(vec, "tolist") else list(vec),
            payload={**base, **payload},
        )
        for vec, payload in zip(vectors, payloads)
    ]
    if points:
        _get_client().upsert(collection_name=name, points=points)


def delete(app_name: str, doc_id: str | None = None, namespace: str | None = None) -> int:
    """Remove chunks matching `doc_id` and/or `namespace`. Returns count removed.

    At least one selector must be provided to avoid wiping the whole collection."""
    if not doc_id and not namespace:
        raise ValueError("delete requires a doc_id or namespace")
    client = _get_client()
    name = collection_name(app_name)
    if not _exists(name):
        return 0
    flt = _filter([doc_id] if doc_id else None, namespace)
    before = client.count(collection_name=name, count_filter=flt, exact=True).count
    client.delete(collection_name=name, points_selector=qm.FilterSelector(filter=flt))
    return before


def search(
    app_name: str,
    query_vector,
    doc_ids: list[str] | None,
    limit: int,
    namespace: str | None = None,
) -> list[Hit]:
    client = _get_client()
    name = collection_name(app_name)
    if not _exists(name):
        return []
    results = client.query_points(
        collection_name=name,
        query=query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector),
        query_filter=_filter(doc_ids, namespace),
        limit=limit,
        with_payload=True,
    ).points
    hits: list[Hit] = []
    for r in results:
        payload = r.payload or {}
        hits.append(
            Hit(
                text=payload.get("text", ""),
                heading=payload.get("heading", ""),
                urls=payload.get("urls", []) or [],
                doc_id=payload.get("doc_id", ""),
                score=float(r.score),
            )
        )
    return hits


def max_similarity(app_name: str, vector, namespace: str | None) -> float:
    """Return the cosine score of the nearest existing vector within `namespace`.

    Used to detect near-duplicate chunks before ingesting them. Returns 0.0 when
    the collection or namespace is empty."""
    client = _get_client()
    name = collection_name(app_name)
    if not _exists(name):
        return 0.0
    results = client.query_points(
        collection_name=name,
        query=vector.tolist() if hasattr(vector, "tolist") else list(vector),
        query_filter=_filter(None, namespace),
        limit=1,
        with_payload=False,
    ).points
    return float(results[0].score) if results else 0.0
