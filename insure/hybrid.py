"""Optional MiniLM + in-memory Qdrant retrieval fused with BM25 using RRF (k=60)."""
from __future__ import annotations

import hashlib
import threading

from .policy_store import Clause, PolicyStore

RRF_K = 60
_lock = threading.RLock()
_client = None
_embedder = None
_index_key = None


def _fingerprint(store: PolicyStore) -> str:
    payload = "\n".join(f"{c.policy_id}|{c.clause_id}|{c.kind}|{c.title}|{c.text}" for c in store.clauses)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ensure_index(store: PolicyStore):
    global _client, _embedder, _index_key
    import offline  # noqa: F401; force local-only model loading before transformers imports
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from sentence_transformers import SentenceTransformer
    import offline  # set Hugging Face to local-only before loading models

    key = _fingerprint(store)
    if _client is not None and key == _index_key:
        return
    if _client is not None:
        _client.close()
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    _client = QdrantClient(":memory:")
    _client.create_collection("policy_clauses", vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    if store.clauses:
        vectors = _embedder.encode([f"{c.title}. {c.text}" for c in store.clauses], normalize_embeddings=True)
        points = [PointStruct(id=i, vector=vectors[i].tolist(), payload={"index": i,
                   "policy_id": c.policy_id, "kind": c.kind}) for i, c in enumerate(store.clauses)]
        _client.upsert("policy_clauses", points=points)
    _index_key = key


def hybrid_search(store: PolicyStore, query: str, top_k: int = 5, policy_id: str | None = None,
                  kinds: set[str] | None = None) -> list[tuple[Clause, float]]:
    """Return an RRF-ranked union of BM25 and MiniLM/Qdrant candidates."""
    with _lock:
        _ensure_index(store)
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
        conditions = []
        if policy_id:
            conditions.append(FieldCondition(key="policy_id", match=MatchValue(value=policy_id)))
        if kinds:
            conditions.append(FieldCondition(key="kind", match=MatchAny(any=list(kinds))))
        query_filter = Filter(must=conditions) if conditions else None
        vector = _embedder.encode(query, normalize_embeddings=True).tolist()
        dense = _client.query_points(collection_name="policy_clauses", query=vector,
                                     query_filter=query_filter, limit=max(top_k * 4, top_k)).points
        dense_ranked = [store.clauses[p.payload["index"]] for p in dense]
        bm25_hits = store.search(query, top_k=max(top_k * 4, top_k), policy_id=policy_id, kinds=kinds)
        bm25_ranked = [c for c, _ in bm25_hits]
        bm25_score = {(c.policy_id, c.clause_id): score for c, score in bm25_hits}
        scores: dict[tuple[str, str], float] = {}
        objects: dict[tuple[str, str], Clause] = {}
        for ranking in (bm25_ranked, dense_ranked):
            for rank, clause in enumerate(ranking, start=1):
                identity = (clause.policy_id, clause.clause_id)
                objects[identity] = clause
                scores[identity] = scores.get(identity, 0.0) + 1.0 / (RRF_K + rank)
        ranked = sorted(scores, key=lambda k: scores[k], reverse=True)[:top_k]
        # Preserve the rule-based confidence scale used by the verifier, while RRF controls rank.
        return [(objects[key], max(bm25_score.get(key, 0.0), scores[key] * 100.0)) for key in ranked]


