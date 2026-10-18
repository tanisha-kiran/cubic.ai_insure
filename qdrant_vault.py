"""
Layer 3 — Sovereign Vault (Qdrant vector search)

Replaces the stubbed retriever_node in agentic_graph.py with real semantic
search over your ingested StructuredContent (the same JSON files already
produced by process_backend.py and process_audio.py).

Runs Qdrant in LOCAL, EMBEDDED mode (no separate server process, no cloud
Qdrant account) -- this matches the project's air-gapped/offline thesis.
Data persists to a local folder on disk between runs.

Embedding model: all-MiniLM-L6-v2 (sentence-transformers) -- small (~90MB),
runs on CPU, downloads once from Hugging Face then works fully offline.
This download step is the ONLY point that needs internet access; every
query after that is 100% local.
"""
import offline  # noqa: F401  (models load from the local cache only; ALLOW_ONLINE=1 to download)
import os
import json
import glob
import atexit
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


QDRANT_PATH = "./qdrant_local_storage"
COLLECTION_NAME = "sovereign_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


CHUNK_TARGET_CHARS = 600


def merge_blocks(blocks, target_chars=CHUNK_TARGET_CHARS):
    """
    Merge consecutive PyMuPDF blocks on a page into ~target_chars chunks.
    Raw blocks are often a single line ("Total:" in one block, "$1,450" in
    the next), which embeds poorly and splits facts across chunks. The merged
    bbox is the union, which is what crop_to_evidence needs.
    """
    merged, lines, boxes = [], [], []

    def flush():
        if lines:
            box = None
            if boxes:
                box = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                       max(b[2] for b in boxes), max(b[3] for b in boxes)]
            merged.append({"text_lines": list(lines), "bbox": box})
            lines.clear()
            boxes.clear()

    for block in blocks:
        block_lines = [l for l in block.get("text_lines", []) if l.strip()]
        if not block_lines:
            continue
        if lines and sum(len(l) for l in lines) + sum(len(l) for l in block_lines) > target_chars:
            flush()
        lines.extend(block_lines)
        if block.get("bbox"):
            boxes.append(block["bbox"])
    flush()
    return merged


class SovereignVault:
    """
    Local/offline Qdrant vector vault for document + audio retrieval.

    Improvements over the original:
    - Explicit Qdrant client cleanup to avoid Python shutdown warnings.
    - Stable point IDs based on source file/content rather than resetting to 0.
    - Duplicate-content protection within each indexing run.
    - Optional modality filtering (document/audio).
    - Simple query-aware modality routing for audio/recording questions.
    """

    def __init__(self):
        print("[+] Loading local embedding model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        print("[+] Initializing local Qdrant storage...")
        self.client = QdrantClient(path=QDRANT_PATH)
        self._closed = False

        if not self.client.collection_exists(COLLECTION_NAME):
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            print(f"[+] Created new collection: {COLLECTION_NAME}")
        else:
            print(f"[+] Using existing collection: {COLLECTION_NAME}")

    def _all_points(self):
        offset = None
        while True:
            points, offset = self.client.scroll(
                COLLECTION_NAME, limit=512, offset=offset, with_payload=["doc_id"], with_vectors=False
            )
            yield from points
            if offset is None:
                break

    def _delete_ids(self, ids):
        from qdrant_client.models import PointIdsList

        ids = list(ids)
        for i in range(0, len(ids), 512):
            self.client.delete(COLLECTION_NAME, points_selector=PointIdsList(points=ids[i:i + 512]))
        return len(ids)

    def reset(self):
        """
        Remove every point. (delete_collection is unreliable in local mode on
        Windows: the locked sqlite file survives and is silently reloaded.)
        """
        removed = self._delete_ids(p.id for p in self._all_points())
        print(f"[+] Reset collection: removed {removed} points")

    def delete_document(self, doc_id):
        """Remove all chunks of one document (call when a document is deleted)."""
        return self._delete_ids(
            p.id for p in self._all_points() if (p.payload or {}).get("doc_id") == doc_id
        )

    def prune(self, keep_doc_ids, keep_ids=None):
        """
        Delete points whose document is no longer in ocr_cache, and (if
        keep_ids is given) stale chunks of live documents left by an older
        chunking scheme.
        """
        stale = [
            p.id for p in self._all_points()
            if (p.payload or {}).get("doc_id") not in keep_doc_ids
            or (keep_ids is not None and p.id not in keep_ids)
        ]
        removed = self._delete_ids(stale)
        if removed:
            print(f"[+] Pruned {removed} stale chunks")
        return removed

    def close(self):
        """Explicitly close Qdrant before Python shutdown."""
        if not self._closed:
            try:
                self.client.close()
            except Exception as e:
                print(f"[!] Qdrant close warning: {e}")
            finally:
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _stable_point_id(self, doc_id, modality, text, location=""):
        """
        Generate a stable integer point ID from source content.
        This prevents accidental ID collisions across indexing runs.
        """
        import hashlib

        raw = f"{doc_id}|{modality}|{location}|{text}".encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()

        # Qdrant accepts uint/int IDs. Keep this within signed 64-bit range.
        return int(digest[:15], 16)

    def index_structured_content(self, ocr_cache_dir="ocr_cache"):
        """
        Index StructuredContent JSON files produced by the document/audio
        processing pipeline.

        Supported formats:
        - Document JSON containing "pages" -> modality=document
        - Audio JSON containing "segments" -> modality=audio
        """

        json_files = sorted(
            glob.glob(os.path.join(ocr_cache_dir, "*.json"))
        )

        print(
            f"[+] Found {len(json_files)} StructuredContent "
            f"files to index"
        )

        if not json_files:
            print(
                "[!] No JSON files found in "
                f"{ocr_cache_dir}"
            )
            return 0

        points = []
        seen = set()

        for filepath in json_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[!] Could not read {filepath}: {e}")
                continue

            doc_id = data.get(
                "doc_id",
                os.path.basename(filepath),
            )
            filename = data.get("filename") or data.get("source_file")
            if not filename:
                for source_dir in ("uploaded_pdfs", "uploaded_audio"):
                    matches = glob.glob(os.path.join(source_dir, f"{doc_id}.*"))
                    if matches:
                        filename = os.path.basename(matches[0])
                        break
            filename = filename or os.path.basename(filepath)

            # -----------------------------
            # DOCUMENT CONTENT
            # -----------------------------
            if "pages" in data:
                for page_key, page_val in data["pages"].items():

                    for block_index, block in enumerate(
                        merge_blocks(page_val.get("spatial_blocks", []))
                    ):
                        text = " ".join(
                            block.get("text_lines", [])
                        ).strip()

                        if not text:
                            continue

                        location = f"page:{page_key}|bbox:{block.get('bbox')}"

                        # Prevent exact duplicate chunks
                        dedup_key = (
                            doc_id,
                            "document",
                            text,
                            location,
                        )

                        if dedup_key in seen:
                            continue

                        seen.add(dedup_key)

                        point_id = self._stable_point_id(
                            doc_id,
                            "document",
                            text,
                            location,
                        )

                        points.append(
                            {
                                "id": point_id,
                                "text": text,
                                "doc_id": doc_id,
                                "document_id": doc_id,
                                "filename": filename,
                                "owner": data.get("owner"),
                                "chunk_id": f"{doc_id}:{page_key}:{block_index}",
                                "modality": "document",
                                "page": page_key,
                                "bbox": block.get("bbox"),
                                "image_path": page_val.get("image_path"),
                            }
                        )

            # -----------------------------
            # AUDIO CONTENT
            # -----------------------------
            elif "segments" in data:
                for segment_index, segment in enumerate(data["segments"]):
                    text = segment.get("text", "").strip()

                    if not text:
                        continue

                    start = segment.get("start")
                    end = segment.get("end")

                    location = f"start:{start}|end:{end}"

                    dedup_key = (
                        doc_id,
                        "audio",
                        text,
                        location,
                    )

                    if dedup_key in seen:
                        continue

                    seen.add(dedup_key)

                    point_id = self._stable_point_id(
                        doc_id,
                        "audio",
                        text,
                        location,
                    )

                    points.append(
                        {
                            "id": point_id,
                            "text": text,
                            "doc_id": doc_id,
                                "document_id": doc_id,
                                "filename": filename,
                                "owner": data.get("owner"),
                                "chunk_id": f"{doc_id}:segment:{segment_index}",
                            "modality": "audio",
                            "start": start,
                            "end": end,
                        }
                    )

            else:
                print(
                    f"[!] Skipping unsupported JSON: "
                    f"{os.path.basename(filepath)}"
                )

        if not points:
            print(
                "[!] No indexable content found. "
                "Check ocr_cache/."
            )
            return 0

        texts = [p["text"] for p in points]

        print(
            f"[+] Embedding {len(texts)} unique chunks..."
        )

        embeddings = self.embedder.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        qdrant_points = []

        for i, p in enumerate(points):
            payload = {
                k: v
                for k, v in p.items()
                if k != "id"
            }

            qdrant_points.append(
                PointStruct(
                    id=p["id"],
                    vector=embeddings[i].tolist(),
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=qdrant_points,
        )

        # ocr_cache is the source of truth: drop chunks of deleted documents
        # and leftovers from older chunking schemes.
        self.prune(
            keep_doc_ids={p["doc_id"] for p in points},
            keep_ids={p["id"] for p in points},
        )

        document_count = sum(
            p["modality"] == "document"
            for p in points
        )
        audio_count = sum(
            p["modality"] == "audio"
            for p in points
        )

        print(
            f"[✓] Indexed {len(points)} chunks "
            f"({document_count} document, "
            f"{audio_count} audio)"
        )

        return len(points)

    def search(
        self,
        query: str,
        top_k: int = 5,
        modality_filter: str = None,
        document_id: str = None,
        owner: str = None,
    ):
        """
        Semantic search with optional modality filtering.

        modality_filter:
            None       -> search all modalities
            document   -> document chunks only
            audio      -> audio transcript chunks only
        """

        query = query.strip()

        if not query:
            return []

        query_vector = self.embedder.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        query_filter = None

        if modality_filter or document_id:
            from qdrant_client.models import (
                Filter,
                FieldCondition,
                MatchValue,
            )

            conditions = []
            if modality_filter:
                conditions.append(
                    FieldCondition(
                        key="modality",
                        match=MatchValue(value=modality_filter),
                    )
                )
            if document_id:
                conditions.append(
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=document_id),
                    )
                )
            query_filter = Filter(must=conditions)

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            # With an owner, over-fetch: other users' chunks are removed below, and we still want top_k.
            limit=top_k * 8 if owner is not None else top_k,
        )
        points = results.points
        if owner is not None:
            # A user sees their own documents plus shared ones (no owner, e.g. the demo samples).
            points = [p for p in points if p.payload.get("owner") in (None, owner)][:top_k]

        return [
            {
                "text": r.payload["text"],
                "score": round(r.score, 3),
                "doc_id": r.payload["doc_id"],
                "document_id": r.payload.get("document_id", r.payload["doc_id"]),
                "filename": r.payload.get("filename"),
                "chunk_id": r.payload.get("chunk_id"),
                "modality": r.payload["modality"],
                **{
                    k: v
                    for k, v in r.payload.items()
                    if k not in (
                        "text",
                        "doc_id",
                        "modality",
                    )
                },
            }
            for r in points
        ]


def infer_modality(question: str):
    """
    Lightweight deterministic modality router.

    This does not generate an answer. It only decides whether the
    retrieval stage should prefer audio or document content.
    """

    q = question.lower()

    audio_keywords = [
        "recording",
        "audio",
        "said",
        "spoke",
        "spoken",
        "heard",
        "voice",
        "conversation",
        "transcript",
        "transcription",
        "what did they say",
        "what was said",
        "what did he say",
        "what did she say",
        "what did they mention",
    ]

    document_keywords = [
        "document",
        "pdf",
        "page",
        "table",
        "invoice",
        "bill",
        "form",
        "written",
        "shown",
        "listed",
        "total",
        "balance",
        "amount",
    ]

    audio_score = sum(
        1 for keyword in audio_keywords
        if keyword in q
    )

    document_score = sum(
        1 for keyword in document_keywords
        if keyword in q
    )

    if audio_score > document_score and audio_score > 0:
        return "audio"

    if document_score > audio_score and document_score > 0:
        return "document"

    return None


# Singleton used by agentic_graph.py
_vault_instance = None


def _get_vault_singleton():
    global _vault_instance

    if _vault_instance is None:
        _vault_instance = SovereignVault()

        # Register cleanup for normal program termination.
        atexit.register(_vault_instance.close)

    return _vault_instance


def retriever_node_qdrant(state):
    """
    Drop-in replacement for the original retriever_node.

    Expected AgentState fields:
        state["question"]
        state["retrieved_context"]
        state["trace"]
    """

    vault = _get_vault_singleton()

    question = state["question"]

    # Determine which modality should be searched.
    modality = infer_modality(question)

    results = vault.search(
        question,
        top_k=3,
        modality_filter=modality,
    )

    # Fallback:
    # If a modality-specific search finds nothing, search all modalities.
    if not results and modality is not None:
        results = vault.search(
            question,
            top_k=3,
            modality_filter=None,
        )

        fallback_used = True
    else:
        fallback_used = False

    if results:
        state["retrieved_context"] = "\n".join(
            r["text"]
            for r in results
        )

        state["trace"].append(
            f"Retriever -> Qdrant returned "
            f"{len(results)} chunks "
            f"(modality={modality or 'all'}, "
            f"fallback={fallback_used}, "
            f"top score={results[0]['score']})"
        )
    else:
        state["retrieved_context"] = ""

        state["trace"].append(
            "Retriever -> Qdrant returned no results"
        )

    return state


def close_vault():
    """Optional explicit cleanup for applications."""
    global _vault_instance

    if _vault_instance is not None:
        _vault_instance.close()
        _vault_instance = None


if __name__ == "__main__":
    # Standalone test mode.  `python qdrant_vault.py --reset` rebuilds the
    # index from scratch (run once after changing the chunking).
    import sys

    vault = SovereignVault()
    if "--reset" in sys.argv:
        vault.reset()

    try:
        count = vault.index_structured_content(
            "ocr_cache"
        )

        if count > 0:
            print("\n" + "=" * 60)
            print("TEST SEARCHES")
            print("=" * 60)

            test_queries = [
                "How much money is owed?",
                "What was said in the recording?",
            ]

            for q in test_queries:
                modality = infer_modality(q)

                print(f"\nQuery: {q}")
                print(
                    f"Routing modality: "
                    f"{modality or 'all'}"
                )

                results = vault.search(
                    q,
                    top_k=3,
                    modality_filter=modality,
                )

                # Fallback to all modalities if needed.
                if not results and modality:
                    print(
                        "[!] No modality-specific results; "
                        "falling back to all modalities."
                    )
                    results = vault.search(
                        q,
                        top_k=3,
                        modality_filter=None,
                    )

                for r in results:
                    print(
                        f"  [{r['score']}] "
                        f"({r['modality']}) "
                        f"{r['text'][:120]}"
                    )

    finally:
        vault.close()
        print("\n[+] Qdrant client closed cleanly.")
