"""
cubic.ai backend: login, per-user documents, live (streaming) answers, voice transcription.

Run:
    python app_backend.py                       # this PC only, http://127.0.0.1:8000
    python app_backend.py --https               # + HTTPS on :8443 (needed for the microphone off-localhost)
    python app_backend.py --lan --https         # reachable from other devices on the network
See DEPLOY.md for the offline install and the security notes.
"""
import offline  # noqa: F401  (must be first: forces local-only model loading)

import json
import os
import queue
import re
import sys
import threading
import time
from collections import defaultdict, deque
from difflib import SequenceMatcher
from pathlib import Path

import fitz  # PyMuPDF

# Ingestion modules print "✓"; on a redirected Windows console (cp1252) that
# raises UnicodeEncodeError and fails the upload. Never let logging crash a request.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

import agentic_graph as ag
import auth
from process_office import SUPPORTED_SUFFIXES
from insurance_api import answer_for_chat, router as insurance_router, revoke_submit_permission

class LoginBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=1, max_length=32)
    password: str = Field(min_length=1, max_length=512)

    @field_validator("username")
    @classmethod
    def normalized_username(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("username is required")
        return value

app = FastAPI(title="cubic.ai", docs_url=None, redoc_url=None, openapi_url=None)  # no public API explorer

for folder in ("uploaded_pdfs", "uploaded_audio", "documents", "ocr_cache", auth.DATA_DIR):
    os.makedirs(folder, exist_ok=True)
app.mount("/assets", StaticFiles(directory="assets"), name="assets")   # logo files for the web page
app.include_router(insurance_router)

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "40"))
MAX_AUDIO_MB = 25
STATE_DIR = os.path.join(auth.DATA_DIR, "chats")
AUTH_REQUIRED = os.environ.get("QUIETLY_AUTH", "on").lower() != "off"
PUBLIC_PATHS = {"/", "/health", "/auth/login", "/auth/me", "/favicon.ico"}
PUBLIC_PREFIXES = ("/assets/",)   # the logo must load before anyone signs in

# Recent turns per (user, chat), for follow-up questions. In memory only.
SESSIONS = defaultdict(lambda: deque(maxlen=6))


# ======================================================================== security middleware
@app.middleware("http")
async def guard(request: Request, call_next):
    path = request.url.path
    request.state.user = None
    if path not in PUBLIC_PATHS and not path.startswith(PUBLIC_PREFIXES):
        if AUTH_REQUIRED:
            user = auth.read_token(request.cookies.get(auth.COOKIE_NAME))
            if not user:
                return JSONResponse({"detail": "Sign in required"}, status_code=401)
        else:
            user = "local"
        request.state.user = user
    elif AUTH_REQUIRED:
        request.state.user = auth.read_token(request.cookies.get(auth.COOKIE_NAME))
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "microphone=(self), camera=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; "
        "frame-ancestors 'none'; base-uri 'none'")
    if path.startswith(("/auth", "/query", "/documents", "/state", "/transcribe", "/status", "/insurance", "/insure")):
        response.headers["Cache-Control"] = "no-store"
    return response


def _user(request: Request) -> str:
    return request.state.user


def _owner_filter(user: str):
    """What retrieval may see: admins and the single local user see everything, others only their own."""
    return None if (user == "local" or auth.is_admin(user)) else user


def _check_access(user: str, document_id: str, write: bool = False) -> dict:
    """The document's data if this user may use it, else 404 (never reveal that it exists)."""
    data, _ = ag.load_structured(document_id)
    if not data:
        raise HTTPException(status_code=404, detail="Document not found")
    owner = data.get("owner")
    if user == "local" or auth.is_admin(user) or owner == user or (owner is None and not write):
        return data
    raise HTTPException(status_code=404, detail="Document not found")


# ======================================================================== pages and auth
@app.get("/", response_class=FileResponse)
def quietly_frontend():
    return FileResponse("frontend_quietly.html", media_type="text/html")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/auth/login")
async def login(request: Request, body: LoginBody):
    client = request.client.host if request.client else "?"
    ok, message = auth.check_login(body.username, body.password, client)
    if not ok:
        raise HTTPException(status_code=429 if message.startswith("Too many") else 401, detail=message)
    name = body.username
    response = JSONResponse({"user": name, "admin": auth.is_admin(name)})
    response.set_cookie(auth.COOKIE_NAME, auth.make_token(name), max_age=auth.SESSION_SECONDS, httponly=True,
                        samesite="lax", secure=request.url.scheme == "https")
    return response


@app.post("/auth/logout")
def logout(request: Request):
    auth.revoke_token(request.cookies.get(auth.COOKIE_NAME))
    revoke_submit_permission(getattr(request.state, "user", None))
    response = JSONResponse({"ok": True})
    response.delete_cookie(auth.COOKIE_NAME)
    return response


@app.get("/auth/me")
def whoami(request: Request):
    user = request.state.user if AUTH_REQUIRED else "local"
    return {"user": user, "admin": bool(user) and (user == "local" or auth.is_admin(user)),
            "auth_required": AUTH_REQUIRED,
            "models": {"reasoner": ag.VL_LARGE, "writer": ag.WRITER_MODEL},
            "default_detail": ag.ANSWER_DETAIL, "profile": ag.PROFILE["name"]}


@app.get("/status")
def status():
    """Is the local model server up, and which models are loaded right now?"""
    import requests
    try:
        loaded = requests.get("http://localhost:11434/api/ps", timeout=1.5).json().get("models", [])
        return {"ollama": True, "loaded": [m["name"] for m in loaded]}
    except Exception:
        return {"ollama": False, "loaded": []}


# ======================================================================== documents
def _doc_summary(data: dict, cache_file: str, user: str) -> dict:
    pages = data.get("pages", {})
    doc_id = data.get("doc_id", os.path.splitext(cache_file)[0])
    progress = ag.VISUAL_PROGRESS.get(doc_id)
    return {
        "document_id": doc_id,
        "filename": data.get("filename") or data.get("source_file") or cache_file,
        "pages": len(pages) if pages else None,
        "modality": data.get("modality", "document" if pages else "audio"),
        "has_image": any(p.get("image_path") for p in pages.values()),
        "mine": data.get("owner") == user,
        "shared": data.get("owner") is None,
        "reading": progress if progress and progress["state"] in ("queued", "reading") else None,
    }


@app.get("/documents")
def list_uploaded_documents(request: Request):
    user, documents = _user(request), []
    for cache_path in sorted((n for n in os.listdir("ocr_cache") if n.endswith(".json")), reverse=True):
        try:
            with open(os.path.join("ocr_cache", cache_path), "r", encoding="utf-8") as source:
                data = json.load(source)
        except (OSError, json.JSONDecodeError):
            continue
        owner = data.get("owner")
        if user == "local" or auth.is_admin(user) or owner in (None, user):
            documents.append(_doc_summary(data, cache_path, user))
    return {"documents": documents}


@app.get("/documents/{document_id}/status")
def document_status(document_id: str, request: Request):
    _check_access(_user(request), document_id)
    progress = ag.VISUAL_PROGRESS.get(document_id)
    return {"document_id": document_id, **(progress or {"state": "ready"})}


@app.get("/documents/{document_id}/page/{page}/image")
def document_page_image(document_id: str, page: int, request: Request):
    """The rendered page (for highlighting where an answer came from)."""
    data = _check_access(_user(request), document_id)
    entry = data.get("pages", {}).get(f"page_{page}")
    path = entry.get("image_path") if entry else None
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="No page image")
    if not Path(path).resolve().is_relative_to(Path.cwd().resolve()):
        raise HTTPException(status_code=404, detail="No page image")
    return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})


@app.delete("/documents/{document_id}")
def delete_uploaded_document(document_id: str, request: Request):
    """Delete an indexed upload and its generated cache entry."""
    _check_access(_user(request), document_id, write=True)
    removed = False
    for folder in ("uploaded_pdfs", "uploaded_audio"):
        for path in os.listdir(folder) if os.path.isdir(folder) else []:
            if path.startswith(f"{document_id}."):
                os.remove(os.path.join(folder, path))
                removed = True
    for folder in ("ocr_cache", "documents"):
        for path in os.listdir(folder):
            if path.startswith(f"{document_id}_") or path == f"{document_id}.json":
                os.remove(os.path.join(folder, path))
                removed = True
    if not removed:
        raise HTTPException(status_code=404, detail="Document not found")
    import qdrant_vault
    ag.clear_cache(document_id)
    ag.VISUAL_PROGRESS.pop(document_id, None)
    # Without this, a deleted document's chunks stay searchable forever.
    qdrant_vault._get_vault_singleton().delete_document(document_id)
    return {"deleted": document_id}


def _read_images_in_background(document_ids):
    """Vision-read image uploads / scanned pages so they become searchable."""
    for doc_id in document_ids:
        try:
            ag.index_visual_pages(doc_id, delay=25)
        except Exception as error:
            print(f"[!] background visual indexing failed for {doc_id}: {error}")


@app.post("/documents/upload")
async def upload_documents(request: Request, background_tasks: BackgroundTasks,
                           files: list[UploadFile] = File(...)):
    """Persist and index files selected from the Uploads panel."""
    document_ids = await _ingest_query_files(files, _user(request))
    # Returns immediately; pages with no text layer are read by the reasoner afterwards.
    background_tasks.add_task(_read_images_in_background, document_ids)
    return {"document_ids": document_ids}


def _step_payload(trace):
    """Convert the graph trace into the shape expected by the browser UI."""
    steps = []
    for name in ("ROUTER", "RETRIEVER", "REASONER", "VERIFIER", "SUMMARIZER"):
        matching = [line for line in trace if line.lower().startswith(name.lower())]
        steps.append({
            "name": name,
            "status": "done" if matching else "pending",
            "detail": matching[-1] if matching else "",
        })
    return steps


async def _ingest_query_files(files, user):
    """Persist browser uploads, normalize them, stamp the owner, and refresh the local index."""
    if not files:
        return []

    from uuid import uuid4
    from process_audio import normalize_audio_to_structured_content
    from process_backend import normalize_pdf_to_structured_content
    from process_office import normalize_office_to_structured_content
    import qdrant_vault

    document_ids = []
    for upload in files:
        if not getattr(upload, "filename", None) or not hasattr(upload, "read"):
            continue
        suffix = Path(upload.filename).suffix.lower()
        is_audio = upload.content_type.startswith("audio/") if upload.content_type else suffix in {
            ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"
        }
        is_image = suffix in {".jpg", ".jpeg", ".png", ".webp"}
        if not is_audio and not is_image and suffix != ".pdf" and suffix not in SUPPORTED_SUFFIXES:
            supported = ", ".join(sorted(SUPPORTED_SUFFIXES | {".pdf"}))
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {upload.filename}. Supported: {supported}, images and audio.")

        content = await upload.read()
        if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"{upload.filename} is larger than {MAX_UPLOAD_MB} MB.")
        source_dir = "uploaded_audio" if is_audio else "uploaded_pdfs"
        os.makedirs(source_dir, exist_ok=True)
        doc_id = f"{re.sub(r'[^a-zA-Z0-9_-]', '_', Path(upload.filename).stem)}_{uuid4().hex[:8]}"
        document_ids.append(doc_id)
        source_path = os.path.join(source_dir, f"{doc_id}{suffix}")
        with open(source_path, "wb") as output:
            output.write(content)

        if is_audio:
            structured = normalize_audio_to_structured_content(
                source_path, doc_id, filename=upload.filename
            )
        elif suffix == ".pdf":
            structured = normalize_pdf_to_structured_content(
                source_path, doc_id, filename=upload.filename
            )
        elif is_image:
            structured = {
                "doc_id": doc_id,
                "filename": upload.filename,
                "pages": {
                    "page_1": {
                        "image_path": source_path,
                        "spatial_blocks": [{"text_lines": [upload.filename]}],
                        "raw_text_stream": upload.filename,
                    }
                },
            }
            with open(f"ocr_cache/{doc_id}_structured.json", "w", encoding="utf-8") as output:
                json.dump(structured, output, ensure_ascii=False, indent=2)
        else:  # Word, Excel, PowerPoint, HTML, JSON, text, CSV ...
            structured = normalize_office_to_structured_content(
                source_path, doc_id, filename=upload.filename
            )
        if structured is None:
            os.remove(source_path)  # do not leave an unindexed file behind
            document_ids.remove(doc_id)
            raise HTTPException(
                status_code=400,
                detail=f"Could not read {upload.filename} (corrupt, password-protected or empty).")

        # Stamp the owner so retrieval can keep one user's documents away from another's.
        stored, path = ag.load_structured(doc_id)
        if stored is not None and user != "local":
            stored["owner"] = user
            with open(path, "w", encoding="utf-8") as output:
                json.dump(stored, output, ensure_ascii=False, indent=2)

    qdrant_vault._get_vault_singleton().index_structured_content("ocr_cache")
    return document_ids


# ======================================================================== asking questions
def _payload(result: dict, session_id: str, document_id):
    trace = result.get("trace", [])
    return {
        "answer": result.get("final_answer"),
        "final_answer": result.get("final_answer"),
        "confidence": result.get("confidence"),
        "verified": result.get("verified"),
        "retry_count": result.get("retry_count"),
        "model_used": result.get("model_used"),
        "reasoning_model_used": result.get("reasoning_model_used"),
        "response_model_used": result.get("response_model_used"),
        "modality": result.get("modality"),
        "query_type": result.get("query_type"),
        "document_id": result.get("document_id") or document_id,
        "evidence_sufficient": result.get("evidence_sufficient"),
        "unsupported_claims": result.get("unsupported_claims", []),
        "verification_reason": result.get("verification_reason", ""),
        "image_read_unverified": result.get("image_read_unverified", False),
        "confidence_source": result.get("confidence_source"),
        "escalated": result.get("retry_count", 0) > 0,
        "attempts": result.get("attempts", []),
        "timings": result.get("timings", {}),
        "timings_original": result.get("timings_original"),
        "citations": result.get("citations", []),
        "short_answer": result.get("short_answer"),
        "session_id": session_id,
        "cache_hit": result.get("cache_hit", False),
        "trace": trace,
        "thinking": trace,
        "steps": _step_payload(trace),
    }


async def _read_query(request: Request):
    """Return query data, including an optional policy carried into normal chat."""
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        return (str(form.get("message", "")).strip(), str(form.get("document_id", "")).strip() or None,
                str(form.get("session_id", "")).strip() or "default", str(form.get("detail", "")).strip() or None,
                form.getlist("files"), str(form.get("model", "")), str(form.get("insurance_policy_id", "")).strip() or None)
    body = await request.json()
    return (str(body.get("message", body.get("question", ""))).strip(), body.get("document_id") or None,
            str(body.get("session_id") or "default"), body.get("detail") or None, [], str(body.get("model", "")),
            str(body.get("insurance_policy_id") or "").strip() or None)


@app.post("/query")
async def query_pipeline(request: Request):
    """Answer a question (waits for the whole answer; /query/stream shows it live)."""
    user = _user(request)
    question, document_id, session_id, detail, files, model, insurance_policy_id = await _read_query(request)
    if not question and not files:
        raise HTTPException(status_code=400, detail="Message or attachment required")

    uploaded = await _ingest_query_files(files, user)
    document_id = uploaded[0] if uploaded else document_id
    if document_id:
        _check_access(user, document_id)

    question = question or "Summarize the attached source."
    session_key = (user, session_id)
    if insurance_policy_id:
        result = await run_in_threadpool(answer_for_chat, user, insurance_policy_id, question)
    else:
        result = await run_in_threadpool(
            ag.run_query, question, model=model or None, document_id=document_id,
            history=list(SESSIONS[session_key]), detail=detail, owner=_owner_filter(user))
    document_id = result.get("document_id") or document_id
    SESSIONS[session_key].append({"question": question, "answer": result.get("final_answer"),
                                  "document_id": document_id})
    return _payload(result, session_id, document_id)


@app.post("/query/stream")
async def query_stream(request: Request):
    """
    Server-sent events: one JSON object per `data:` line.
      step / progress / token events while the answer is being made, then one `final` event
      holding the same payload /query returns (or `error`).
    """
    user = _user(request)
    question, document_id, session_id, detail, _, model, insurance_policy_id = await _read_query(request)
    if not question:
        raise HTTPException(status_code=400, detail="Message required")
    if document_id:
        _check_access(user, document_id)
    session_key = (user, session_id)
    history = list(SESSIONS[session_key])
    out: "queue.Queue" = queue.Queue()

    def produce():
        try:
            if insurance_policy_id:
                result = answer_for_chat(user, insurance_policy_id, question)
                SESSIONS[session_key].append({"question": question, "answer": result.get("final_answer"),
                                              "document_id": None})
                out.put({"type": "final", **_payload(result, session_id, None)})
            else:
                for event in ag.stream_query(question, model=model or None, document_id=document_id,
                                             history=history, detail=detail, owner=_owner_filter(user)):
                    if event["type"] == "final":
                        result = event["state"]
                        doc = result.get("document_id") or document_id
                        SESSIONS[session_key].append({"question": question, "answer": result.get("final_answer"),
                                                      "document_id": doc})
                        event = {"type": "final", **_payload(result, session_id, doc)}
                    out.put(event)
        except Exception as error:
            out.put({"type": "error", "message": f"{type(error).__name__}: {error}"})
        finally:
            out.put(None)

    threading.Thread(target=produce, daemon=True).start()

    async def events():
        while True:
            try:
                item = await run_in_threadpool(out.get, True, 15)
            except queue.Empty:
                yield ": keep-alive\n\n"  # keeps proxies and browsers from closing a quiet connection
                continue
            if item is None:
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ======================================================================== voice and saved state
AUDIO_SUFFIXES = {"audio/webm": ".webm", "audio/ogg": ".ogg", "audio/mp4": ".m4a", "audio/mpeg": ".mp3",
                  "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/flac": ".flac"}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), language: str = Form("")):
    """Speech to text with the local Whisper model (the voice recorder in the UI sends audio here)."""
    import speech
    data = await audio.read()
    if len(data) > MAX_AUDIO_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Recording too long.")
    if len(data) < 1200:
        raise HTTPException(status_code=400, detail="That recording was too short to hear anything.")
    kind = (audio.content_type or "").split(";")[0].strip().lower()
    suffix = AUDIO_SUFFIXES.get(kind) or Path(audio.filename or "").suffix.lower() or ".webm"
    try:
        text, detected, seconds = await run_in_threadpool(
            speech.transcribe_bytes, data, suffix, language.strip() or None)
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"Could not read that recording ({type(error).__name__}).")
    return {"text": text, "language": detected, "seconds": round(seconds, 1)}


def _state_path(user: str) -> str:
    os.makedirs(STATE_DIR, exist_ok=True)
    return os.path.join(STATE_DIR, f"{user}.json")


@app.get("/state")
def get_state(request: Request):
    """The signed-in user's saved chats and settings (so they follow them to another browser)."""
    path = _state_path(_user(request))
    if not os.path.exists(path):
        return {"chats": [], "settings": {}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.put("/state")
async def put_state(request: Request):
    raw = await request.body()
    if len(raw) > 3_000_000:
        raise HTTPException(status_code=413, detail="Saved chats are too large; delete some conversations.")
    body = json.loads(raw)
    state = {"chats": list(body.get("chats", []))[-100:], "settings": dict(body.get("settings", {}))}
    path = _state_path(_user(request))
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(path + ".tmp", path)
    return {"saved": len(state["chats"])}


# ======================================================================== legacy mock endpoint
MAX_FILE_SIZE_MB = 25
STOPWORDS = {"what", "when", "where", "which", "does", "the", "this", "that",
             "with", "from", "have", "there", "about", "into"}


def score_block(question, block_text):
    """Fuzzy relevance score between a question and a text block, 0-1."""
    q_words = {w.lower() for w in re.findall(r"\w+", question) if len(w) > 3 and w.lower() not in STOPWORDS}
    b_words = {w.lower() for w in re.findall(r"\w+", block_text) if len(w) > 3}
    if not q_words or not b_words:
        return 0.0
    overlap = len(q_words & b_words) / len(q_words)
    fuzzy = SequenceMatcher(None, question.lower(), block_text.lower()).ratio()
    return 0.7 * overlap + 0.3 * fuzzy


@app.post("/upload_and_query")
async def handle_upload_and_query(file: UploadFile = File(...), question: str = Form(...)):
    """Early fuzzy-matching prototype (no model). Kept for the old stress tests; requires sign-in."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit.")

    doc_id = re.sub(r"[^a-zA-Z0-9_-]", "_", os.path.splitext(file.filename)[0])
    pdf_path = f"uploaded_pdfs/{doc_id}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(contents)

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open PDF: {e}")

    if len(doc) == 0:
        raise HTTPException(status_code=400, detail="PDF has no pages.")

    structured_content = {"doc_id": doc_id, "total_pages": len(doc), "pages": {}}

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1
        image_target_path = f"documents/{doc_id}_page_{page_num}.png"
        pix = page.get_pixmap(dpi=150)
        pix.save(image_target_path)

        raw_blocks = page.get_text("blocks")
        normalized_blocks = []
        for b in raw_blocks:
            if len(b) > 4 and isinstance(b[4], str) and b[4].strip():
                normalized_blocks.append({
                    "bbox": [round(b[0], 2), round(b[1], 2), round(b[2], 2), round(b[3], 2)],
                    "text_lines": [l.strip() for l in b[4].split("\n") if l.strip()],
                })

        structured_content["pages"][f"page_{page_num}"] = {
            "image_path": image_target_path,
            "spatial_blocks": normalized_blocks,
            "raw_text_stream": page.get_text("text").strip(),
        }

    cache_path = f"ocr_cache/{doc_id}_structured.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(structured_content, f, ensure_ascii=False, indent=2)

    # Score every block across every page, keep the best match instead of the first hit
    best_score = 0.0
    best_answer = None
    best_bbox = [0, 0, 0, 0]
    best_page = 1

    for p_key, p_val in structured_content["pages"].items():
        for block in p_val["spatial_blocks"]:
            block_text = " ".join(block["text_lines"])
            s = score_block(question, block_text)
            if s > best_score:
                best_score = s
                best_answer = block_text
                best_bbox = block["bbox"]
                best_page = int(p_key.split("_")[1])

    CONFIDENCE_THRESHOLD = 0.15
    found = best_score >= CONFIDENCE_THRESHOLD

    return {
        "status": "success",
        "doc_id": doc_id,
        "model_response": best_answer if found else "No sufficiently confident match found in document layout.",
        "confidence": round(best_score, 3),
        "evidence_page": best_page if found else None,
        "evidence_coordinates": best_bbox if found else None,
        "total_pages": len(doc),
        "engine_trace": [
            "Pipeline -> PDF validated and saved locally",
            "PyMuPDF -> Text and layout extracted for all pages",
            f"Mock_Reasoner -> Scored {sum(len(p['spatial_blocks']) for p in structured_content['pages'].values())} blocks across {len(doc)} page(s)",
            f"Mock_Reasoner -> Best match confidence: {round(best_score, 3)} (threshold {CONFIDENCE_THRESHOLD})",
        ],
    }


# ======================================================================== launcher
def _warm_up():
    """Load the embedding model now, so the first question does not pay ~20 s for it."""
    try:
        import qdrant_vault
        qdrant_vault._get_vault_singleton()
    except Exception as error:
        print(f"[!] warm-up skipped: {error}")


def main():
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser(description="cubic.ai offline document assistant")
    ap.add_argument("--host", default=os.environ.get("QUIETLY_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--lan", action="store_true", help="listen on all network interfaces (needs --https)")
    ap.add_argument("--https", action="store_true", help="serve HTTPS with a self-signed certificate")
    ap.add_argument("--allow-insecure-lan", action="store_true", help="LAN over plain HTTP (not recommended)")
    args = ap.parse_args()

    host = "0.0.0.0" if args.lan else args.host
    on_network = host not in ("127.0.0.1", "localhost", "::1")
    if on_network and not AUTH_REQUIRED:
        sys.exit("Refusing to listen on the network with QUIETLY_AUTH=off: anyone could read every document.")
    if on_network and not args.https and not args.allow_insecure_lan:
        sys.exit("Listening on the network needs --https (passwords and documents would travel in clear text, "
                 "and browsers block the microphone on plain http). Use --allow-insecure-lan to override.")

    ssl = {}
    if args.https:
        import make_cert
        cert, key = make_cert.make_cert()
        ssl = {"ssl_certfile": cert, "ssl_keyfile": key}
    port = args.port or (8443 if args.https else 8000)

    if AUTH_REQUIRED:
        message = auth.ensure_first_user()
        if message:
            print(f"\n*** {message}\n")
    threading.Thread(target=_warm_up, daemon=True).start()
    scheme = "https" if args.https else "http"
    print(f"cubic.ai is starting at {scheme}://{'localhost' if not on_network else host}:{port}"
          f"  (sign-in {'required' if AUTH_REQUIRED else 'OFF - this PC only'})")
    uvicorn.run(app, host=host, port=port, log_level="warning", **ssl)


if __name__ == "__main__":
    main()





