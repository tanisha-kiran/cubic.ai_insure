"""
Layer 2 — Agentic Reasoning (LangGraph)

    Router -> Retriever -> Reasoner -> Verifier --FAIL--> Reasoner (next plan step)
                                          |
                                          +--PASS / plan exhausted--> Summarizer (writer)

Two pipeline modes (PIPELINE_MODE env var, or run_query(mode=...)):

reason_then_write (default)
    Reasoner = sovereign-qwen, the QLoRA fine-tuned Qwen2.5-VL-7B. It is used
    the way it was trained: page image + bare question -> short answer, with
    the image resized to the training pixel budget (512x512).
        step 1: evidence crop (retrieved region), question only
        step 2: full page + extracted text  (only if step 1 fails verification)
    Writer = qwen2.5vl:3b. Turns the verified answer into a reply. It may not
    change the 7B's answer: if its sentence does not contain the extracted
    answer verbatim, the extracted answer is returned instead.
    Summary questions skip the 7B (it was trained for short extractive
    answers) and the writer summarizes page-ordered text from the whole document.

cascade
    Cheapest model first, escalating only on verifier FAIL:
    qwen2.5:3b on the text layer -> qwen2.5vl:3b on the crop -> 7B on the full page.

Shared machinery: retrieval-guided cropping, token-level confidence from
Ollama logprobs (refusal heuristic if unavailable), grounding / numeric-claim
checks, page+bbox citations, and an answer cache.

LangGraph note: state mutations only persist inside nodes. verifier_router
must stay pure -- mutating there caused an infinite loop in an earlier version.
"""

import offline  # noqa: F401  (models load from the local cache only; ALLOW_ONLINE=1 to download)
import base64
import glob
import hashlib
import io
import json
import math
import os
import re
import threading
import time
from collections import OrderedDict
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Optional, TypedDict

import requests
from langgraph.graph import END, StateGraph

from hardware_profile import PROFILE


OLLAMA_URL = "http://localhost:11434/api/generate"

TEXT_MODEL = "qwen2.5:3b"
VL_SMALL = "qwen2.5vl:3b"
VL_LARGE = os.environ.get("REASONER_MODEL", "sovereign-qwen:latest")
WRITER_MODEL = os.environ.get("WRITER_MODEL", VL_SMALL)
# Describes / analyzes images. The stock instruct 3B is used rather than the
# fine-tuned 7B, which was trained to emit short extractive answers only.
VISION_ANALYST_MODEL = os.environ.get("VISION_ANALYST_MODEL", VL_SMALL)
# Charts and small print need more pixels than the extraction crop; this is
# the budget for whole-image analysis by the 3B (vision tokens ~ pixels / 784).
ANALYST_MAX_PIXELS = int(os.environ.get("ANALYST_MAX_PIXELS",
                                        (1024 * 1024) if PROFILE["name"] == "gpu" else (768 * 768)))
# Answer length. The fine-tuned 7B was trained on 1-3 word answers, so it does the
# reading/extraction and the writer model composes the answer at this length.
ANSWER_DETAIL = os.environ.get("ANSWER_DETAIL", "detailed")   # "brief" | "detailed"
DETAIL_SPEC = {
    "brief":    {"words": "40 to 80",   "tokens": 320},
    "detailed": {"words": "150 to 250", "tokens": 900},
}
# Code, formulas and architecture sketches are longer than prose and must not be cut mid-block.
FORMAT_TOKENS = int(os.environ.get("FORMAT_TOKENS", "1400"))
# Context window for the writer: prompt (evidence + facts) plus the larger answer budget.
WRITER_NUM_CTX = int(os.environ.get("WRITER_NUM_CTX", "6144"))
# Pixel budget when the 7B looks at a whole photo/chart/diagram. It was trained at
# 512x512; a little more helps small text and costs CPU time (tokens ~ pixels).
SOVEREIGN_VISUAL_PIXELS = int(os.environ.get("SOVEREIGN_VISUAL_PIXELS", 640 * 640))
# A document whose whole text fits in this many characters (~1500 tokens) is given to the model in full
# instead of searched: a search can miss, and reading 3,000 characters costs far less than a failed search
# followed by an image read (measured: a one-page resume took 58 s + 304 s + 67 s that way).
SMALL_DOC_CHARS = int(os.environ.get("SMALL_DOC_CHARS", "6000"))
# Reading a page IMAGE with the 7B costs minutes on a CPU. A document that already has a text layer is not
# re-read as a picture unless this is on (default: on with a GPU, off without one).
IMAGE_FALLBACK_FOR_TEXT_DOCS = os.environ.get("IMAGE_FALLBACK_FOR_TEXT_DOCS", "1" if PROFILE["name"] == "gpu" else "0") == "1"
THIN_TEXT_CHARS = 40        # less real text than this => no text layer (image upload, scan)
TEXT_ONLY_MODELS = {TEXT_MODEL}
PIPELINE_MODE = os.environ.get("PIPELINE_MODE", "reason_then_write")
# Ablation switches for evaluations only (e.g. ABLATE=retry or ABLATE=verifier,router). Empty = the product as shipped.
#   verifier: accept the first attempt without any check     retry: still check, but never try another plan step
#   router:   treat every question as a lookup
ABLATE = {x for x in os.environ.get("ABLATE", "").replace(" ", "").split(",") if x}

# Pixel budget per model. The fine-tuned 7B was trained with
# max_pixels=512*512 -- sending it bigger pages is both off-distribution and
# the main source of CPU latency (vision tokens scale with pixels).
MODEL_MAX_PIXELS = {
    VL_LARGE: 512 * 512,
    VL_SMALL: PROFILE["max_image_side"] ** 2,
}

CONFIDENCE_THRESHOLD = 0.60   # calibrated by evaluate.py report; see runs/<tag>/report.md
GROUNDING_THRESHOLD = 0.80    # fuzzy match ratio for "answer appears in evidence"
SUMMARY_CHARS = 6000          # document text budget for summary questions (~1500 tokens)
REQUEST_TIMEOUT = PROFILE["timeout"]
NUM_THREAD = PROFILE["num_thread"]
RENDER_DPI = 150              # must match get_pixmap(dpi=...) in process_backend.py
CROP_DIR = os.path.join("documents", "crops")


def build_plan(mode: str, tiers=None) -> list:
    """Ordered reasoning attempts. Each step: model, view (text|crop|page), use_text."""
    if tiers is not None:  # explicit model list, used by evaluation ablations
        return [{"model": m, "view": "text" if m in TEXT_ONLY_MODELS else "crop", "use_text": True}
                for m in tiers]
    if mode == "cascade":
        plan = []
        if PROFILE["use_text_tier"]:
            plan.append({"model": TEXT_MODEL, "view": "text", "use_text": True})
        plan.append({"model": VL_SMALL, "view": "crop", "use_text": True})
        if PROFILE["allow_large"]:
            plan.append({"model": VL_LARGE, "view": "page", "use_text": True})
        return plan
    # Measured on a 4-core laptop CPU: the 7B spends ~150 s encoding each new
    # image but answers from text in seconds. So digital PDFs are read from
    # their text layer first (and the answer must then appear verbatim in that
    # text); the image is only encoded if that fails or there is no text.
    return [
        {"model": VL_LARGE, "view": "text", "use_text": True},       # best-matching passages, strictly grounded
        {"model": VL_LARGE, "view": "text_wide", "use_text": True},  # a wider slice of the text (seconds, not minutes)
        {"model": VL_LARGE, "view": "crop", "use_text": False},      # image, matches training format
        {"model": VL_LARGE, "view": "page", "use_text": True},       # zoom out + text layer
    ]


# =============================================================================
# State
# =============================================================================
class AgentState(TypedDict):
    owner: Optional[str]                # signed-in user; retrieval is limited to their documents
    emit: object                        # callable(event dict) for live progress/tokens, or None
    question: str
    effective_question: str             # question, expanded with the previous turn for follow-ups
    history: list                       # recent turns: {"question", "answer", "document_id"}
    document_id: Optional[str]
    mode: str                           # "reason_then_write" | "cascade"
    modality: str                       # "document" | "audio"                     (Router)
    force_query_type: Optional[str]     # evaluation hook: skip the router's classification
    query_type: str                     # extractive | advisory | summary | visual (Router)
    text_is_thin: bool                  # source has (almost) no text layer -> needs vision
    small_doc: bool                     # the whole document text is in retrieved_context
    wide_context: str                   # a wider slice of passages for the second text attempt
    visual_fallback: bool               # image-only source: hand to the image analyst if the 7B finds nothing
    fresh_look: bool                    # re-read the pixels instead of the saved description
    image_read_unverified: bool         # answer came straight from the image with nothing to check it against
    detail: str                         # "brief" | "detailed" answer length
    use_retrieval: bool
    retrieved_context: Optional[str]    # (Retriever)
    retrieved_chunks: list
    citations: list                     # page/bbox/snippet per chunk
    image_path: Optional[str]           # full page: supplied by caller or Retriever
    crop_path: Optional[str]            # evidence crop
    plan: list                          # reasoning steps, see build_plan
    tier_index: int                     # current position in plan
    escalate: bool                      # set by Verifier, read by verifier_router
    attempts: list                      # one record per Reasoner call
    candidate_answer: Optional[str]     # short answer from the reasoner
    confidence: Optional[float]
    confidence_source: Optional[str]    # "logprobs" | "heuristic"
    verified: Optional[bool]
    evidence_sufficient: Optional[bool]
    unsupported_claims: list
    verification_reason: str
    retry_count: int                    # number of escalations performed
    final_answer: Optional[str]
    short_answer: Optional[str]         # extractive span, what ANLS/EM should score
    model_used: Optional[str]
    reasoning_model_used: Optional[str]
    response_model_used: Optional[str]
    cache_hit: bool
    timings: dict                       # seconds per node
    trace: list


def _timed(name):
    """Record wall-clock seconds spent in a node (summed across loop-backs)."""
    def wrap(fn):
        def inner(state):
            start = time.perf_counter()
            out = fn(state)
            out["timings"][name] = round(out["timings"].get(name, 0.0) + time.perf_counter() - start, 3)
            return out
        inner.__name__ = fn.__name__
        return inner
    return wrap


def _emit(state, event: dict):
    """Send a live event (progress text / answer token) to the UI if someone is listening."""
    sink = state.get("emit")
    if sink:
        try:
            sink(event)
        except Exception:
            pass  # a closed stream must never break the answer


# =============================================================================
# Router
# =============================================================================
AUDIO_KEYWORDS = ("said", "mentioned", "spoke", "spoken", "transcript", "recording",
                  "audio", "heard", "voice", "call", "meeting")

SUMMARY_PATTERNS = (
    r"\bsummar(y|ise|ize|ies)\b", r"\boverview\b", r"\bkey (points|takeaways|ideas)\b",
    r"\bmain (points|ideas|topics)\b", r"\bwhat is (this|the) (document|file|pdf|paper|module|chapter) about\b",
    r"\btl;?dr\b", r"\bgist\b", r"\boutline\b",
    # "what is inside it" / "key details" want an overview, not one short span (tuning: "what are the contents
    # inside it?" returned the two words "payload data" at confidence 0.34)
    r"\bcontents?\b", r"\bwhat('s| is| are)\b.*\b(in|inside|within)\b (it|this|the|that)\b",
    r"\bwhat does (it|this|the \w+) (contain|cover|include|say)\b", r"\bkey (details|facts|info\w*)\b",
    r"\btell me (about|more about) (this|the|it)\b", r"\bgive me (a|an) (rundown|brief|breakdown)\b",
)
ADVISORY_PATTERNS = (
    r"\bshould\b", r"\brecommend", r"\badvi[cs]e", r"\bsuggest", r"\bwhy\b",
    r"\bexplain", r"\bhow (can|could|do|should) (i|we)\b", r"\bwhat do you think\b",
    r"\bpros and cons\b", r"\bcompare\b", r"\brisks?\b", r"\bimplications?\b",
    r"\bis it (worth|safe|wise)\b", r"\bnext steps?\b", r"\bdescribe\b", r"\bdifference\b",
    # Process / flow questions need a multi-sentence answer, not a 48-token span
    # (tuning: "walk me through the flow" was cut off after step 4).
    r"\bwalk (me )?through\b", r"\bstep[- ]by[- ]step\b", r"\bwhat happens\b", r"\bhow does\b",
    r"\bhow (is|are|was|were)\b.*\b(handled|processed|stored|used|connected|work\w*)\b",
    r"\b(sequence|flow|lifecycle|pipeline|workflow) of\b", r"\bin (what|which) order\b",
)


VISUAL_PATTERNS = (
    r"\b(describe|analy[sz]e|explain|interpret|read)\b.*\b(image|picture|photo|chart|graph|diagram|figure|"
    r"flowchart|plot|screenshot|infographic|map|drawing|visual)\b",
    r"\bwhat (does|do|is) (this|the|these) (image|picture|photo|chart|graph|diagram|figure|flowchart|plot|"
    r"screenshot|infographic)\b",
    r"\b(image|picture|photo|chart|graph|diagram|figure|flowchart|screenshot)\b.*\b(show|depict|represent|mean)\w*\b",
    r"\bwhat (can you )?see\b", r"\blook at (this|the)\b",
)


# Questions that ask for a particular output shape rather than a fact. They are answered by the
# writer (advisory path) with a format-specific instruction block, like a chat assistant would.
FORMAT_PATTERNS = {
    "architecture": (
        r"\barchitecture\b", r"\bskeleton\b", r"\bsystem (design|diagram|structure)\b",
        r"\bfolder structure\b", r"\bproject structure\b", r"\bblock diagram\b", r"\bcomponent diagram\b",
        r"\b(draw|sketch|map out|lay out)\b.*\b(structure|system|pipeline|flow|layout)\b",
    ),
    "formula": (
        r"\bformulas?\b", r"\bequations?\b", r"\bderivation\b", r"\bhow is .* (calculated|computed)\b",
        r"\bhow (do|to|can) (i |we |you )?(calculate|compute)\b", r"\bcalculation of\b",
    ),
    # Needs a request verb or a language name, so "what is the postal code" stays an extractive lookup.
    "code": (
        r"\b(write|show|give|generate|create|provide|draft|implement)\b.*\b(code|script|function|snippet|query|regex|program|class)\b",
        r"\b(code|script|snippet|query) (for|to|that)\b", r"\bin (python|javascript|java|c\+\+|c#|sql|bash|typescript|go|rust)\b",
        r"\b(python|javascript|typescript|sql|bash) (code|script|function|snippet|query)\b",
        r"\bcode (example|sample|snippet)\b", r"\bpseudo-?code\b", r"\bimplementation of\b",
    ),
}


def detect_format(question: str):
    """'architecture' | 'formula' | 'code' | None. Architecture wins over code ("code architecture")."""
    q = question.lower()
    for kind in ("architecture", "formula", "code"):
        if any(re.search(p, q) for p in FORMAT_PATTERNS[kind]):
            return kind
    return None


def classify_query_type(question: str) -> str:
    q = question.lower()
    if any(re.search(p, q) for p in VISUAL_PATTERNS):
        return "visual"
    if detect_format(q) and not re.search(r"\bsummar", q):   # "what is the formula in this file" is a formula request
        return "advisory"
    if any(re.search(p, q) for p in SUMMARY_PATTERNS):
        return "summary"
    if detect_format(q) or any(re.search(p, q) for p in ADVISORY_PATTERNS):
        return "advisory"
    return "extractive"


@_timed("router")
def router_node(state: AgentState) -> AgentState:
    q = state["question"].lower()
    state["modality"] = "audio" if any(re.search(rf"\b{k}\b", q) for k in AUDIO_KEYWORDS) else "document"
    state["query_type"] = (state.get("force_query_type")        # the override is for evaluations only
                           or ("extractive" if "router" in ABLATE else classify_query_type(state["question"])))
    state["trace"].append(
        f"Router -> modality='{state['modality']}', query_type='{state['query_type']}', mode='{state['mode']}'"
    )
    return state


# =============================================================================
# Retriever
# =============================================================================
def crop_to_evidence(image_path: str, chunks: list, pad_px: int = 80, min_height: int = 240):
    """
    Region around the retrieved chunks on this page. Because the 7B sees at
    most 512x512 pixels, a crop means far more pixels per character -- text
    that is illegible on a downscaled full page becomes readable -- and fewer
    pixels to encode (the dominant CPU cost). Horizontally it keeps a wide
    margin so "label: value" pairs stay together; full width when the
    evidence spans most of the page.
    Returns a crop path, or None if cropping would not save much.
    """
    boxes = [c["bbox"] for c in chunks if c.get("image_path") == image_path and c.get("bbox")]
    if not boxes or not os.path.exists(image_path):
        return None
    from PIL import Image

    scale = RENDER_DPI / 72.0  # PyMuPDF bboxes are in PDF points
    with Image.open(image_path) as img:
        width, height = img.size
        top = max(0, int(min(b[1] for b in boxes) * scale) - pad_px)
        bottom = min(height, int(max(b[3] for b in boxes) * scale) + pad_px)
        if bottom - top < min_height:
            middle = (top + bottom) // 2
            top, bottom = max(0, middle - min_height // 2), min(height, middle + min_height // 2)
        if bottom - top > 0.7 * height:
            return None
        left = max(0, int(min(b[0] for b in boxes) * scale) - 2 * pad_px)
        right = min(width, int(max(b[2] for b in boxes) * scale) + 3 * pad_px)  # values sit to the right
        if right - left > 0.75 * width:
            left, right = 0, width
        os.makedirs(CROP_DIR, exist_ok=True)
        key = hashlib.md5(f"{image_path}|{left}|{top}|{right}|{bottom}".encode()).hexdigest()[:12]
        crop_path = os.path.join(CROP_DIR, f"{key}.png")
        if not os.path.exists(crop_path):
            img.crop((left, top, right, bottom)).save(crop_path)
    return crop_path


def load_structured(document_id: str):
    """(data, path) of a document's StructuredContent JSON in ocr_cache, or (None, None)."""
    paths = (glob.glob(os.path.join("ocr_cache", f"{document_id}_*structured.json"))
             + glob.glob(os.path.join("ocr_cache", f"{document_id}.json")))
    if not paths:
        return None, None
    with open(paths[0], encoding="utf-8") as f:
        return json.load(f), paths[0]


def document_text_for_summary(document_id: str, budget: int = SUMMARY_CHARS):
    """
    Text sampled evenly across the whole document (so a summary of a 58-page
    PDF covers all of it), plus the first page image for vision fallback.
    Returns (text, first_page_image_path).
    """
    data, _ = load_structured(document_id)
    if not data:
        return "", None
    if "segments" in data:  # audio transcript
        return data.get("raw_text_stream", "")[:budget], None
    pages = sorted(data.get("pages", {}).items(), key=lambda kv: int(re.sub(r"\D", "", kv[0]) or 0))
    if not pages:
        return "", None
    per_page = max(150, budget // len(pages))
    parts = []
    for key, page in pages:
        text = " ".join((page.get("raw_text_stream") or "").split())
        if real_text_length(text) < THIN_TEXT_CHARS:
            text = ""  # image uploads store only their filename here
        if page.get("visual_description"):
            text = f"{text} [visual reading] {page['visual_description']['text']}".strip()
        if text:
            page_number = re.sub(r"\D", "", key)
            parts.append(f"(page {page_number}) {text[:per_page]}")
    return "\n".join(parts)[:budget], pages[0][1].get("image_path")


FILE_NAME_RE = re.compile(r"\S+\.(png|jpe?g|webp|pdf|csv|txt|md)\b", re.I)


def real_text_length(context: str) -> int:
    """Characters of actual content, ignoring the filename placeholders image uploads are indexed with."""
    return len(FILE_NAME_RE.sub("", context or "").strip())


def _reindex():
    try:
        from qdrant_vault import _get_vault_singleton
        _get_vault_singleton().index_structured_content("ocr_cache")
    except Exception as e:  # indexing is best-effort; the JSON is already saved
        print(f"[!] visual memory saved but not indexed: {e}")


def save_visual_memory(document_id: str, description: str, model: str, page_key: str = None,
                       reindex: bool = True) -> bool:
    """
    Persist a vision reading of a page (image upload or scanned PDF page) as
    searchable text, so later questions have text evidence (faster, checkable)
    and the page becomes findable by content instead of by filename.
    """
    data, path = load_structured(document_id)
    if not data or not data.get("pages"):
        return False
    page_key = page_key or next(iter(data["pages"]))
    page = data["pages"].get(page_key)
    if page is None or page.get("visual_description"):
        return False
    page["visual_description"] = {"text": description, "model": model}
    page.setdefault("spatial_blocks", []).append(
        {"bbox": None, "text_lines": [f"Visual analysis: {description}"]}
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    if reindex:
        _reindex()
    return True


_VISUAL_INDEX_LOCK = threading.Lock()


QUERIED_DOCS = set()   # documents a user has already asked about in this server session
VISUAL_PROGRESS = {}   # document_id -> {"state": queued|reading|done|skipped, "done": n, "total": n}

READ_PROBES = (
    ("Type", "What type of image is this? Answer with one of: bar chart, line chart, pie chart, table, "
             "flowchart, diagram, form, document, photo, screenshot.", 20),
    ("Title", "What is the title or main heading of this image? If there is none, reply: not found", 40),
    ("Visible text", "List the text, labels and numbers that are written in this image.", 160),
    ("Content", "Describe what is shown in this image.", 120),
)


def read_page_with_sovereign(image_path: str, on_progress=None) -> str:
    """The fine-tuned reader transcribes one image page (used at upload time, with no question)."""
    parts = []
    for i, (label, prompt, limit) in enumerate(READ_PROBES, 1):
        if on_progress:
            on_progress(f"{label.lower()} ({i}/{len(READ_PROBES)})")
        text, _, _, error = _probe(VL_LARGE, prompt, image_path, limit, max_pixels=SOVEREIGN_VISUAL_PIXELS)
        if error:
            raise RuntimeError(error)
        if text:
            parts.append(f"{label}: {text}")
    return "\n".join(parts)


def pages_needing_reading(document_id: str, limit: int = 3):
    data, _ = load_structured(document_id)
    out = []
    for page_key, page in (data or {}).get("pages", {}).items():
        image = page.get("image_path")
        if (image and os.path.exists(image) and not page.get("visual_description")
                and real_text_length(page.get("raw_text_stream", "")) < THIN_TEXT_CHARS):
            out.append(page_key)
    return out[:limit]


def index_visual_pages(document_id: str, max_pages: int = 3, delay: float = 0) -> int:
    """
    Background job after upload: read pages that have no text layer (image
    uploads, scanned PDFs) with the fine-tuned reader and store the reading as
    visual memory, so later questions answer from text instead of re-encoding
    the image. Capped at max_pages to bound CPU time; later pages are read on
    demand by questions.

    delay: wait this long first; if the user asks about the document in the
    meantime, skip -- the question reads the image itself, and a background
    read would otherwise hold the single CPU-bound model for minutes first.
    VISUAL_PROGRESS[document_id] tells the UI what is happening.
    """
    todo = pages_needing_reading(document_id, max_pages)
    if not todo:
        VISUAL_PROGRESS[document_id] = {"state": "done", "done": 0, "total": 0}
        return 0
    progress = VISUAL_PROGRESS[document_id] = {"state": "queued", "done": 0, "total": len(todo), "detail": ""}
    if delay:
        time.sleep(delay)
        if document_id in QUERIED_DOCS:
            progress["state"] = "skipped"
            return 0
    with _VISUAL_INDEX_LOCK:  # one background reader at a time
        if document_id in QUERIED_DOCS and delay:
            progress["state"] = "skipped"
            return 0
        data, _ = load_structured(document_id)
        if not data:
            progress["state"] = "done"
            return 0
        done = 0
        progress["state"] = "reading"
        for page_key in todo:
            image = data["pages"][page_key]["image_path"]
            try:
                started = time.perf_counter()
                reading = read_page_with_sovereign(
                    image, lambda step, n=done + 1: progress.update(detail=f"page {n} of {len(todo)}: {step}"))
            except Exception as e:
                print(f"[!] visual indexing failed for {document_id}/{page_key}: {e}")
                break
            if reading.strip() and save_visual_memory(document_id, reading.strip(), VL_LARGE,
                                                      page_key, reindex=False):
                done += 1
                progress["done"] = done
                print(f"[+] visual memory: {document_id}/{page_key} read in {time.perf_counter() - started:.0f}s")
        if done:
            _reindex()
        progress["state"] = "done"
        return done


# Phrases that mean "look at this picture" when the source is an image with no text.
THIN_VISUAL_RE = re.compile(
    r"\b(analy[sz]\w*|describe|caption|what(?:'s|\s+is|\s+are)\s+(?:in|on|inside|happening|going on|written)"
    r"|tell me (?:about|what)|look(?:s)? like|what colou?rs?|what (?:is|are) (?:this|that|these|those))\b",
    re.I)


def _finalize_query_type(state: AgentState):
    """
    Image-only sources (photos, screenshots, scanned pages) need vision.
    - analysis / description / summary / advice questions go straight to the image analyst;
    - fact questions try the fine-tuned 7B first (trained on documents and charts) and fall
      back to the image analyst if it cannot find the answer (visual_fallback).
    """
    context = state.get("retrieved_context") or ""
    # Image-based = no text layer, OR the only text is a vision reading of the image
    # (visual memory only ever exists for pages that had no text layer).
    state["text_is_thin"] = (real_text_length(context) < THIN_TEXT_CHARS
                             or "Visual analysis:" in context or "[visual reading]" in context)
    qt = state["query_type"]
    if state["text_is_thin"] and state.get("image_path") and (
            qt in ("summary", "advisory") or THIN_VISUAL_RE.search(state["question"])):
        state["query_type"] = "visual"
        state["trace"].append(f"Retriever -> source has no text layer; '{qt}' question routed to image analysis")
    elif state["text_is_thin"] and state.get("image_path") and qt == "extractive":
        state["visual_fallback"] = True
    elif qt == "visual" and not state.get("image_path"):
        state["query_type"] = "advisory"
        state["trace"].append("Retriever -> visual question but no image found; answering from text")


@_timed("retriever")
def retriever_node(state: AgentState) -> AgentState:
    if not state["use_retrieval"]:
        state["retrieved_context"] = state.get("retrieved_context") or ""
        state["trace"].append("Retriever -> skipped (image supplied directly)")
        _finalize_query_type(state)
        return state

    if state["query_type"] in ("summary", "visual") and state.get("document_id"):
        text, first_image = document_text_for_summary(state["document_id"])
        state["image_path"] = state.get("image_path") or first_image
        if text:
            state["retrieved_context"] = text
            state["citations"] = [{"document_id": state["document_id"], "page": None, "bbox": None,
                                   "snippet": "whole document (sampled across pages)"}]
            state["trace"].append(f"Retriever -> whole-document mode: {len(text)} chars sampled across pages, "
                                  f"page image={'yes' if state.get('image_path') else 'no'}")
            _finalize_query_type(state)
            return state

    from qdrant_vault import _get_vault_singleton

    vault = _get_vault_singleton()
    top_k = 8 if state["query_type"] == "summary" else 3
    # Over-fetch then dedupe: older index runs left duplicate points for the same chunk.
    results, seen = [], set()
    for r in vault.search(state["effective_question"], top_k=max(top_k * 2, 20), document_id=state.get("document_id"),
                          owner=state.get("owner")):
        key = (r.get("doc_id"), r.get("page"), str(r.get("bbox")), r.get("start"), r["text"])
        if key not in seen:
            seen.add(key)
            results.append(r)
    wide, size = [], 0
    for r in results[:10]:   # the second text attempt reads up to 10 passages (about 5,000 characters)
        if size + len(r["text"]) > 5000 and wide:
            break
        wide.append(r["text"])
        size += len(r["text"])
    state["wide_context"] = "\n".join(wide)
    results = results[:top_k]

    state["retrieved_chunks"] = results
    state["retrieved_context"] = "\n".join(r["text"] for r in results)
    if state.get("document_id") and state["query_type"] != "summary":
        whole, _ = document_text_for_summary(state["document_id"], budget=10 ** 7)
        if whole and real_text_length(whole) <= SMALL_DOC_CHARS:
            # Small document: give the model all of it. A search can miss; reading it whole cannot.
            state["retrieved_context"] = whole
            state["small_doc"] = True
            state["trace"].append(f"Retriever -> small document ({len(whole):,} characters): using the whole text, no search needed")
    state["citations"] = [
        {"document_id": r.get("doc_id"), "filename": r.get("filename"), "page": r.get("page"),
         "bbox": r.get("bbox"), "start": r.get("start"), "end": r.get("end"),
         "score": r.get("score"), "snippet": r["text"][:200]}
        for r in results
    ]
    if not state.get("document_id") and results:
        # Cross-document search: remember which document answered, so a
        # follow-up question stays on it.
        state["document_id"] = results[0].get("doc_id")

    if not state.get("image_path"):
        for r in results:
            if r.get("image_path"):
                state["image_path"] = r["image_path"]
                break
    if not state.get("image_path") and state.get("document_id"):
        # Nothing retrieved carried a page image (e.g. a scanned PDF with no text
        # layer): use the document's first page image so the vision models can read it.
        data, _ = load_structured(state["document_id"])
        for page in (data or {}).get("pages", {}).values():
            if page.get("image_path") and os.path.exists(page["image_path"]):
                state["image_path"] = page["image_path"]
                break
    if state.get("image_path") and state["query_type"] == "extractive":
        state["crop_path"] = crop_to_evidence(state["image_path"], results)

    if results:
        state["trace"].append(
            f"Retriever -> {len(results)} chunks (top score={results[0]['score']}), "
            f"page image={'yes' if state.get('image_path') else 'no'}, "
            f"evidence crop={'yes' if state.get('crop_path') else 'no'}"
        )
    else:
        state["trace"].append("Retriever -> no indexed content matched")
    _finalize_query_type(state)
    return state


# =============================================================================
# Reasoner
# =============================================================================
UNCERTAIN_PHRASES = ("i don't know", "i do not know", "not visible", "cannot determine",
                     "can't determine", "no information", "unable to", "not provided",
                     "not mentioned", "not specified", "insufficient")


def _image_to_base64(image_path: str, max_pixels: int = None) -> str:
    """Encode an image, downscaling (aspect preserved) to at most max_pixels pixels."""
    from PIL import Image

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if max_pixels and w * h > max_pixels:
            s = math.sqrt(max_pixels / (w * h))
            img = img.resize((max(28, int(w * s)), max(28, int(h * s))), Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_prompt(question: str, context: str, query_type: str, has_image: bool, model: str = None) -> str:
    # The fine-tuned 7B was trained on exactly "<image> + question" -> answer.
    # Any instruction text is off-distribution for it, so send the bare question.
    if model == VL_LARGE and has_image and not context.strip() and query_type == "extractive":
        return question
    evidence = f"Extracted text from the source:\n{context}\n\n" if context.strip() else ""
    if model == VL_LARGE and query_type != "extractive":
        # Advisory: the 7B gathers the facts, the writer does the reasoning.
        return (
            f"{evidence}List the facts visible in this document (amounts, dates, names, terms, "
            f"conditions) that are relevant to this question: {question}\n"
            "Write short facts, one per line. If nothing relevant is visible, reply exactly: not found"
        )
    if query_type == "extractive":
        return (
            f"{evidence}Question: {question}\n"
            "Answer using only information visible in the "
            f"{'image' if has_image else 'source'}. Reply with the shortest exact "
            "answer (a word, number or short phrase) copied as written. "
            "If the answer is not present, reply exactly: not found"
        )
    return (
        f"{evidence}Question: {question}\n"
        "Answer in 2-5 sentences, using only facts from the "
        f"{'image and text' if has_image else 'text'} above. Quote the figures you "
        "rely on. If the source does not contain enough information, say so plainly."
    )


def clean_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(answer|final answer)\s*[:\-]\s*", "", text, flags=re.I)
    text = text.strip().strip('"').strip()
    if text.endswith(".") and len(text.split()) <= 6:
        text = text[:-1]
    return text


# Ollama's own counters for the most recent call (tokens and nanosecond timings), used by the efficiency benchmark
LAST_STATS = {}


def call_ollama(model: str, prompt: str, image_path: Optional[str], num_predict: int,
                max_pixels: int = None, num_ctx: int = 4096, extra_options: dict = None):
    """Returns (text, confidence or None, seconds). Token/time counters of the call are left in LAST_STATS."""
    options = {"temperature": 0.0, "num_predict": num_predict, "num_ctx": num_ctx, **(extra_options or {})}
    if NUM_THREAD:
        options["num_thread"] = NUM_THREAD
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "logprobs": True,
        "keep_alive": "15m",
        "options": options,
    }
    if image_path:
        budget = max_pixels or MODEL_MAX_PIXELS.get(model, PROFILE["max_image_side"] ** 2)
        payload["images"] = [_image_to_base64(image_path, budget)]

    start = time.perf_counter()
    response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    body = response.json()
    seconds = time.perf_counter() - start
    LAST_STATS.clear()
    LAST_STATS.update({k: body.get(k) for k in ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration",
                                               "eval_count", "eval_duration", "done_reason")})

    confidence = None
    logprobs = body.get("logprobs")
    if logprobs:
        values = [lp["logprob"] for lp in logprobs if isinstance(lp, dict) and "logprob" in lp]
        if values:
            confidence = math.exp(sum(values) / len(values))
    return body.get("response", ""), confidence, seconds


def repetition_start(text: str, min_len: int = 12):
    """
    Index where a run of already-seen lines begins (the model has started looping), or None. Greedy decoding
    on long structured output can repeat a block until the token limit (tuning: an architecture sketch
    repeated "[1 User Input]" and "[100 Content]" blocks until stopped). Two or more consecutive long lines
    that already appeared earlier count as a loop; one repeated line does not (trees repeat short leaves).
    """
    seen, pos, run_start, run = set(), 0, None, 0
    for line in text.split("\n"):
        key = line.strip()
        if len(key) >= min_len and key in seen:
            run += 1
            run_start = pos if run == 1 else run_start
            if run >= 2:
                return run_start
        else:
            run, run_start = 0, None
            if len(key) >= min_len:
                seen.add(key)
        pos += len(line) + 1
    return None


def call_ollama_stream(model: str, prompt: str, image_path: Optional[str], num_predict: int, on_token,
                       max_pixels: int = None, num_ctx: int = 4096, extra_options: dict = None):
    """Like call_ollama, but delivers text to on_token() as it is generated. Returns (text, seconds).
    Generation is stopped early if the model starts repeating itself (see repetition_start)."""
    options = {"temperature": 0.0, "num_predict": num_predict, "num_ctx": num_ctx, **(extra_options or {})}
    if NUM_THREAD:
        options["num_thread"] = NUM_THREAD
    payload = {"model": model, "prompt": prompt, "stream": True, "keep_alive": "15m", "options": options}
    if image_path:
        budget = max_pixels or MODEL_MAX_PIXELS.get(model, PROFILE["max_image_side"] ** 2)
        payload["images"] = [_image_to_base64(image_path, budget)]

    start, pieces = time.perf_counter(), []
    with requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT, stream=True) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            piece = chunk.get("response", "")
            if piece:
                pieces.append(piece)
                on_token(piece)
                if "\n" in piece and repetition_start("".join(pieces)) is not None:
                    LAST_STATS.clear()
                    LAST_STATS.update({"done_reason": "repetition"})
                    break
            if chunk.get("done"):
                LAST_STATS.clear()
                LAST_STATS.update({"done_reason": chunk.get("done_reason"), "eval_count": chunk.get("eval_count")})
                break
    return "".join(pieces), time.perf_counter() - start


def is_uncertain_answer(answer: str) -> bool:
    lowered = answer.lower()
    return (not answer) or lowered == "not found" or any(p in lowered for p in UNCERTAIN_PHRASES)


def _resolve_step(state: AgentState, index: int):
    """Concrete inputs for a plan step, or None if the step cannot run / would repeat an attempt."""
    step = state["plan"][index]
    context = state.get("retrieved_context") or ""
    has_text = real_text_length(context) > 20  # filenames of image uploads do not count
    if step["view"] in ("text", "text_wide") or step["model"] in TEXT_ONLY_MODELS:
        if not has_text:
            return None
        if step["view"] == "text_wide":
            context = state.get("wide_context") or context
        image_path, view = None, "text only" if step["view"] == "text" else "text (wider)"
    elif step["view"] == "crop" and state.get("crop_path"):
        image_path, view = state["crop_path"], "evidence crop"
    elif state.get("image_path"):
        image_path, view = state["image_path"], "full page"
    else:
        image_path, view = None, "text only"
    use_text = step["use_text"] and has_text
    if image_path is None and not use_text:
        return None  # no image and no text: the model would be asked with no evidence at all
    if image_path and has_text and not state.get("text_is_thin") and not IMAGE_FALLBACK_FOR_TEXT_DOCS:
        return None  # a document with a real text layer is not re-read as a picture on a CPU (minutes per page)
    signature = (step["model"], image_path, use_text, hash(context) if use_text else 0)
    if any(a.get("signature") == list(signature) for a in state["attempts"]):
        return None
    if image_path and state.get("text_is_thin") and any(
            a.get("signature") and a["signature"][0] == step["model"] and a["signature"][1] == image_path
            for a in state["attempts"]):
        return None  # same model, same image: adding text would not change what it sees
    return {"model": step["model"], "image_path": image_path, "view": view,
            "context": context if use_text else "", "signature": list(signature)}


def _next_runnable(state: AgentState, start: int):
    for i in range(start, len(state["plan"])):
        if _resolve_step(state, i):
            return i
    return None


READING_RULES = (
    "For a chart: give the title, axis labels and units, and the values of the main series.\n"
    "For a table: give the column headers and the rows that matter.\n"
    "For a diagram or flowchart: name the parts and how they connect, in order.\n"
    "Only state what is visible. If text is unreadable, say so instead of guessing."
)


def build_visual_prompt(question: str, context: str, from_memory: bool = False) -> str:
    """
    Tuning notes (qwen2.5vl:3b on real charts/diagrams):
    - Type goes first: placed after a long answer, a bar chart was called a "Diagram".
    - No examples: the 3B copied a sample bullet verbatim into an unrelated answer.
    - Values, not rankings: it read chart values correctly but invented comparisons.
    - Short prompts: with a long format spec it silently dropped whole sections.
    """
    if from_memory:
        source = f"A reading of the image (its text and a description):\n{context}\n\n"
    elif real_text_length(context) >= THIN_TEXT_CHARS:
        source = f"Text already extracted from this source:\n{context}\n\n"
    else:
        source = ""
    return (
        f"{source}Question: {question}\n"
        "Line 1: \"Type:\" then the kind of image (bar chart, line chart, pie chart, table, flowchart, "
        "diagram, form, document, photo or screenshot).\n"
        "Then answer the question in 2-3 sentences.\n"
        "Then list 2-4 values or labels you can read, one per line, each starting with \"- \".\n"
        "Report only what is printed or plotted; do not rank or compare items unless asked."
    )


INDEX_PROMPT = (
    "Read this image for a search index.\n"
    "First transcribe all readable text in reading order. Then, in 2-3 sentences, say what kind of "
    "image it is and what it shows (structure, key values, trends).\n"
    f"{READING_RULES}"
)


SUPERLATIVE_RE = re.compile(
    r"\b(highest|lowest|most|least|largest|smallest|biggest|best|worst|top|second|third|"
    r"more than|less than|greater|fewer|outperform\w*|exceed\w*)\b", re.I)


def tidy_visual_answer(text: str, question: str = "", max_bullets: int = 4,
                       max_bullet_words: int = 30) -> str:
    """
    Enforce the crisp format the 3B does not always follow. Comparison
    bullets are dropped unless the question asks for a comparison: in tuning
    the 3B read chart values correctly but stated wrong rankings ("X has the
    highest ...") despite being told not to, while the fine-tuned 7B (which
    handles comparison questions via the extractive path) got them right.
    """
    allow_comparisons = bool(SUPERLATIVE_RE.search(question)) or bool(
        re.search(r"\b(compare|comparison|rank|which)\b", question, re.I))
    lines, bullets, seen = [], 0, set()
    for line in text.strip().splitlines():
        stripped = line.strip().replace("**", "")
        if not stripped or stripped.lower() in seen:  # the 3B sometimes repeats a line
            continue
        seen.add(stripped.lower())
        if stripped.lower().startswith("answer:"):
            sentences = re.split(r"(?<=[.!?])\s+", stripped)
            stripped = " ".join(sentences[:3])
        elif stripped[:1] in "-*•":
            if not allow_comparisons and SUPERLATIVE_RE.search(stripped):
                continue
            bullets += 1
            if bullets > max_bullets:
                continue
            words = stripped.lstrip("-*• ").split()
            if len(words) > max_bullet_words:
                stripped = "- " + " ".join(words[:max_bullet_words]) + " …"
            else:
                stripped = "- " + " ".join(words)
        lines.append(stripped)
    # Drop a final line cut off mid-sentence by the token limit.
    if len(lines) > 1 and lines[-1][-1:] not in ".!?)\"'…%" and len(lines[-1].split()) > 4:
        lines.pop()
    return "\n".join(lines)


NO_ANSWER_RE = re.compile(r"^\W*(not found|none|n/a|no\b.*|nothing|unknown|unclear|not visible)\W*$", re.I)


def _probe(model: str, prompt: str, image_path, num_predict: int, max_pixels=None):
    """One short question to the reasoner. Returns (text or '', confidence or None, seconds, error)."""
    try:
        raw, confidence, seconds = call_ollama(model, prompt, image_path, num_predict, max_pixels=max_pixels)
    except requests.exceptions.ConnectionError:
        return "", None, 0.0, "Ollama not running"
    except Exception as e:
        return "", None, 0.0, str(e)
    text = clean_answer(raw)
    return ("" if NO_ANSWER_RE.match(text) else text), confidence, seconds, None


def _record_attempt(state, model, view, answer, confidences, seconds, error, signature):
    confs = [c for c in confidences if c is not None]
    uncertain = not answer
    state["confidence_source"] = "logprobs" if confs else "heuristic"
    state["confidence"] = round(sum(confs) / len(confs) if confs else (0.1 if uncertain else 0.8), 4)
    state["candidate_answer"] = answer
    state["reasoning_model_used"] = model
    state["attempts"].append({
        "model": model, "view": view, "with_text": view == "visual memory", "answer": answer,
        "confidence": state["confidence"], "uncertain": uncertain, "seconds": round(seconds, 3),
        "error": error, "signature": signature,
    })


def _supported_by(answer: str, other_text: str) -> bool:
    """Is this answer backed by the other observations? Numbers must appear; words must mostly overlap."""
    others = _normalize(other_text)
    numbers = NUMBER_RE.findall(answer)
    if numbers:
        known = {_number_core(n) for n in NUMBER_RE.findall(other_text)}
        return all(_number_core(n) in known for n in numbers)
    words = {w for w in re.findall(r"[a-z0-9]+", answer.lower()) if len(w) > 3}
    if not words:
        return True
    return len(words & set(others.split())) / len(words) >= 0.5


def sovereign_read_image(state: AgentState):
    """
    The fine-tuned 7B reads the image by answering several SHORT questions --
    the format it was trained on -- instead of one open-ended "analyze this"
    request, which it answers in a few words. The first call encodes the image
    (slow on CPU); the rest reuse Ollama's cached encoding of the same image.
    The writer later composes the long answer from these observations.
    Returns True if it produced at least one usable observation.
    """
    model, image_path = VL_LARGE, state["image_path"]
    context = state.get("retrieved_context") or ""
    has_memory = "[visual reading]" in context or "Visual analysis:" in context
    look_again = bool(re.search(r"\b(look again|re-?read|zoom|closer look|more detail)\b",
                                state["question"], re.I))
    from_memory = has_memory and not look_again and not state.get("fresh_look")
    question = state["effective_question"]

    if from_memory and detect_format(question):
        # "Draw the architecture" / "give the formula" needs the whole saved reading, not a filtered list of
        # facts (tuning: the filter returned nothing, which sent the question to a 281 s image re-read).
        _record_attempt(state, model, "visual memory", context.strip(), [0.8], 0.0, None, [model, image_path, False])
        state["trace"].append("Reasoner (saved visual memory used whole for a structured answer, no model call)")
        return True

    if from_memory:
        # Already read: ask the reasoner which saved observations bear on the question (text only, fast).
        facts, conf, secs, error = _probe(
            model, f"Reading of an image:\n{context}\n\nList the facts from this reading that are relevant to: "
                   f"{question}\nOne fact per line. If none are relevant, reply exactly: not found",
            None, 200)
        _record_attempt(state, model, "visual memory", facts, [conf], secs, error, [model, image_path, False])
        state["trace"].append(
            f"Reasoner (sovereign, saved visual memory, no re-encoding) -> {len(facts)} chars, {secs:.1f}s"
            if not error else f"Reasoner (sovereign, saved visual memory) -> FAILED: {error}")
        return bool(facts) or bool(error)

    probes = [
        ("Type", "What type of image is this? Answer with one of: bar chart, line chart, pie chart, table, "
                 "flowchart, diagram, form, document, photo, screenshot.", 20),
        ("Title", "What is the title or main heading of this image? If there is none, reply: not found", 40),
        ("Visible text", "List the text, labels and numbers that are written in this image.", 160),
        ("Content", "Describe what is shown in this image.", 120),
        ("Answer to the question", question, 120),
    ]
    observations, confs, total, error = [], [], 0.0, None
    if detect_format(question):
        probes = probes[:-1]  # the reader cannot draw a diagram in 120 tokens; the writer does that from the reading
    for i, (label, prompt, limit) in enumerate(probes, 1):
        _emit(state, {"type": "progress", "text": f"Reading the image: question {i} of {len(probes)} ({label.lower()})"})
        text, conf, secs, error = _probe(model, prompt, image_path, limit, max_pixels=SOVEREIGN_VISUAL_PIXELS)
        total += secs
        if error:
            break
        if label == "Answer to the question" and text and not _supported_by(
                text, " ".join(o[1] for o in observations)):
            # The 7B answers every question, even ones the image cannot answer. Keep its answer only
            # if the rest of what it read from the image backs it up.
            state["trace"].append(f"Reasoner -> dropped unsupported answer '{text[:60]}' "
                                  "(nothing else it read from the image backs it up)")
            continue
        if text and not any(text.lower() == o[1].lower() for o in observations):
            observations.append((label, text))
            confs.append(conf)
    facts = "\n".join(f"{label}: {text}" for label, text in observations)
    _record_attempt(state, model, "full page", facts, confs, total, error, [model, image_path, False])
    state["trace"].append(
        f"Reasoner (sovereign read the image: {len(observations)}/{len(probes)} observations, {total:.1f}s)"
        if not error else f"Reasoner (sovereign, image) -> FAILED: {error}")
    return bool(facts) or bool(error)


def sovereign_read_text_groups(state: AgentState, max_groups: int = 4, group_chars: int = 1500) -> None:
    """Summary path: the 7B extracts the key facts from each slice of the document; the writer composes."""
    context = state.get("retrieved_context") or ""
    lines, groups, current = context.split("\n"), [], ""
    for line in lines:
        if current and len(current) + len(line) > group_chars:
            groups.append(current)
            current = ""
        current += line + "\n"
    if current.strip():
        groups.append(current)
    if len(groups) > max_groups:  # keep the spread across the document
        step = len(groups) / max_groups
        groups = [groups[int(i * step)] for i in range(max_groups)]

    facts, confs, total, error = [], [], 0.0, None
    for i, group in enumerate(groups, 1):
        _emit(state, {"type": "progress", "text": f"Reading the document: part {i} of {len(groups)}"})
        text, conf, secs, error = _probe(
            VL_LARGE, f"Excerpts from a document ('(page N)' marks the page):\n{group}\n\n"
                      "List the key facts, names and figures in these excerpts, one per line, "
                      "keeping the page number in brackets.", None, 200)
        total += secs
        if error:
            break
        if text:
            facts.append(text)
            confs.append(conf)
    _record_attempt(state, VL_LARGE, "text only", "\n".join(facts), confs, total, error,
                    [VL_LARGE, None, True])
    state["trace"].append(
        f"Reasoner (sovereign extracted key facts from {len(groups)} slice(s) of the document, {total:.1f}s)"
        if not error else f"Reasoner (sovereign, summary) -> FAILED: {error}")


def analyze_image_with_analyst(state: AgentState) -> AgentState:
    """
    Backup visual path: the stock instruct 3B analyzes the whole image. Used only when the
    fine-tuned reasoner produced nothing usable.

    On a laptop CPU, encoding a new image takes ~2 minutes, while a text
    answer takes seconds. If the image was already read (visual memory, saved
    at upload or by an earlier question), the answer comes from that reading
    without re-encoding. Asking to "look again" forces a fresh look.
    """
    model, image_path = VISION_ANALYST_MODEL, state["image_path"]
    context = state.get("retrieved_context") or ""
    has_memory = "[visual reading]" in context or "Visual analysis:" in context
    look_again = bool(re.search(r"\b(look again|re-?read|zoom|closer look|more detail)\b",
                                state["question"], re.I))
    from_memory = has_memory and not look_again and not state.get("fresh_look")
    try:
        raw, confidence, seconds = call_ollama(
            model, build_visual_prompt(state["effective_question"], context, from_memory=from_memory),
            None if from_memory else image_path, 300, max_pixels=ANALYST_MAX_PIXELS)
        error = None
    except requests.exceptions.ConnectionError:
        raw, confidence, seconds, error = "", None, 0.0, "Ollama not running"
    except Exception as e:
        raw, confidence, seconds, error = "", None, 0.0, str(e)
    answer = tidy_visual_answer(raw, state["question"])
    uncertain = not answer
    state["confidence_source"] = "logprobs" if confidence is not None else "heuristic"
    state["confidence"] = round(confidence if confidence is not None else (0.1 if uncertain else 0.8), 4)
    state["candidate_answer"] = answer
    state["reasoning_model_used"] = model
    state["attempts"].append({
        "model": model, "view": "visual memory" if from_memory else "full page", "with_text": from_memory, "answer": answer,
        "confidence": state["confidence"], "uncertain": uncertain, "seconds": round(seconds, 3),
        "error": error, "signature": [model, image_path, False],
    })
    state["trace"].append(
        f"Reasoner (image analysis: {model}, {'saved visual memory, no re-encoding' if from_memory else 'full image'}) -> "
        + (f"FAILED: {error}" if error else f"{len(answer)} chars, {seconds:.1f}s")
    )
    return state


def analyze_image(state: AgentState) -> AgentState:
    """The fine-tuned reasoner reads the image; the stock 3B analyst is the backup."""
    if state["mode"] == "reason_then_write" and sovereign_read_image(state):
        return state
    if state["mode"] == "reason_then_write":
        state["trace"].append("Reasoner -> sovereign found nothing usable; backup image analyst takes over")
        state["attempts"].pop()  # replace the empty attempt
    state["fresh_look"] = True  # the saved description (if any) evidently lacked the answer
    return analyze_image_with_analyst(state)


@_timed("reasoner")
def reasoner_node(state: AgentState) -> AgentState:
    if state["query_type"] == "visual":
        return analyze_image(state)
    if state["query_type"] == "summary" and state["mode"] == "reason_then_write":
        if real_text_length(state.get("retrieved_context") or "") < 20:
            state["trace"].append("Reasoner -> nothing to summarize")
            return state
        sovereign_read_text_groups(state)
        return state

    index = _next_runnable(state, state["tier_index"])
    if index is None:  # nothing applicable (e.g. text-only plan, no text): run the step as-is
        index = state["tier_index"]
    for skipped in state["plan"][state["tier_index"]:index]:
        state["trace"].append(f"Reasoner -> skipping {skipped['model']} ({skipped['view']} not applicable)")
    state["tier_index"] = index
    step = _resolve_step(state, index) or {
        "model": state["plan"][index]["model"], "image_path": state.get("image_path"),
        "view": "full page" if state.get("image_path") else "text only",
        "context": state.get("retrieved_context") or "", "signature": None}

    model = step["model"]
    prompt = build_prompt(state["effective_question"], step["context"], state["query_type"],
                          has_image=bool(step["image_path"]), model=model)
    num_predict = 160 if state["query_type"] == "extractive" else (280 if model == VL_LARGE else 480)
    label = f"step {index + 1}/{len(state['plan'])}"

    try:
        raw, confidence, seconds = call_ollama(model, prompt, step["image_path"], num_predict)
        error = None
    except requests.exceptions.ConnectionError:
        raw, confidence, seconds, error = "", 0.0, 0.0, "Ollama not running"
    except Exception as e:  # timeouts, missing model, bad image
        raw, confidence, seconds, error = "", 0.0, 0.0, str(e)

    answer = clean_answer(raw)
    uncertain = is_uncertain_answer(answer)
    if confidence is None:
        state["confidence_source"] = "heuristic"
        confidence = 0.1 if uncertain else 0.8
    else:
        state["confidence_source"] = "logprobs"
        if uncertain:
            confidence = min(confidence, 0.1)

    state["candidate_answer"] = answer
    state["confidence"] = round(confidence, 4)
    state["reasoning_model_used"] = model
    state["attempts"].append({
        "model": model, "view": step["view"], "with_text": bool(step["context"]), "answer": answer,
        "confidence": state["confidence"], "uncertain": uncertain, "seconds": round(seconds, 3),
        "error": error, "signature": step["signature"],
    })
    if error:
        state["trace"].append(f"Reasoner ({label}: {model}, {step['view']}) -> FAILED: {error}")
    else:
        state["trace"].append(
            f"Reasoner ({label}: {model}, {step['view']}{' + text' if step['context'] else ''}) -> "
            f"answer='{answer}' confidence={state['confidence']} ({state['confidence_source']}), {seconds:.1f}s"
        )
    return state


# =============================================================================
# Verifier
# =============================================================================
NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*%?")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w%.$ ]", " ", text.lower())).strip()


def _number_core(token: str) -> str:
    return token.replace(",", "").rstrip("%").rstrip(".")


def answer_grounded(answer: str, context: str) -> bool:
    """True if the answer string appears (approximately) in the evidence text."""
    a, c = _normalize(answer), _normalize(context)
    if not a or not c:
        return False
    if a in c:
        return True
    if len(c) > 30000 or len(a) > 600:
        return False  # fuzzy matching is quadratic; for big texts only an exact match counts
    window = len(a)
    best = max(
        (SequenceMatcher(None, a, c[i:i + window]).ratio() for i in range(0, max(1, len(c) - window + 1))),
        default=0.0,
    )
    return best >= GROUNDING_THRESHOLD


PAGE_CITE_RE = re.compile(r"\(?\b(?:pp?|pages?)\.?\s*\d+(?:\s*[-,]\s*\d+)*\)?", re.I)
LIST_MARK_RE = re.compile(r"(?m)^\s*\d+[.)]\s+")


def claimed_numbers_missing(reply: str, source: str) -> list:
    """Figures in a WRITTEN answer that the source does not contain. Page citations "(p. 2)" and list numbers
    "1." are the writer's own formatting, not claims about the document, so they are ignored."""
    cleaned = LIST_MARK_RE.sub("", PAGE_CITE_RE.sub(" ", reply))
    return unsupported_numbers(cleaned, source)


def unsupported_numbers(answer: str, context: str) -> list:
    context_numbers = {_number_core(n) for n in NUMBER_RE.findall(context)}
    return [n for n in NUMBER_RE.findall(answer) if _number_core(n) not in context_numbers]


@_timed("verifier")
def verifier_node(state: AgentState) -> AgentState:
    context = state.get("retrieved_context") or ""
    has_text_evidence = real_text_length(context) > 20  # filenames of image uploads do not count
    state["evidence_sufficient"] = has_text_evidence or bool(state.get("image_path"))

    if not state["attempts"]:  # summary in reason_then_write mode: nothing to verify yet
        state["verified"], state["escalate"] = has_text_evidence, False
        state["verification_reason"] = ("document text available for summary" if has_text_evidence
                                        else "no document text to summarize")
        state["trace"].append(f"Verifier -> {state['verification_reason']}")
        return state

    if "verifier" in ABLATE:      # ablation: the first attempt is accepted unchecked
        state["verified"], state["escalate"] = True, False
        state["verification_reason"] = "verifier disabled (ablation)"
        state["trace"].append("Verifier -> disabled (ablation)")
        return state

    answer = state["candidate_answer"] or ""
    confidence = state["confidence"] or 0.0
    attempt = state["attempts"][-1]
    saw_image = attempt["view"] in ("evidence crop", "full page")

    unsupported, reasons = [], []
    if attempt["error"]:
        reasons.append(f"model error: {attempt['error']}")
    if attempt["uncertain"]:
        reasons.append("model reported the answer is not in the source")
    # Mean token probability is only comparable across short answers; long
    # answers (fact lists, image analyses) naturally score lower.
    if state["query_type"] == "extractive" and confidence < CONFIDENCE_THRESHOLD:
        reasons.append(f"confidence {confidence} < {CONFIDENCE_THRESHOLD}")

    if state["query_type"] == "visual":
        pass  # the analysis IS the reading of the image; there is no text layer to check it against
    elif (state["query_type"] == "extractive" and saw_image and not has_text_evidence
          and answer and not attempt["uncertain"]):
        # First look at an image with no text layer: there is nothing to check the answer against.
        state["image_read_unverified"] = True
        state["trace"].append("Verifier -> note: answer read straight from the image, nothing to cross-check it against")
    elif (state["query_type"] in ("summary", "advisory") and attempt["model"] == VL_LARGE
          and has_text_evidence and answer and not attempt["uncertain"]):
        # The reasoner's output here is a LIST of facts, not one quoted span, so it cannot be
        # matched verbatim. Flag figures that are not in the evidence; the writer re-checks them.
        unsupported = unsupported_numbers(answer, context)
        if unsupported:
            state["trace"].append(f"Verifier -> note: figures not in the text layer: {', '.join(unsupported)}")
    elif has_text_evidence and answer and not attempt["uncertain"]:
        if state["query_type"] == "extractive":
            # A vision model may read text the PDF text layer lacks, so only a
            # text-only answer is required to appear verbatim in the evidence.
            # Check against the WHOLE document's text, not only the retrieved passages: an answer found in
            # another part of the PDF's own text is not "read from an image" (tuning trace: a resume answer
            # was wrongly flagged because the 3 retrieved passages happened not to contain it).
            ground = context
            if state.get("document_id") and not state.get("text_is_thin"):
                whole, _ = document_text_for_summary(state["document_id"], budget=10 ** 7)
                ground = context + "\n" + whole
            if not answer_grounded(answer, ground):
                # Once the image has been transcribed (visual memory), an answer that is NOT in that
                # transcription is not trusted: in tuning, the 7B -- fine-tuned only on questions that
                # always have answers -- invented "port 8000" for a diagram that shows no port, and
                # re-reading the image returned the same invention.
                has_memory = "Visual analysis:" in context or "[visual reading]" in context
                if saw_image and not has_memory:
                    state["image_read_unverified"] = True
                    state["trace"].append("Verifier -> note: answer read from the image, not cross-checkable")
                else:
                    unsupported.append(answer)
                    reasons.append("answer not found in retrieved evidence")
        else:
            unsupported = unsupported_numbers(answer, context)
            if unsupported:
                reasons.append(f"numbers not in evidence: {', '.join(unsupported)}")

    passed = not reasons
    # Visual, summary and advisory reading is not retried with other plan steps (those are
    # extractive, and image retries cost minutes); the writer says what the evidence lacks.
    next_index = (None if passed or "retry" in ABLATE or state["query_type"] in ("visual", "summary", "advisory")
                  else _next_runnable(state, state["tier_index"] + 1))
    if not passed and next_index is None and state.get("visual_fallback") and "retry" not in ABLATE:
        # The document model could not find the answer in an image-only source:
        # let the general image analyst look at it before giving up.
        state["visual_fallback"] = False
        state["fresh_look"] = True  # the saved description evidently lacked the detail
        state["query_type"] = "visual"
        state["verified"], state["escalate"] = False, True
        state["retry_count"] += 1
        state["verification_reason"] = "; ".join(reasons)
        state["trace"].append(f"Verifier -> FAIL ({state['verification_reason']}) -> "
                              f"handing the image to the image analyst ({VISION_ANALYST_MODEL})")
        return state
    state["verified"] = passed
    state["escalate"] = next_index is not None
    state["unsupported_claims"] = unsupported
    state["verification_reason"] = "passed all checks" if passed else "; ".join(reasons)

    if state["escalate"]:
        state["tier_index"] = next_index
        state["retry_count"] += 1
        nxt = state["plan"][next_index]
        state["trace"].append(
            f"Verifier -> FAIL ({state['verification_reason']}) -> retrying with "
            f"{nxt['model']} ({nxt['view']}{' + text' if nxt['use_text'] else ''})"
        )
    else:
        state["trace"].append(
            f"Verifier -> {'PASS' if passed else 'FAIL, no further steps'} ({state['verification_reason']})"
        )
    return state


def verifier_router(state: AgentState) -> str:
    """Pure routing decision -- no state mutation (see module docstring)."""
    return "reasoner" if state["escalate"] else "summarizer"


# =============================================================================
# Summarizer (the writer)
# =============================================================================
NOT_FOUND_MESSAGE = "The source does not appear to contain this information."


def _page_label(page_key) -> str:
    return f"page {re.sub(r'[^0-9]', '', str(page_key))}" if page_key else ""


def _citation_hint(state: AgentState) -> str:
    pages = [c["page"] for c in state.get("citations", []) if c.get("page")]
    return f" (source: {_page_label(pages[0])})" if pages and _page_count(state) > 1 else ""


def _labeled_evidence(state: AgentState) -> str:
    """Retrieved chunks tagged with their page, so the writer can cite pages."""
    chunks = state.get("retrieved_chunks") or []
    if not chunks or state.get("small_doc"):
        return state.get("retrieved_context") or ""
    return "\n".join(f"[{_page_label(c.get('page')) or c.get('filename') or 'source'}] {c['text']}"
                     for c in chunks)


def _history_block(state: AgentState) -> str:
    turns = state.get("history") or []
    if not turns:
        return ""
    lines = [f"User: {t['question']}\nAssistant: {str(t.get('answer', ''))[:400]}" for t in turns]
    return "Conversation so far:\n" + "\n".join(lines) + "\n\n"


WRITER_OPTIONS = {"repeat_penalty": 1.1, "repeat_last_n": 256}
SKETCH_OPTIONS = {"repeat_penalty": 1.2, "repeat_last_n": 512}   # diagrams repeat short lines by nature; loops need a firmer push


def _write(prompt: str, num_predict: int, image_path: str = None, model: str = None, state=None, extra_options=None):
    """The writer call. When a live UI is attached (state["emit"]), the answer streams token by token."""
    model = model or WRITER_MODEL
    if model in TEXT_ONLY_MODELS:
        image_path = None
    if state is not None and state.get("emit"):
        raw, seconds = call_ollama_stream(model, prompt, image_path, num_predict,
                                          lambda t: _emit(state, {"type": "token", "text": t}),
                                          num_ctx=WRITER_NUM_CTX, extra_options=extra_options or WRITER_OPTIONS)
        return raw.strip(), seconds
    raw, _, seconds = call_ollama(model, prompt, image_path, num_predict, num_ctx=WRITER_NUM_CTX,
                                  extra_options=extra_options or WRITER_OPTIONS)
    return raw.strip(), seconds


KIND_INSTRUCTIONS = {
    "visual": ("Describe the image for someone who cannot see it: what kind of image it is, its title, "
               "the main elements and how they relate to each other, and the specific text and numbers "
               "that are visible. Then answer the question."),
    "summary": ("Summarize the whole document: what it is, then its main points in the order they "
                "appear, with the key names and figures."),
    "advisory": ("Give your recommendation or explanation, and the reasons for it that the evidence "
                 "supports."),
    "extractive": ("Give the answer, then the surrounding context from the evidence that makes it "
                   "meaningful."),
}


FORMAT_INSTRUCTIONS = {
    "code": (
        "The user wants code. Reply like a careful engineer:\n"
        "1. One sentence saying what the code does.\n"
        "2. The code in ONE fenced block with the language after the opening fence, e.g. ```python. "
        "Complete and runnable, with short comments; never leave a block unfinished.\n"
        "3. Then 'How it works' as 2-5 bullets and, if useful, one usage example in its own fenced block.\n"
        "Take names, fields, values and rules from the material above. Standard language syntax and library "
        "calls may come from general knowledge; if the material does not give a detail the code needs, use a "
        "clearly named placeholder and say so."
    ),
    "formula": (
        "The user wants a formula. Reply like a good textbook:\n"
        "1. One sentence saying what the formula computes.\n"
        "2. The formula in ONE fenced block tagged ```formula, written in plain text with Unicode "
        "(e.g. Precision = TP / (TP + FP), σ = √(Σ(x − μ)² / N)); no LaTeX commands.\n"
        "3. 'Where' followed by one bullet per symbol: **symbol**: its meaning.\n"
        "4. A short worked example with numbers only if the material gives numbers; otherwise skip it.\n"
        "5. 'What it means' in 1-2 sentences of plain-language interpretation.\n"
        "Use the formula as the material states it. If the material does not contain it, say so first, then "
        "you may give the standard textbook form labelled '(general knowledge, not from the document)'."
    ),
    "architecture": (
        "The user wants an architecture / skeleton. Reply like a software architect:\n"
        "1. One sentence naming the system and its purpose. Do not add a 'System Name' line unless a name is "
        "written in the material.\n"
        "2. A fenced block tagged ```architecture. Draw the pipeline stages (inputs, processing, outputs) side by "
        "side at the TOP level in flow order, joined by arrows, e.g. [Inputs] --> [Processing] --> [Outputs]. "
        "Then, under each stage, use an ASCII tree (├── └── │) only for the parts inside that stage. "
        "A later stage is never a child of an earlier one.\n"
        "3. 'Components' as bullets: **Name**: its role, in one line each.\n"
        "4. 'Data flow' as a numbered list in order; write each step as 'X → Y: what passes between them'.\n"
        "Include only components and links that the material mentions. If a layer, name or technology is not "
        "in the material, leave it out instead of inventing it."
    ),
}

GENERAL_STYLE = (
    "Format like a chat assistant: a direct opening sentence, then short sections. Use **bold** for key "
    "terms, '-' bullets for lists, numbered lists for steps, '## ' headings only when there are 3 or more "
    "parts, a markdown table when comparing items, and fenced blocks for anything code-like."
)


PROCESS_RE = re.compile(r"\b(step|steps|flow|pipeline|stage|stages|process|system|architecture|module|component|layer|workflow|input|output)\b", re.I)
NUMERIC_RE = re.compile(r"\b(formula|equation|calculat\w*|ratio|rate|percent\w*|average|total|cost|score|probability)\b|\d+\s*[=/*+x]\s*\d+", re.I)


def followup_offer(material: str, fmt=None) -> str:
    """
    What to offer next, chosen from what the material contains, so the closing line fits the content instead
    of being the same sentence every time. Returns a phrase like "a skeleton architecture diagram of it".
    """
    offers = []
    if fmt != "architecture" and len(PROCESS_RE.findall(material)) >= 3:
        offers.append("a skeleton architecture diagram of it")
    if fmt != "formula" and NUMERIC_RE.search(material):
        offers.append("the formula behind it and what it means")
    offers.append("a deeper explanation of any part, or why it works that way")
    return " or ".join(offers[:2])


def _followup_block(kind: str, material: str, fmt=None) -> str:
    """Instruction for the end of a reply: suggested questions (summaries only) and one closing offer."""
    offer = followup_offer(material, fmt)
    ask = ""
    if kind == "summary":
        ask = ("After the summary, add a short section titled '**You could also ask**' with 3 bullets. Each "
               "bullet is a specific question that the material above can answer (use its real names, topics "
               "or figures); never a question the material cannot answer. ")
    return (f"{ask}Finish with ONE short closing sentence offering {offer}. Word it naturally for this "
            "content; do not copy this instruction. These closing parts do not count toward the word target.\n")


def build_long_prompt(state: AgentState, facts: str, kind: str, span: str = "") -> str:
    """
    One prompt for every question type. The fine-tuned reasoner supplies the facts; the writer
    only phrases them at the requested length. Numbered rules because in tuning the 3B ignored
    a prose "do not assume" instruction and added general advice the document never gave.
    """
    spec = DETAIL_SPEC.get(state.get("detail") or ANSWER_DETAIL, DETAIL_SPEC["detailed"])
    if len((facts or "").split()) < 25 and kind != "visual":
        # Little to say: a 150-250 word target made the writer invent reasons (tuning: "this ensures the
        # answer is not outdated", "where the PASS and FAIL paths intersect").
        spec = DETAIL_SPEC["brief"]
    evidence = _labeled_evidence(state) if real_text_length(state.get("retrieved_context") or "") > 20 else ""
    multi_page = _page_count(state) > 1
    parts = []
    if evidence and kind not in ("summary", "visual"):
        parts.append(f"Evidence from the source:\n{evidence}")
    label = {"visual": "Observations from reading the image",
             "summary": "Key facts extracted from across the document"}.get(kind, "Facts extracted by the reader")
    parts.append(f"{label}:\n{facts or '(none found)'}")
    if kind == "extractive" and span and len(span.split()) > 14:
        must_include = f"\nThe extracted passage (cover all of its details, in your own words): {span}"
    elif kind == "extractive" and span:
        must_include = f"\nThe answer you must include exactly as written: {span}"
    else:
        must_include = ""
    fmt = detect_format(state["effective_question"]) if kind in ("advisory", "visual") else None
    if fmt:
        return (
            "\n\n".join(parts) + "\n\n"
            f"{_history_block(state)}Question: {state['effective_question']}\n\n"
            f"{FORMAT_INSTRUCTIONS[fmt]}\n\n"
            "Rules: do not claim the document says something it does not; never mention \"the material\", "
            "\"the observations\" or \"the facts\"; finish every fenced block and every sentence.\n"
            + _followup_block(kind, f"{facts} {state.get('retrieved_context') or ''}", fmt) +
            "Answer:"
        )
    return (
        "\n\n".join(parts) + "\n\n"
        f"{_history_block(state)}Question: {state['effective_question']}\n\n"
        f"{KIND_INSTRUCTIONS[kind]}{must_include}\n\n"
        f"{GENERAL_STYLE}\n"
        f"Write about {spec['words']} words, in this shape:\n"
        "1. Open with one sentence that directly answers the question.\n"
        "2. Then explain, using the specific details, names and exact figures above."
        + (" Cite the page like (p. 2) where the source gives one." if multi_page else
           " Do not cite page numbers.") + "\n"
        "3. If the question asks for something the source does not show (a port, a name, a number, "
        "a reason), say plainly that the source does not show it. Never guess it.\n"
        "Rules: use ONLY the material above; never add general knowledge, typical practices, "
        "purposes, benefits or reasons the material does not state, and never say where something "
        "\"is shown\" unless the material says so; use short paragraphs or bullet points; write as if "
        "answering directly and never mention \"the material\", \"the observations\" or \"the facts\".\n"
        + (_followup_block(kind, f"{facts} {state.get('retrieved_context') or ''}")
           if kind in ("summary", "advisory", "visual") else "") +
        "Answer:"
    )


def _page_count(state: AgentState) -> int:
    """Pages in the document being answered from (a single image or a one-page PDF has none to cite)."""
    if state.get("document_id"):
        data, _ = load_structured(state["document_id"])
        if data and data.get("pages"):
            return len(data["pages"])
    return len({c.get("page") for c in state.get("citations", []) if c.get("page")})


def _finish_sentence(text: str, truncated: bool = True) -> str:
    """
    Tidy the end of a reply. An unclosed code fence is always closed. A trailing fragment is dropped only
    when the model really hit the token limit (`truncated`); a reply that stopped by itself is left whole,
    so a finished bullet list or table is not shortened.
    """
    text = text.strip()
    if text.count("```") % 2 == 1:
        # Cut inside a fenced block: drop the half-written last line and close the fence so it still renders.
        body = text.rsplit("\n", 1)[0] if "\n" in text.rsplit("```", 1)[1] else text
        return body.rstrip() + "\n```\n\n*(Output reached the length limit; ask me to continue for the rest.)*"
    if not truncated or not text or text[-1] in ".!?)\"'…`*|:":
        return text
    cut = max(text.rfind(". "), text.rfind(".\n"), text.rfind("! "), text.rfind("? "))
    if cut > 0.5 * len(text):
        return text[:cut + 1].strip()
    lines = text.splitlines()
    return "\n".join(lines[:-1]).strip() if len(lines) > 1 else text


@_timed("summarizer")
def summarizer_node(state: AgentState) -> AgentState:
    """
    Picks the reasoner answer to keep, then (reason_then_write mode) asks the
    3B writer to phrase the reply. The writer can never override the 7B:
    extractive replies must contain the extracted answer verbatim.
    """
    attempts = state["attempts"]
    context = state.get("retrieved_context") or ""
    use_writer = state["mode"] == "reason_then_write"

    chosen = None
    if attempts:
        if state["verified"]:
            chosen = attempts[-1]
        else:
            usable = [a for a in attempts if a["answer"] and not a["uncertain"] and not a["error"]]
            chosen = max(usable, key=lambda a: a["confidence"]) if usable else attempts[-1]

    span = chosen["answer"] if chosen and not chosen["error"] and not chosen["uncertain"] else ""
    state["short_answer"] = span or None
    state["candidate_answer"] = chosen["answer"] if chosen else None
    if chosen:
        state["confidence"] = chosen["confidence"]
        state["model_used"] = chosen["model"]
    state["response_model_used"] = chosen["model"] if chosen else None

    kind = state["query_type"]
    detail = state.get("detail") or ANSWER_DETAIL
    spec = DETAIL_SPEC.get(detail, DETAIL_SPEC["detailed"])

    def compose(facts, span_text=""):
        """The writer phrases the reasoner's facts at the requested length."""
        fmt = detect_format(state["effective_question"]) if kind in ("advisory", "visual") else None
        budget = max(spec["tokens"], FORMAT_TOKENS) if fmt else spec["tokens"]
        reply, secs = _write(build_long_prompt(state, facts, kind, span_text), budget, state=state,
                             extra_options=SKETCH_OPTIONS if fmt == "architecture" else None)
        loop_at = repetition_start(reply)
        if loop_at is not None:   # keep what came before the loop, never the repeated blocks
            reply = reply[:loop_at].rstrip()
            state["trace"].append("Summarizer -> writer started repeating itself; cut the repeated part")
        state["response_model_used"] = WRITER_MODEL
        state["trace"].append(f"Summarizer -> writer {WRITER_MODEL} composed a {detail} answer ({secs:.1f}s)")
        return _finish_sentence(reply, truncated=LAST_STATS.get("done_reason") in ("length", "repetition"))

    try:
        if chosen and chosen["error"]:
            answer = (f"Image analysis failed: {chosen['error']}" if kind == "visual"
                      else f"Model error: {chosen['error']}")

        elif kind == "visual":
            if not span:
                answer = "I could not make out the content of this image."
            else:
                answer = (compose(span) if use_writer else span) or span
                # Save only what was READ from the image, not the answer to this one question.
                reading = span.split("\nAnswer to the question:")[0]
                already_saved = "[visual reading]" in span or "Visual analysis:" in span
                if (state.get("document_id") and state.get("text_is_thin") and not already_saved
                        and save_visual_memory(state["document_id"], reading, chosen["model"])):
                    state["trace"].append("Summarizer -> saved the image reading as searchable visual memory")

        elif kind == "summary" and use_writer:
            if real_text_length(context) < 20:
                answer = "There is no indexed text to summarize. Upload or select a document first."
            else:
                # If the reasoner produced no facts, the writer summarizes the excerpts directly.
                # The fine-tuned reader gives terse facts; when they are thin the writer also gets the document
                # text, otherwise it pads them with the prompt's own wording ("the key facts extracted from...").
                thin = len((span or "").split()) < 40
                answer = compose(f"{span}\n\n{context[:SUMMARY_CHARS]}".strip() if thin else span) or span or context[:600]

        elif kind == "advisory" and use_writer:
            answer = compose(span) or span
            fmt = detect_format(state["effective_question"])
            # Code, formulas and sketches contain numbers (indices, constants) that are not document claims.
            bad = [] if fmt else claimed_numbers_missing(answer, context + " " + span)
            state["unsupported_claims"] = bad
            if bad and not state.get("text_is_thin"):
                answer += f"\n\n(Check: these figures were not found in the source: {', '.join(bad)}.)"

        elif not span:
            answer = NOT_FOUND_MESSAGE

        elif not use_writer or detail == "brief":
            answer = span + (_citation_hint(state) if kind == "extractive" else "")
            if not state["verified"] and kind != "extractive":
                answer += f"\n\n(Low confidence: {state['verification_reason']}.)"

        else:  # extractive, detailed: the verified span must appear verbatim inside the fuller answer
            reply = compose(span, span)
            if len(span.split()) > 14:
                # A list or sentence-long extraction cannot be repeated word for word, and rejecting the writer
                # for that wasted a 67 s call in tuning. Accept its version unless it adds numbers the
                # extraction and the source do not contain (the usual sign of an invented detail).
                invented = claimed_numbers_missing(reply, span + " " + context) if reply else ["empty reply"]
                if reply and not invented:
                    answer = reply
                else:
                    answer = span + _citation_hint(state)
                    state["trace"].append(f"Summarizer -> writer's version rejected ({invented}); returned the extraction")
            elif reply and _normalize(span) in _normalize(reply):
                answer = reply
            else:
                answer = span + _citation_hint(state)
                state["trace"].append("Summarizer -> writer dropped the extracted answer; returned it unchanged")
        if kind == "extractive" and not state["verified"] and span:
            answer += f"\n\n(Low confidence: {state['verification_reason']}.)"
        elif kind == "extractive" and span and state.get("image_read_unverified"):
            answer += "\n\n(Read directly from the image; not cross-checked against extracted text.)"

    except Exception as e:  # writer failed: fall back to the reasoner's own output
        answer = span or f"Answer model error: {e}"
        state["trace"].append(f"Summarizer -> writer failed ({e}); returned the reasoner's output")

    state["final_answer"] = answer
    state["trace"].append(f"Summarizer -> final_answer='{answer[:200]}'")
    return state


# =============================================================================
# Graph
# =============================================================================
@lru_cache(maxsize=1)
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("reasoner", reasoner_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("summarizer", summarizer_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "retriever")
    graph.add_edge("retriever", "reasoner")
    graph.add_edge("reasoner", "verifier")
    graph.add_conditional_edges(
        "verifier", verifier_router, {"reasoner": "reasoner", "summarizer": "summarizer"}
    )
    graph.add_edge("summarizer", END)
    return graph.compile()


# =============================================================================
# Answer cache -- repeated questions on the same source return instantly
# =============================================================================
ANSWER_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
CACHE_SIZE = 256


def _cache_key(question, document_id, image_path, mode, tiers, detail=None):
    return (" ".join(question.lower().split()), document_id, image_path, mode,
            tuple(tiers) if tiers else None, detail)


def clear_cache(document_id=None):
    """Drop cached answers, e.g. after a document is deleted or re-uploaded."""
    for key in [k for k in ANSWER_CACHE if document_id is None or k[1] == document_id]:
        del ANSWER_CACHE[key]


FOLLOW_UP_RE = re.compile(
    r"^(and|also|what about|how about|then|so|but|why)\b"
    r"|\b(it|its|that|those|these|they|them|their|this one|the same|above|previous|earlier)\b"
    r"|\b(more|elaborate|expand|example|examples|simpler|shorter|in detail|continue|go on|again)\b",
    re.I,
)
# Replies to the closing offer ("yes", "sure, do it") and to the suggested questions ("the second one").
AFFIRM_RE = re.compile(
    r"^\W*(yes|yeah|yep|yup|sure|ok|okay|please|go ahead|do it|go on|continue|sounds good|show me|why not)"
    r"(\W+(please|do it|go ahead|thanks|thank you|show me))*\W*$", re.I)
ORDINAL_RE = re.compile(r"\b(first|second|third|1st|2nd|3rd|[123])\b", re.I)
ORDINALS = {"first": 1, "1st": 1, "1": 1, "second": 2, "2nd": 2, "2": 2, "third": 3, "3rd": 3, "3": 3}


def is_follow_up(question: str, history: list) -> bool:
    return bool(history) and len(question.split()) <= 15 and bool(FOLLOW_UP_RE.search(question))


def suggested_questions(answer: str) -> list:
    """The bullets under 'You could also ask' in a previous answer, as plain questions."""
    marker = re.search(r"you could also ask", answer or "", re.I)
    if not marker:
        return []
    out = []
    for line in answer[marker.end():].splitlines():
        m = re.match(r"^\s*(?:[-*•]|\d+[.)])\s+(.*\S)", line)
        if m:
            out.append(re.sub(r"[*_`]", "", m.group(1)).strip())
        elif out and not line.strip():
            break
    return out[:3]


def rewrite_follow_up(question: str, history: list):
    """
    Turn a reply to the previous answer into a complete standalone question, or None.
    "yes" after "Want a skeleton architecture diagram?" must become a diagram request; "the second one"
    must become the second suggested question. Without this the app searched the document for the word
    "yes" (tuning: the model could not keep a conversation going past two questions).
    """
    if not history:
        return None
    prev_q, prev_a = history[-1]["question"], str(history[-1].get("answer") or "")
    q = question.strip()
    topic = re.sub(r"^(please|can you|could you)\s+", "", prev_q.strip(), flags=re.I)
    topic = re.sub(r"\bsummar\w*\s*(of\s+)?", "", topic, flags=re.I).strip() or "this document"   # "summarize" would re-route the reply
    if topic.lower().startswith("the whole"):
        topic = "this document"
    suggestions = suggested_questions(prev_a)
    m = ORDINAL_RE.search(q)
    if suggestions and m and len(q.split()) <= 8:
        i = ORDINALS[m.group(1).lower()]
        if i <= len(suggestions):
            return suggestions[i - 1]
    if AFFIRM_RE.match(q):
        tail = prev_a[-600:].lower()
        if "architecture diagram" in tail or "skeleton architecture" in tail:
            return f"Show the skeleton architecture diagram for: {topic}"
        if "formula" in tail:
            return f"What is the formula behind this and what does it mean? Topic: {topic}"
        return f"Explain in more depth, and why it works that way. Topic: {topic}"
    return None


def _prepare_run(question, model=None, document_id=None, image_path=None, tiers=None,
                 use_retrieval=True, context=None, use_cache=True, mode=None, history=None, detail=None,
                 owner=None, emit=None, force_query_type=None):
    """
    Build the initial graph state (shared by run_query and stream_query).
    Returns (state, cache_key, cached_result); cached_result is None unless a stored answer matches.

    mode: "reason_then_write" (default; 7B reasoner + 3B writer) or "cascade".
    tiers: explicit model list (cheapest first) for evaluation ablations;
           implies the deterministic summarizer (no writer call).
    detail: "detailed" (default; ~150-250 word answers) or "brief".
    history: recent turns [{"question", "answer", "document_id"}], oldest first.
             Follow-ups ("and the due date?") are expanded with the previous
             question and stay on the previous document.
    model: accepted for UI compatibility; not used to pick the pipeline.
    """
    mode = mode or ("cascade" if tiers else PIPELINE_MODE)
    history = list(history or [])[-4:]
    effective_question = question
    rewritten = rewrite_follow_up(question, history)
    if rewritten:   # a complete question now, so routing, search and the models all see what the user meant
        question = effective_question = rewritten
        document_id = document_id or history[-1].get("document_id")
    elif is_follow_up(question, history):
        previous = str(history[-1].get("answer") or "")[:200].replace("\n", " ")
        effective_question = (f"{question} (follow-up to: {history[-1]['question']}; "
                              f"the previous answer began: {previous})")
        document_id = document_id or history[-1].get("document_id")
    if document_id:
        QUERIED_DOCS.add(document_id)
    detail = detail if detail in DETAIL_SPEC else ANSWER_DETAIL
    key = _cache_key(effective_question, document_id, image_path, mode, tiers, detail) + (owner,)
    if use_cache and key in ANSWER_CACHE:
        ANSWER_CACHE.move_to_end(key)
        cached = dict(ANSWER_CACHE[key])
        cached["cache_hit"] = True
        cached["trace"] = cached["trace"] + ["Cache -> hit, returned stored verified answer"]
        cached["timings_original"] = dict(ANSWER_CACHE[key].get("timings") or {})  # what the first run really cost
        cached["timings"] = {"total": 0.0}
        return None, key, cached

    state: AgentState = {
        "owner": owner,
        "emit": emit,
        "question": question,
        "effective_question": effective_question,
        "history": history,
        "document_id": document_id,
        "mode": mode,
        "modality": "",
        "query_type": "",
        "force_query_type": force_query_type,
        "text_is_thin": False,
        "visual_fallback": False,
        "small_doc": False,
        "wide_context": "",
        "fresh_look": False,
        "image_read_unverified": False,
        "detail": detail,
        "use_retrieval": use_retrieval,
        "retrieved_context": context,
        "retrieved_chunks": [],
        "citations": [],
        "image_path": image_path,
        "crop_path": None,
        "plan": build_plan(mode, tiers),
        "tier_index": 0,
        "escalate": False,
        "attempts": [],
        "candidate_answer": None,
        "confidence": None,
        "confidence_source": None,
        "verified": None,
        "evidence_sufficient": None,
        "unsupported_claims": [],
        "verification_reason": "",
        "retry_count": 0,
        "final_answer": None,
        "short_answer": None,
        "model_used": None,
        "reasoning_model_used": None,
        "response_model_used": None,
        "cache_hit": False,
        "timings": {},
        "trace": [],
    }
    return state, key, None


def _store(result, key, use_cache):
    result["emit"] = None  # a stored answer must not keep a reference to a finished request's stream
    if use_cache and result["verified"]:
        ANSWER_CACHE[key] = result
        if len(ANSWER_CACHE) > CACHE_SIZE:
            ANSWER_CACHE.popitem(last=False)


def run_query(question: str, use_cache=True, **kwargs):
    """Answer one question (see _prepare_run for the options). Returns the final graph state."""
    state, key, cached = _prepare_run(question, use_cache=use_cache, **kwargs)
    if cached is not None:
        return cached
    start = time.perf_counter()
    result = build_graph().invoke(state, config={"recursion_limit": 25})
    result["timings"]["total"] = round(time.perf_counter() - start, 3)
    _store(result, key, use_cache)
    return result


STEP_LABELS = {"router": "Understanding the question", "retriever": "Finding the evidence",
               "reasoner": "Reading and reasoning", "verifier": "Checking the answer",
               "summarizer": "Writing the answer"}


def stream_query(question: str, use_cache=True, **kwargs):
    """
    Like run_query, but a generator of events for a live UI:
      {"type": "step",     "node": "reasoner", "label": ..., "trace": [new trace lines]}
      {"type": "progress", "text": "Reading the image: question 2 of 5"}
      {"type": "token",    "text": "..."}                      (the writer's answer, as it is produced)
      {"type": "final",    "state": <final graph state>}       (always last)
    Progress and token events are produced inside the nodes, so they are delivered through a queue
    that the graph thread fills while this generator drains it.
    """
    import queue

    events = queue.Queue()
    state, key, cached = _prepare_run(question, use_cache=use_cache, emit=events.put, **kwargs)
    if cached is not None:
        yield {"type": "final", "state": cached}
        return

    result_box, error_box = {}, {}

    def worker():
        start, seen_trace = time.perf_counter(), 0
        try:
            merged = dict(state)
            for update in build_graph().stream(state, config={"recursion_limit": 25}, stream_mode="updates"):
                for node, delta in update.items():
                    merged.update(delta or {})
                    new_lines = merged["trace"][seen_trace:]
                    seen_trace = len(merged["trace"])
                    events.put({"type": "step", "node": node, "label": STEP_LABELS.get(node, node),
                                "trace": new_lines})
            merged["timings"]["total"] = round(time.perf_counter() - start, 3)
            result_box["state"] = merged
        except Exception as error:  # surface it to the UI instead of dying silently in a thread
            error_box["error"] = error
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        event = events.get()
        if event is None:
            break
        yield event
    if error_box:
        raise error_box["error"]
    _store(result_box["state"], key, use_cache)
    yield {"type": "final", "state": result_box["state"]}


if __name__ == "__main__":
    # python agentic_graph.py               -> invoice questions + image analysis demo
    # python agentic_graph.py image         -> only the image analysis demo
    import sys

    print(f"Hardware profile: {PROFILE}\nMode: {PIPELINE_MODE}")
    demo = [("What is the total balance due?", "doc_test_01"),
            ("Should we pay this invoice early? Explain why.", "doc_test_01"),
            ("Summarize this document.", "doc_test_01")]
    image_docs = sorted(p.split("_structured")[0].split(os.sep)[-1]
                        for p in glob.glob(os.path.join("ocr_cache", "mermaid-diagram*_structured.json")))
    if image_docs:
        demo.append(("Explain this diagram.", image_docs[0]))
    if sys.argv[1:] == ["image"]:
        demo = demo[3:]
    for q, doc in demo:
        print(f"\n{'=' * 60}\nQUESTION: {q}   [document: {doc}]\n{'=' * 60}")
        result = run_query(q, document_id=doc)
        for line in result["trace"]:
            print(f"  {line}")
        print(f"\nFINAL ANSWER: {result['final_answer']}\nTIMINGS: {result['timings']}")
