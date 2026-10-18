"""Authenticated HTTP adapter for the local, synthetic insurance agent."""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from uuid import uuid4
from datetime import date
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from insure.agent import InsureAgent
from insure.llm import OllamaClient
from insure.permissions import AuditLog, Session, Tier
from insure.policy_store import PolicyStore, parse_policy_file
from insure.sandbox_insurer import REQUIRED_DOCS, SandboxInsurer
from insure.ingest_policy import KINDS, ingest_policy_pdf, write_policy_markdown

ROOT = Path(__file__).resolve().parent
POLICY_STORE = PolicyStore.load(ROOT / "insure" / "policies")
DATA_DIR = ROOT / "insurance_data"
AUDIT_DIR = DATA_DIR / "audit"
DRAFT_DIR = DATA_DIR / "drafts"
router = APIRouter()

_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()
_insurers: dict[str, SandboxInsurer] = {}
_insurer_lock = threading.RLock()
_submit_grants: dict[str, float] = {}
_grant_lock = threading.Lock()


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _clean_text(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("must not be blank")
    return value


class AskBody(InputModel):
    policy_id: str = Field(min_length=1, max_length=40)
    question: str = Field(min_length=1, max_length=1000)

    @field_validator("policy_id", "question")
    @classmethod
    def clean(cls, value: str) -> str:
        return _clean_text(value)


class GapBody(InputModel):
    policy_id: str = Field(min_length=1, max_length=40)
    situations: list[str] = Field(min_length=1, max_length=20)

    @field_validator("policy_id")
    @classmethod
    def clean_policy(cls, value: str) -> str:
        return _clean_text(value)

    @field_validator("situations")
    @classmethod
    def clean_situations(cls, values: list[str]) -> list[str]:
        if any(not isinstance(value, str) or not value.strip() or len(value) > 500 for value in values):
            raise ValueError("each situation must contain 1 to 500 characters")
        return [value.strip() for value in values]


class ChecklistBody(InputModel):
    policy_id: str = Field(min_length=1, max_length=40)
    claim_type: str = Field(min_length=1, max_length=60)
    account_id: str | None = Field(default=None, max_length=40)

    @field_validator("policy_id", "claim_type", "account_id")
    @classmethod
    def clean_optional_text(cls, value: str | None) -> str | None:
        return _clean_text(value) if value is not None else value


class ClaimBody(InputModel):
    policy_id: str = Field(min_length=1, max_length=40)
    account_id: str = Field(min_length=1, max_length=40)
    claim_type: str = Field(min_length=1, max_length=60)
    incident_date: date
    amount: float = Field(gt=0, allow_inf_nan=False)
    description: str = Field(min_length=1, max_length=2000)
    documents: list[str] = Field(default_factory=list, max_length=40)

    @field_validator("policy_id", "account_id", "claim_type", "description")
    @classmethod
    def clean_required_text(cls, value: str) -> str:
        return _clean_text(value)

    @field_validator("documents")
    @classmethod
    def clean_documents(cls, values: list[str]) -> list[str]:
        if any(not isinstance(value, str) or not value.strip() or len(value) > 80 for value in values):
            raise ValueError("each document must contain 1 to 80 characters")
        return [value.strip() for value in values]


class SubmitBody(ClaimBody):
    confirmed: bool = False


class PermissionBody(InputModel):
    enabled: bool


def _principal(request: Request) -> str:
    return str(getattr(request.state, "user", None) or "local")


def _user_key(user: str) -> str:
    return hashlib.sha256(user.encode("utf-8")).hexdigest()[:24]


def _lock_for(user: str) -> threading.RLock:
    with _locks_guard:
        return _locks.setdefault(user, threading.RLock())


def _insurer_for(user: str) -> SandboxInsurer:
    key = _user_key(user)
    with _insurer_lock:
        if key not in _insurers:
            _insurers[key] = SandboxInsurer(DATA_DIR / "claims" / f"{key}.json")
        return _insurers[key]


def _store_for(user: str) -> PolicyStore:
    """Combine built-in sample policies with only this user's uploaded policies."""
    folder = DATA_DIR / "policies" / _user_key(user)
    policies = list(POLICY_STORE.policies.values())
    if folder.exists():
        for path in sorted(folder.glob("*.md")):
            policies.append(parse_policy_file(path))
    return PolicyStore(policies)


def _uploaded_policy_path(user: str, policy_id: str) -> Path:
    path = DATA_DIR / "policies" / _user_key(user) / f"{policy_id}.md"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Uploaded policy not found")
    return path

def _agent(user: str) -> InsureAgent:
    key = _user_key(user)
    # Reuse cubic.ai's configured local reasoner. OllamaClient is localhost-only and
    # returns None when Ollama/model is unavailable, preserving deterministic fallback.
    model = os.environ.get("INSURE_MODEL", os.environ.get("REASONER_MODEL", "sovereign-qwen:latest"))
    return InsureAgent(_store_for(user), _insurer_for(user), Session(user=user),
                       AuditLog(AUDIT_DIR / f"{key}.jsonl"), llm=OllamaClient(model=model, timeout=15.0),
                       workdir=DRAFT_DIR / key)


def revoke_submit_permission(user: str | None) -> None:
    """Clear the temporary grant when the user's authenticated session ends."""
    if not user:
        return
    with _grant_lock:
        _submit_grants.pop(user, None)


def _policy(user: str, policy_id: str):
    policy = _store_for(user).policies.get(policy_id)
    if not policy:
        raise HTTPException(status_code=404, detail="Unknown policy")
    return policy


def answer_for_chat(user: str, policy_id: str, question: str) -> dict:
    """Answer a regular cubic.ai chat turn using the user's selected policy."""
    policy = _policy(user, policy_id)
    with _lock_for(user):
        result = _agent(user).ask(question, policy_id)

    citations = _chat_citations(policy, result.get("citations", []))
    intent = result.get("intent", "lookup_clause")
    if intent == "coverage_gap":
        rows = result.get("results", [])
        parts = []
        for row in rows:
            parts.append(f"**{str(row.get('verdict', 'unclear')).upper()}** — {row.get('situation', question)}")
            if row.get("note"):
                parts.append(row["note"])
            citations.extend(_chat_citations(policy, row.get("citations", [])))
        answer = "\n\n".join(parts) or result.get("message", "The policy could not be assessed.")
    elif intent == "claim_checklist" and result.get("ok"):
        answer = ("**Required documents**\n- " + "\n- ".join(result.get("required_documents", [])) +
                  "\n\n**Next steps**\n- " + "\n- ".join(result.get("steps", [])))
        clause_ids = set(result.get("source_clauses", []))
        citations.extend(_chat_citations(policy, [
            {"clause": clause.cite(), "title": clause.title, "text": clause.text}
            for clause in policy.clauses if clause.cite() in clause_ids
        ]))
    else:
        answer = result.get("answer") or result.get("message") or "This policy does not say."

    unique, seen = [], set()
    for citation in citations:
        key = (citation.get("clause"), citation.get("page"))
        if key not in seen:
            unique.append(citation)
            seen.add(key)
    return {
        "final_answer": answer,
        "answer": answer,
        "citations": unique,
        "query_type": f"insurance · {intent.replace('_', ' ')}",
        "model_used": "policy tools",
        "response_model_used": "policy tools",
        "verified": bool(unique),
        "evidence_sufficient": bool(unique),
        "verification_reason": "Grounded in the selected policy's cited clauses.",
        "trace": ["Insurance policy selected", f"Policy tool → {intent}", "Citations attached"],
        "timings": {"total": 0.0},
        "insurance_policy_id": policy_id,
        "insurance_policy_title": policy.title,
    }


def _chat_citations(policy, citations: list[dict]) -> list[dict]:
    """Adapt insurance citations to cubic.ai's regular chat citation format."""
    by_cite = {clause.cite(): clause for clause in policy.clauses}
    output = []
    for item in citations:
        clause = by_cite.get(item.get("clause"))
        output.append({
            "document_id": None,
            "insurance_policy_id": policy.policy_id,
            "filename": f"{policy.title} policy",
            "page": clause.page if clause else item.get("page"),
            "snippet": item.get("text", ""),
            "clause": item.get("clause", ""),
            "title": item.get("title", ""),
        })
    return output


def _account(user: str, account_id: str, policy_id: str | None = None) -> dict:
    account = _insurer_for(user).get_account(account_id)
    if account is None and policy_id and policy_id.startswith("UPL-") and account_id == f"SIM-{policy_id}":
        policy = _policy(user, policy_id)
        account = _insurer_for(user).ensure_demo_account(policy_id, policy.ptype)
    if not account:
        raise HTTPException(status_code=404, detail="Unknown simulated account")
    if policy_id and account["policy_id"] != policy_id:
        raise HTTPException(status_code=422, detail="Account does not belong to the selected policy")
    return account


def _claim_type(account: dict, claim_type: str) -> str:
    if claim_type not in REQUIRED_DOCS.get(account["ptype"], {}):
        raise HTTPException(status_code=422, detail="Claim type is not valid for this account")
    return claim_type


def _claim_fields(user: str, body: ClaimBody) -> dict:
    _policy(user, body.policy_id)
    account = _account(user, body.account_id, body.policy_id)
    claim_type = _claim_type(account, body.claim_type)
    return {"policy_id": account["policy_id"], "account_id": account["account_id"],
            "claim_type": claim_type, "incident_date": body.incident_date.isoformat(),
            "amount": float(body.amount), "description": body.description,
            "documents": body.documents}


def _audit(user: str) -> AuditLog:
    return AuditLog(AUDIT_DIR / f"{_user_key(user)}.jsonl")


def _audit_records(user: str, limit: int = 100) -> list[dict]:
    path = AUDIT_DIR / f"{_user_key(user)}.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


@router.get("/insurance", include_in_schema=False)
def insurance_page():
    # Keep the legacy URL inside the main cubic.ai interface.
    return FileResponse(ROOT / "frontend_quietly.html", media_type="text/html", headers={"Cache-Control": "no-store"})


@router.get("/insure/policies")
@router.get("/insurance/api/policies")
def list_policies(request: Request):
    store = _store_for(_principal(request))
    policies = []
    for item in store.list_policies():
        policy = store.policies[item["policy_id"]]
        item["claim_types"] = sorted(REQUIRED_DOCS.get(policy.ptype, {}))
        policies.append(item)
    return {"policies": policies}


@router.post("/insure/policies/upload")
async def upload_policy(request: Request, file: UploadFile = File(...), title: str = Form(...), ptype: str = Form(...)):
    user = _principal(request)
    title, ptype = title.strip(), ptype.strip().lower()
    if not title or len(title) > 120 or "|" in title:
        raise HTTPException(status_code=422, detail="title must contain 1-120 characters")
    if ptype not in {"travel", "health", "motor"}:
        raise HTTPException(status_code=422, detail="ptype must be travel, health or motor")
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=422, detail="Upload a PDF policy")
    content = await file.read(20 * 1024 * 1024 + 1)
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Policy PDF exceeds 20 MB")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=422, detail="Uploaded file is not a PDF")
    policy_id = f"UPL-{uuid4().hex[:12].upper()}"
    folder = DATA_DIR / "policies" / _user_key(user)
    folder.mkdir(parents=True, exist_ok=True)
    pdf_path, md_path = folder / f"{policy_id}.pdf", folder / f"{policy_id}.md"
    pdf_path.write_bytes(content)
    try:
        parsed = ingest_policy_pdf(pdf_path, md_path, policy_id, title, ptype)
    except ValueError as exc:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    with _lock_for(user):
        _agent(user).audit.write(event="policy_upload", user=user, tool="ingest_policy_pdf",
                                 tier="DRAFT", allowed=True, reason="uploaded policy parsed for review",
                                 args={"policy_id": policy_id, "clauses": len(parsed["clauses"])})
        account = _insurer_for(user).ensure_demo_account(policy_id, ptype)
    return {"policy_id": policy_id, "title": title, "type": ptype,
            "review_required": True, "clauses": parsed["clauses"],
            "simulated_account_id": account["account_id"]}


class ClauseReview(InputModel):
    clause_id: str = Field(min_length=1, max_length=80)
    kind: str

    @field_validator("kind")
    @classmethod
    def valid_kind(cls, value: str) -> str:
        if value not in KINDS:
            raise ValueError("unknown clause kind")
        return value


class ReviewBody(InputModel):
    clauses: list[ClauseReview] = Field(min_length=1, max_length=500)


@router.get("/insure/policies/{policy_id}/pages/{page_number}")
def policy_page_image(request: Request, policy_id: str, page_number: int):
    if page_number < 1 or page_number > 1000:
        raise HTTPException(status_code=404, detail="Policy page not found")
    folder = DATA_DIR / "policies" / _user_key(_principal(request))
    image = folder / f"{policy_id}_page_{page_number}.png"
    if not image.is_file():
        raise HTTPException(status_code=404, detail="Policy page image not found")
    return FileResponse(image, media_type="image/png", headers={"Cache-Control": "no-store"})

@router.get("/insure/policies/{policy_id}/clauses")
def review_policy(request: Request, policy_id: str):
    path = _uploaded_policy_path(_principal(request), policy_id)
    policy = parse_policy_file(path)
    return {"policy_id": policy.policy_id, "title": policy.title, "type": policy.ptype,
            "clauses": [{"clause_id": c.clause_id, "page": c.page, "kind": c.kind,
                         "title": c.title, "text": c.text} for c in policy.clauses]}


@router.post("/insure/policies/{policy_id}/review")
def update_policy_review(request: Request, policy_id: str, body: ReviewBody):
    user = _principal(request)
    path = _uploaded_policy_path(user, policy_id)
    policy = parse_policy_file(path)
    by_id = {c.clause_id: c for c in policy.clauses}
    updates = {row.clause_id: row.kind for row in body.clauses}
    if not set(updates).issubset(by_id):
        raise HTTPException(status_code=422, detail="Review contains an unknown clause id")
    for clause_id, kind in updates.items():
        by_id[clause_id].kind = kind
    parsed = {"policy_id": policy.policy_id, "title": policy.title, "type": policy.ptype,
              "clauses": [{"clause_id": c.clause_id, "page": c.page, "kind": c.kind,
                           "title": c.title, "text": c.text} for c in policy.clauses]}
    write_policy_markdown(parsed, path)
    with _lock_for(user):
        _agent(user).audit.write(event="policy_review", user=user, tool="review_policy_clauses",
                                 tier="DRAFT", allowed=True, reason="clause kinds reviewed",
                                 args={"policy_id": policy_id, "updated_count": len(updates)})
    return {"policy_id": policy_id, "reviewed": True, "clauses": parsed["clauses"]}


@router.get("/insure/accounts")
@router.get("/insurance/api/accounts")
def list_accounts(request: Request):
    user = _principal(request)
    with _lock_for(user):
        insurer = _insurer_for(user)
        for policy in _store_for(user).policies.values():
            if policy.policy_id.startswith("UPL-"):
                insurer.ensure_demo_account(policy.policy_id, policy.ptype)
        accounts = [insurer.get_account(account_id) for account_id in insurer.accounts]
        return {"accounts": [{k: account[k] for k in ("account_id", "holder", "policy_id", "ptype")}
                             for account in accounts]}


@router.get("/insure/claims")
def list_claims(request: Request):
    user = _principal(request)
    with _lock_for(user):
        return {"claims": [vars(claim) for claim in _insurer_for(user).claims.values()]}


@router.post("/insure/ask")
@router.post("/insurance/api/ask")
def ask_policy(request: Request, body: AskBody):
    user = _principal(request)
    _policy(user, body.policy_id)
    with _lock_for(user):
        return _agent(user).ask(body.question, body.policy_id)


@router.post("/insure/gap")
@router.post("/insurance/api/coverage-gap")
def coverage_gap(request: Request, body: GapBody):
    user = _principal(request)
    _policy(user, body.policy_id)
    with _lock_for(user):
        return _agent(user).run_tool("coverage_gap", policy_id=body.policy_id, situations=body.situations)


@router.post("/insure/claim/checklist")
@router.post("/insurance/api/checklist")
def claim_checklist(request: Request, body: ChecklistBody):
    user = _principal(request)
    policy = _policy(user, body.policy_id)
    if body.account_id:
        account = _account(user, body.account_id, body.policy_id)
        claim_type = _claim_type(account, body.claim_type)
    else:
        claim_type = body.claim_type
        if claim_type not in REQUIRED_DOCS.get(policy.ptype, {}):
            raise HTTPException(status_code=422, detail="Claim type is not valid for this policy")
    with _lock_for(user):
        return _agent(user).run_tool("claim_checklist", policy_id=body.policy_id, claim_type=claim_type)


@router.post("/insure/claim/draft")
@router.post("/insurance/api/draft")
def draft_claim(request: Request, body: ClaimBody):
    user = _principal(request)
    fields = _claim_fields(user, body)
    with _lock_for(user):
        return _agent(user).run_tool("draft_claim", **fields)


@router.post("/insure/submit-permission")
@router.post("/insurance/api/submit-permission")
def submit_permission(request: Request, body: PermissionBody):
    user = _principal(request)
    with _grant_lock:
        if body.enabled:
            _submit_grants[user] = time.monotonic() + 600
        else:
            _submit_grants.pop(user, None)
    with _lock_for(user):
        _agent(user).audit.write(event="permission", user=user, tool="submit_claim_sandbox",
                                 tier="SUBMIT", allowed=body.enabled,
                                 reason="user enabled sandbox submissions for 10 minutes" if body.enabled else "user revoked permission",
                                 args={})
    return {"enabled": body.enabled, "expires_in_seconds": 600 if body.enabled else 0}


@router.get("/insure/submit-permission")
@router.get("/insurance/api/submit-permission")
def submit_permission_status(request: Request):
    user = _principal(request)
    with _grant_lock:
        expires_at = _submit_grants.get(user, 0)
        remaining = max(0, int(expires_at - time.monotonic()))
    return {"enabled": remaining > 0, "expires_in_seconds": remaining}


@router.post("/insure/claim/submit")
@router.post("/insurance/api/submit")
def submit_claim(request: Request, body: SubmitBody):
    user = _principal(request)
    fields = _claim_fields(user, body)
    with _grant_lock:
        allowed = _submit_grants.get(user, 0) > time.monotonic()
    with _insurer_lock, _lock_for(user):
        agent = _agent(user)
        agent.session = Session(user=user, max_tier=Tier.SUBMIT if allowed else Tier.DRAFT)
        result = agent.run_tool("submit_claim_sandbox", confirmed=body.confirmed,
                                **{k: fields[k] for k in ("account_id", "claim_type", "amount", "incident_date", "description", "documents")})
        if result.get("denied"):
            raise HTTPException(status_code=403, detail=result.get("reason", "Submission denied"))
        return result


@router.get("/insure/audit/verify")
@router.get("/insurance/api/audit")
def audit_verify(request: Request):
    user = _principal(request)
    with _lock_for(user):
        try:
            intact, records = _audit(user).verify()
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            intact, records = False, 0
        return {"ok": intact, "records": records, "intact": intact, "entries": records}


@router.get("/insure/audit")
def audit_records(request: Request):
    user = _principal(request)
    with _lock_for(user):
        try:
            intact, count = _audit(user).verify()
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            intact, count = False, 0
        return {"ok": intact, "records": _audit_records(user, 100), "verified_records": count}



