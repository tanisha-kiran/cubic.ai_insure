"""Permission tiers, confirmation gate and tamper-evident audit log.

Tier 0  READ     answer questions, analyse coverage (no side effects)
Tier 1  DRAFT    produce checklists / claim drafts (written to the local workspace only)
Tier 2  SUBMIT   actions with an external effect (sandbox insurer): need explicit user confirmation

Every tool call, allowed or denied, is appended to a JSONL audit log. Each record
carries the SHA-256 of the previous record, so deletion or editing of earlier lines
is detectable with `AuditLog.verify()`.
"""
from __future__ import annotations
import hashlib
import json
import re
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path


_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ().-]{6,}\d)(?!\w)")
_LONG_DIGITS_RE = re.compile(r"(?<!\d)(?:\d[ -]?){7,}(?!\d)")


def redact_sensitive(value):
    """Redact common identifiers in audit args and transcript values."""
    if isinstance(value, str):
        value = _EMAIL_RE.sub("[REDACTED_EMAIL]", value)
        value = _PHONE_RE.sub("[REDACTED_NUMBER]", value)
        return _LONG_DIGITS_RE.sub("[REDACTED_NUMBER]", value)
    if isinstance(value, dict):
        return {key: redact_sensitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_sensitive(item) for item in value]
    return value

class Tier(IntEnum):
    READ = 0
    DRAFT = 1
    SUBMIT = 2


class PermissionDenied(Exception):
    pass


@dataclass
class Session:
    """What the current user has allowed. Default is read + draft, never submit."""
    user: str
    max_tier: Tier = Tier.DRAFT


class AuditLog:
    GENESIS = "0" * 64

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last = self._read_last_hash()

    def _read_last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return self.GENESIS
        last = None
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = line
        return json.loads(last)["hash"] if last else self.GENESIS

    @staticmethod
    def _digest(prev: str, body: dict) -> str:
        payload = prev + json.dumps(body, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def write(self, **fields) -> dict:
        if "args" in fields:
            fields["args"] = redact_sensitive(fields["args"])
        body = {"ts": round(time.time(), 3), **fields}
        h = self._digest(self._last, body)
        rec = {**body, "prev": self._last, "hash": h}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._last = h
        return rec

    def verify(self) -> tuple[bool, int]:
        prev, n = self.GENESIS, 0
        if not self.path.exists():
            return True, 0
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                h, p = rec.pop("hash"), rec.pop("prev")
                if p != prev or self._digest(prev, rec) != h:
                    return False, n
                prev, n = h, n + 1
        return True, n


def authorize(session: Session, tool: str, tier: Tier, confirmed: bool, audit: AuditLog,
              args_summary: dict | None = None) -> None:
    """Raise PermissionDenied unless the session may run this tool. Always logs the decision."""
    reason = None
    if tier > session.max_tier:
        reason = f"tier {tier.name} exceeds session limit {session.max_tier.name}"
    elif tier == Tier.SUBMIT and not confirmed:
        reason = "tier SUBMIT requires explicit user confirmation"
    audit.write(event="authorize", user=session.user, tool=tool, tier=tier.name,
                allowed=reason is None, reason=reason, args=args_summary or {})
    if reason:
        raise PermissionDenied(reason)

