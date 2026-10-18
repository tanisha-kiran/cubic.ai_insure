"""Tool-calling agent: route -> authorise -> run tool -> verify -> log.

The router is a transparent keyword classifier (like the cubic.ai router). All tool
use goes through `run_tool`, which enforces permission tiers and writes the audit log.
"""
from __future__ import annotations
import re
from pathlib import Path

from . import tools
from .llm import NullLLM
from .permissions import AuditLog, PermissionDenied, Session, Tier, authorize
from .policy_store import PolicyStore
from .sandbox_insurer import SandboxInsurer

TOOL_TIERS = {"lookup_clause": Tier.READ, "coverage_gap": Tier.READ, "claim_checklist": Tier.DRAFT,
              "draft_claim": Tier.DRAFT, "submit_claim_sandbox": Tier.SUBMIT}


def route(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(how (do|can) i (make|start|file)|what (documents|do i need)|checklist|steps|claim process|make a claim)\b.*\bclaim|\bclaim\b.*\b(documents|checklist|steps|process|file)\b", t):
        return "claim_checklist"
    if re.search(r"\b(draft|write|prepare)\b.*\bclaim\b", t):
        return "draft_claim"
    if re.search(r"\b(submit|send)\b.*\bclaim\b|\bfile\s+(my|the)\s+claim\s+(now|today)\b|\bfile\s+a\s+claim\s+for\b", t):
        return "submit_claim_sandbox"
    if re.search(r"\b(gap|gaps|coverage gap|what am i missing|what is missing from my coverage|portfolio gap|not covered|am i covered|would .* be covered|does .* cover|is .* covered|can i claim|could i claim|will .* cover|insured for|eligible for|protection for)\b", t):
        return "coverage_gap"
    return "lookup_clause"


class InsureAgent:
    def __init__(self, store: PolicyStore, insurer: SandboxInsurer, session: Session, audit: AuditLog,
                 llm=None, workdir: str | Path = "workspace"):
        self.store, self.insurer, self.session, self.audit = store, insurer, session, audit
        self.llm = llm or NullLLM()
        self.workdir = Path(workdir)

    def run_tool(self, name: str, confirmed: bool = False, **args) -> dict:
        tier = TOOL_TIERS[name]
        summary = {k: (v if not isinstance(v, str) or len(v) < 120 else v[:117] + "...") for k, v in args.items()}
        try:
            authorize(self.session, name, tier, confirmed, self.audit, summary)
        except PermissionDenied as e:
            return {"ok": False, "denied": True, "reason": str(e), "tool": name}
        if name == "lookup_clause":
            res = tools.lookup_clause(self.store, args["question"], args["policy_id"], self.llm)
        elif name == "coverage_gap":
            res = tools.coverage_gap(self.store, args["policy_id"], args["situations"])
        elif name == "claim_checklist":
            res = tools.claim_checklist(self.store, args["policy_id"], args["claim_type"])
        elif name == "draft_claim":
            res = tools.draft_claim(self.store, out_dir=self.workdir / "drafts", **args)
        elif name == "submit_claim_sandbox":
            res = tools.submit_claim_sandbox(self.insurer, **args)
        else:
            return {"ok": False, "error": "unknown_tool"}
        self.audit.write(event="result", user=self.session.user, tool=name, tier=tier.name,
                         outcome=_outcome(name, res))
        return res

    def ask(self, question: str, policy_id: str) -> dict:
        """Route free text to a read-only policy or claims guidance tool."""
        intent = route(question)
        if intent == "lookup_clause":
            return {"intent": intent, **self.run_tool("lookup_clause", question=question, policy_id=policy_id)}
        if intent == "coverage_gap":
            result = self.run_tool("coverage_gap", policy_id=policy_id, situations=[question])
            return {"intent": intent, **result}
        if intent == "claim_checklist":
            policy = self.store.policies.get(policy_id)
            valid = sorted(tools.REQUIRED_DOCS.get(policy.ptype, {})) if policy else []
            normalized = question.lower()
            aliases = {"baggage": ("baggage", "lost bag", "luggage", "stolen bag", "theft"),
                       "medical": ("medical", "hospital", "illness", "injury", "health"),
                       "cancellation": ("cancellation", "cancelled trip", "cancel trip"),
                       "delay": ("flight delay", "delayed flight", "flight delayed"),
                       "hospitalisation": ("hospital", "illness", "injury", "health"),
                       "own_damage": ("vehicle damage", "car damage", "collision", "accident"),
                       "theft": ("theft", "stolen", "robbery")}
            claim_type = next((kind for kind in valid if kind in normalized or
                               any(alias in normalized for alias in aliases.get(kind, ()))), None)
            if not claim_type:
                return {"intent": intent, "needs_claim_type": True, "valid_claim_types": valid,
                        "message": "I can prepare the policy-based checklist. Which claim type applies?"}
            return {"intent": intent, **self.run_tool("claim_checklist", policy_id=policy_id, claim_type=claim_type)}
        return {"intent": intent, "needs_confirmation": True,
                "message": "Claim submission requires the claim details, an enabled submit permission, and your explicit confirmation in the claim form."}


def _outcome(name: str, res: dict) -> dict:
    if name == "lookup_clause":
        return {"found": res.get("found"), "mode": res.get("mode"),
                "citations": [c["clause"] for c in res.get("citations", [])]}
    if name == "coverage_gap":
        return res.get("summary", {})
    if name == "submit_claim_sandbox":
        return {"claim_id": res.get("claim_id"), "status": res.get("status"), "reason": res.get("decision_reason"),
                "ok": res.get("ok")}
    return {"ok": res.get("ok")}
