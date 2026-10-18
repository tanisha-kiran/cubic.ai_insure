"""Mock insurer used as the sandbox / simulated account for execution evidence.

It is deterministic and local. It holds simulated customers and policies and applies a
few simple adjudication rules, so the agent's tier-2 action (submitting a claim) has a
real, observable effect that is logged. It never contacts a network.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path

REQUIRED_DOCS = {
    "travel": {"medical": ["doctor_report", "hospital_invoice", "discharge_summary"],
               "baggage": ["police_or_airline_report", "proof_of_ownership", "purchase_receipt"],
               "cancellation": ["booking_confirmation", "cancellation_invoice"],
               "delay": ["carrier_delay_confirmation"]},
    "health": {"hospitalisation": ["claim_form", "discharge_summary", "original_bills", "prescriptions"]},
    "motor": {"own_damage": ["claim_form", "registration_certificate", "driving_licence", "repair_estimate"],
              "theft": ["claim_form", "police_report", "registration_certificate", "both_keys"]},
}


@dataclass
class SimAccount:
    account_id: str
    holder: str
    policy_id: str
    ptype: str
    start_date: str
    sum_insured: float
    claims_paid: float = 0.0


@dataclass
class Claim:
    claim_id: str
    account_id: str
    claim_type: str
    amount: float
    incident_date: str
    description: str
    documents: list
    status: str = "RECEIVED"
    decision_reason: str = ""
    approved_amount: float = 0.0
    history: list = field(default_factory=list)


class SandboxInsurer:
    def __init__(self, state_path: str | Path | None = None):
        self.state_path = Path(state_path) if state_path else None
        self.accounts: dict[str, SimAccount] = {}
        self.claims: dict[str, Claim] = {}
        self._seed()

    def _seed(self) -> None:
        for a in [
            SimAccount("ACC-1001", "Asha Rao (simulated)", "TRV-001", "travel", "2026-09-01", 100000),
            SimAccount("ACC-1002", "Vikram Shah (simulated)", "HLT-001", "health", "2026-08-20", 500000),
            SimAccount("ACC-1003", "Meera Nair (simulated)", "MTR-001", "motor", "2025-11-10", 600000),
            SimAccount("ACC-1004", "Dev Patel (simulated)", "HLT-001", "health", "2024-01-15", 500000),
        ]:
            self.accounts[a.account_id] = a

    # -- read-only -----------------------------------------------------------------
    def get_account(self, account_id: str) -> dict | None:
        a = self.accounts.get(account_id)
        return asdict(a) if a else None

    def ensure_demo_account(self, policy_id: str, ptype: str) -> dict:
        """Create an explicitly simulated account for an uploaded policy, once."""
        account_id = f"SIM-{policy_id}"
        if account_id not in self.accounts:
            limits = {"travel": 100000, "health": 500000, "motor": 600000}
            self.accounts[account_id] = SimAccount(
                account_id, "Policy holder (simulated)", policy_id, ptype,
                "2024-01-01", limits[ptype],
            )
        return self.get_account(account_id)

    def claim_status(self, claim_id: str) -> dict | None:
        c = self.claims.get(claim_id)
        return asdict(c) if c else None

    # -- state-changing ------------------------------------------------------------
    def submit_claim(self, account_id: str, claim_type: str, amount: float, incident_date: str,
                     description: str, documents: list[str]) -> dict:
        acc = self.accounts.get(account_id)
        if acc is None:
            return {"ok": False, "error": "unknown_account"}
        if amount <= 0:
            return {"ok": False, "error": "invalid_amount"}
        cid = f"CLM-{len(self.claims) + 1:04d}"
        claim = Claim(cid, account_id, claim_type, float(amount), incident_date, description, list(documents))
        claim.history.append({"status": "RECEIVED"})
        self._adjudicate(claim, acc)
        self.claims[cid] = claim
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps({k: asdict(v) for k, v in self.claims.items()}, indent=2))
        return {"ok": True, **asdict(claim)}

    def _adjudicate(self, claim: Claim, acc: SimAccount) -> None:
        required = REQUIRED_DOCS.get(acc.ptype, {}).get(claim.claim_type)
        if required is None:
            return self._decide(claim, "REJECTED", f"claim type '{claim.claim_type}' not valid for {acc.ptype} policy")
        missing = [d for d in required if d not in claim.documents]
        if missing:
            return self._decide(claim, "PENDING_DOCUMENTS", "missing documents: " + ", ".join(missing))
        try:
            inc = date.fromisoformat(claim.incident_date)
            start = date.fromisoformat(acc.start_date)
        except ValueError:
            return self._decide(claim, "REJECTED", "invalid date format, use YYYY-MM-DD")
        if inc < start:
            return self._decide(claim, "REJECTED", "incident occurred before the policy start date")
        # health: 30-day initial waiting period for illness (accident excepted)
        if acc.ptype == "health" and (inc - start).days < 30 and "accident" not in claim.description.lower():
            return self._decide(claim, "REJECTED", "within 30-day initial waiting period (clause HLT-3.1)")
        text = claim.description.lower()
        if any(w in text for w in ("drunk", "alcohol", "intoxicat")):
            return self._decide(claim, "REJECTED", "loss involved alcohol or intoxicants (exclusion)")
        remaining = acc.sum_insured - acc.claims_paid
        approved = min(claim.amount, remaining)
        acc.claims_paid += approved
        claim.approved_amount = approved
        status = "APPROVED" if approved >= claim.amount else "APPROVED_PARTIAL"
        self._decide(claim, status, "within sum insured" if status == "APPROVED" else "capped at remaining sum insured")

    @staticmethod
    def _decide(claim: Claim, status: str, reason: str) -> None:
        claim.status, claim.decision_reason = status, reason
        claim.history.append({"status": status, "reason": reason, "at": datetime(2026, 1, 1).isoformat() + "Z-sim"})
