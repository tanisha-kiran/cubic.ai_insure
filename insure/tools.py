"""The five agent tools. Each returns a plain dict and never raises on bad input.

Tier 0: lookup_clause, coverage_gap
Tier 1: claim_checklist, draft_claim
Tier 2: submit_claim_sandbox (needs confirmation)
"""
from __future__ import annotations
import os
import re
from pathlib import Path

from .llm import NullLLM, extract_json
from .policy_store import Clause, PolicyStore, tokenize
from .sandbox_insurer import REQUIRED_DOCS, SandboxInsurer

NOT_FOUND = "This policy does not say. I could not find a clause that answers that."
DISCLAIMER = "Decision support only, not insurance or legal advice. Check the cited clause and your insurer."

# Retrieval confidence needed before the agent will answer instead of abstaining.
MIN_SCORE = 3.0
MIN_MATCH_RATIO = 0.34


def _terms(text: str) -> set[str]:
    return set(tokenize(text))


def match_ratio(query: str, clause: Clause) -> float:
    q = _terms(query)
    if not q:
        return 0.0
    return len(q & _terms(clause.title + " " + clause.text)) / len(q)


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d[\d,]*\.?\d*", text.replace(",", "")))


def _best_sentence(question: str, clause: Clause) -> str:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clause.text) if s.strip()]
    if not sents:
        return clause.text
    q = _terms(question)
    return max(sents, key=lambda s: len(q & _terms(s)))



def _search(store: PolicyStore, query: str, top_k: int, policy_id: str, kinds=None, retrieval: str | None = None):
    mode = (retrieval or os.environ.get("INSURE_RETRIEVAL", "bm25")).lower()
    if mode == "bm25":
        return store.search(query, top_k=top_k, policy_id=policy_id, kinds=kinds)
    if mode == "hybrid":
        from .hybrid import hybrid_search
        return hybrid_search(store, query, top_k=top_k, policy_id=policy_id, kinds=kinds)
    raise ValueError("retrieval must be bm25 or hybrid")
# ------------------------------------------------------------------ tier 0
def lookup_clause(store: PolicyStore, question: str, policy_id: str, llm=None, top_k: int = 4, retrieval: str | None = None) -> dict:
    llm = llm or NullLLM()
    hits = _search(store, question, top_k=top_k, policy_id=policy_id, retrieval=retrieval)
    if not hits:
        return {"found": False, "answer": NOT_FOUND, "citations": [], "mode": "abstain", "disclaimer": DISCLAIMER}
    best, score = hits[0]
    ratio = match_ratio(question, best)
    if score < MIN_SCORE or ratio < MIN_MATCH_RATIO:
        return {"found": False, "answer": NOT_FOUND, "citations": [], "mode": "abstain",
                "best_score": round(score, 2), "match_ratio": round(ratio, 2), "disclaimer": DISCLAIMER}

    answer, cites, mode = _best_sentence(question, best), [best], "extractive"
    if getattr(llm, "name", "none") != "none":
        ctx = "\n".join(f"[{c.clause_id}] {c.text}" for c, _ in hits)
        prompt = ("Answer the question using ONLY the policy clauses below. If they do not answer it, "
                  'reply {"found": false}. Otherwise reply with JSON only: '
                  '{"found": true, "answer": "<one or two sentences>", "clause_ids": ["<id>", ...]}.\n\n'
                  f"Clauses:\n{ctx}\n\nQuestion: {question}\nJSON:")
        obj = extract_json(llm.generate(prompt))
        verified = _verify_llm_answer(obj, hits)
        if obj is not None and obj.get("found") is False:
            return {"found": False, "answer": NOT_FOUND, "citations": [], "mode": "llm_abstain",
                    "disclaimer": DISCLAIMER}
        if verified:
            answer, mode = obj["answer"], "llm_verified"
            by_id = {c.clause_id: c for c, _ in hits}
            cites = [by_id[i] for i in obj["clause_ids"]]
    return {"found": True, "answer": answer, "mode": mode, "disclaimer": DISCLAIMER,
            "citations": [{"clause": c.cite(), "title": c.title, "page": c.page, "text": c.text} for c in cites],
            "best_score": round(score, 2), "match_ratio": round(ratio, 2)}


def _verify_llm_answer(obj: dict | None, hits) -> bool:
    """Accept a model answer only if it cites retrieved clauses and invents no numbers."""
    if not obj or obj.get("found") is not True:
        return False
    ids, ans = obj.get("clause_ids"), obj.get("answer")
    if not isinstance(ids, list) or not ids or not isinstance(ans, str) or not ans.strip():
        return False
    by_id = {c.clause_id: c for c, _ in hits}
    if any(i not in by_id for i in ids):
        return False
    allowed = set().union(*[_numbers(by_id[i].text) for i in ids])
    return _numbers(ans) <= allowed


def coverage_gap(store: PolicyStore, policy_id: str, situations: list[str], retrieval: str | None = None) -> dict:
    """For each situation: covered / excluded / unclear, each with citations and conditions."""
    results = []
    for sit in situations:
        cov = _search(store, sit, top_k=3, policy_id=policy_id, kinds={"coverage"}, retrieval=retrieval)
        exc = _search(store, sit, top_k=2, policy_id=policy_id, kinds={"exclusion"}, retrieval=retrieval)
        cond = _search(store, sit, top_k=2, policy_id=policy_id, kinds={"condition", "limit"}, retrieval=retrieval)
        cov_best = cov[0] if cov else None
        exc_best = exc[0] if exc else None
        cov_ok = bool(cov_best and cov_best[1] >= MIN_SCORE and match_ratio(sit, cov_best[0]) >= MIN_MATCH_RATIO)
        exc_ok = bool(exc_best and exc_best[1] >= MIN_SCORE and match_ratio(sit, exc_best[0]) >= MIN_MATCH_RATIO)
        # An exclusion decides the verdict only if it matches the situation at least as strongly as the
        # best coverage clause. Weaker, possibly relevant exclusions are still shown as warnings, because in
        # insurance an exclusion can override cover.
        if exc_ok and (not cov_ok or exc_best[1] >= cov_best[1]):
            verdict, main = "excluded", [exc_best[0]]
        elif cov_ok:
            verdict, main = "covered", [cov_best[0]]
        else:
            verdict, main = "unclear", []
        main_ids = {c.clause_id for c in main}
        warn = [c for c, sc in exc if c.clause_id not in main_ids and sc >= 0.6 * MIN_SCORE
                and match_ratio(sit, c) >= 0.2] if verdict == "covered" else []
        conds = [c for c, sc in cond if sc >= 0.6 * MIN_SCORE and match_ratio(sit, c) >= 0.2] if verdict != "unclear" else []
        results.append({
            "situation": sit, "verdict": verdict,
            "citations": [{"clause": c.cite(), "title": c.title, "text": c.text} for c in main],
            "conditions": [{"clause": c.cite(), "title": c.title, "text": c.text} for c in conds],
            "possible_exclusions": [{"clause": c.cite(), "title": c.title, "text": c.text} for c in warn],
            "note": ("Policy text does not clearly address this. Ask the insurer in writing."
                     if verdict == "unclear" else
                     "Check the possible exclusions listed before relying on this cover." if warn else "")})
    summary = {v: sum(1 for r in results if r["verdict"] == v) for v in ("covered", "excluded", "unclear")}
    return {"policy_id": policy_id, "results": results, "summary": summary, "disclaimer": DISCLAIMER}


# ------------------------------------------------------------------ tier 1
def claim_checklist(store: PolicyStore, policy_id: str, claim_type: str) -> dict:
    """Required documents, deadlines and conditions, built from the policy's own claims clauses."""
    pol = store.policies.get(policy_id)
    if pol is None:
        return {"ok": False, "error": "unknown_policy"}
    valid = REQUIRED_DOCS.get(pol.ptype, {})
    if claim_type not in valid:
        return {"ok": False, "error": "unknown_claim_type", "valid_claim_types": sorted(valid)}
    claims_cl = [c for c in pol.clauses if c.kind == "claims"]
    cond_cl = [c for c in pol.clauses if c.kind == "condition"]
    deadlines = []
    for c in claims_cl + cond_cl:
        for sent in re.split(r"(?<=[.!?])\s+", c.text):
            if re.search(r"\b(within|before|at least|immediately)\b.*\b(days?|hours?)\b|\bimmediately\b", sent, re.I):
                deadlines.append({"rule": sent.strip(), "clause": c.cite()})
    return {"ok": True, "policy_id": policy_id, "claim_type": claim_type,
            "required_documents": valid[claim_type], "deadlines": deadlines,
            "source_clauses": [c.cite() for c in claims_cl + cond_cl],
            "steps": ["Notify the insurer or assistance team as the notification clause requires.",
                      "Collect every required document listed above.",
                      "Do not repair or dispose of damaged property before inspection if the policy says so.",
                      "Submit the claim form before the deadlines listed.",
                      "Keep copies of everything you send."],
            "disclaimer": DISCLAIMER}


def draft_claim(store: PolicyStore, policy_id: str, account_id: str, claim_type: str, incident_date: str,
                amount: float, description: str, documents: list[str], out_dir: str | Path | None = None) -> dict:
    chk = claim_checklist(store, policy_id, claim_type)
    if not chk.get("ok"):
        return chk
    missing = [d for d in chk["required_documents"] if d not in documents]
    lines = [f"CLAIM DRAFT (not submitted)", f"Policy: {policy_id}", f"Account: {account_id}",
             f"Claim type: {claim_type}", f"Incident date: {incident_date}", f"Amount claimed: {amount}",
             "", "Description:", description, "", "Documents attached: " + (", ".join(documents) or "none"),
             "Documents still missing: " + (", ".join(missing) or "none")]
    text = "\n".join(lines)
    path = None
    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        path = p / f"draft_{account_id}_{claim_type}.txt"
        path.write_text(text, encoding="utf-8")
    return {"ok": True, "draft": text, "missing_documents": missing, "ready_to_submit": not missing,
            "path": str(path) if path else None, "disclaimer": DISCLAIMER}


# ------------------------------------------------------------------ tier 2
def submit_claim_sandbox(insurer: SandboxInsurer, account_id: str, claim_type: str, amount: float,
                         incident_date: str, description: str, documents: list[str]) -> dict:
    return insurer.submit_claim(account_id, claim_type, amount, incident_date, description, documents)



