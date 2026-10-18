"""Policy loading, clause chunking and BM25 retrieval (no external services).

BM25 is the default retriever because the cubic.ai evaluation found it beat dense
retrieval inside a single document (R@1 0.816 vs 0.659). Dense/hybrid retrieval
can be added on top by wrapping `PolicyStore.search`.
"""
from __future__ import annotations
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

POLICY_DIR = Path(__file__).parent / "policies"
KINDS = {"coverage", "exclusion", "condition", "limit", "claims", "definition"}
STOP = set("a an the of to in on for and or is are be if any it its this that with by as at from "
           "do does can i my me we you your our not no what which when how there".split())

_POLICY_RE = re.compile(r"^#\s*POLICY\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*(\w+)\s*$")
_CLAUSE_RE = re.compile(r"^###\s*\[([^\]]+)\]\s*\|\s*(\d+)\s*\|\s*(\w+)\s*\|\s*(.+?)\s*$")


@dataclass
class Clause:
    clause_id: str
    policy_id: str
    page: int
    kind: str
    title: str
    text: str
    policy_title: str = ""
    policy_type: str = ""

    def cite(self) -> str:
        return f"{self.policy_id} {self.clause_id} (p.{self.page})"


@dataclass
class Policy:
    policy_id: str
    title: str
    ptype: str
    clauses: list[Clause] = field(default_factory=list)


SYNONYMS = {"stolen": "theft", "steal": "theft", "stole": "theft", "robbed": "theft", "robbery": "theft",
            "bag": "baggage", "luggage": "baggage", "suitcase": "baggage", "hurt": "injury",
            "injured": "injury", "crash": "accident", "collision": "accident", "sick": "illness",
            "ill": "illness", "disease": "illness", "hospital": "hospitalisation",
            "hospitalization": "hospitalisation", "admitted": "hospitalisation", "cancelled": "cancel",
            "cancelling": "cancel", "cancellation": "cancel", "flight": "transport", "delayed": "delay",
            "drunk": "alcohol", "drinking": "alcohol", "drove": "driven", "driving": "driven",
            "scuba": "scuba", "dive": "scuba", "diving": "scuba", "ski": "winter", "skiing": "winter",
            "snowboarding": "winter", "car": "vehicle", "repair": "repair", "reimburse": "pay",
            "reimbursement": "pay", "payout": "pay", "paid": "pay", "covered": "cover", "covers": "cover"}


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for t in toks:
        t = SYNONYMS.get(t, t)
        if t in STOP:
            continue
        # light stemming so "stolen/theft", "claims/claim" etc. line up a little
        if len(t) > 4 and t.endswith("ies"):
            t = t[:-3] + "y"
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        elif len(t) > 5 and t.endswith("ing"):
            t = t[:-3]
        elif len(t) > 4 and t.endswith("ed"):
            t = t[:-2]
        out.append(t)
    return out


def parse_policy_file(path: Path) -> Policy:
    policy = None
    cur: Clause | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        m = _POLICY_RE.match(line)
        if m:
            policy = Policy(m.group(1).strip(), m.group(2).strip(), m.group(3).strip())
            continue
        m = _CLAUSE_RE.match(line)
        if m:
            if policy is None:
                raise ValueError(f"{path.name}: clause before POLICY header")
            kind = m.group(3).strip().lower()
            if kind not in KINDS:
                raise ValueError(f"{path.name}: unknown clause kind {kind!r}")
            cur = Clause(m.group(1).strip(), policy.policy_id, int(m.group(2)), kind,
                         m.group(4).strip(), "", policy.title, policy.ptype)
            policy.clauses.append(cur)
            continue
        if cur is not None and line and not line.startswith(">") and not line.startswith("#"):
            cur.text = (cur.text + " " + line.strip()).strip()
    if policy is None:
        raise ValueError(f"{path.name}: no POLICY header")
    return policy


class PolicyStore:
    def __init__(self, policies: list[Policy], k1: float = 1.5, b: float = 0.75):
        self.policies = {p.policy_id: p for p in policies}
        self.k1, self.b = k1, b
        self.clauses: list[Clause] = [c for p in policies for c in p.clauses]
        self._index()

    @classmethod
    def load(cls, directory: str | Path | None = None) -> "PolicyStore":
        d = Path(directory) if directory else POLICY_DIR
        files = sorted(d.glob("*.md"))
        if not files:
            raise FileNotFoundError(f"no policy files in {d}")
        return cls([parse_policy_file(f) for f in files])

    def _index(self) -> None:
        self._tf = [Counter(tokenize(c.title + " " + c.title + " " + c.text)) for c in self.clauses]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self.clauses)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def get(self, policy_id: str, clause_id: str) -> Clause | None:
        p = self.policies.get(policy_id)
        if not p:
            return None
        return next((c for c in p.clauses if c.clause_id == clause_id), None)

    def list_policies(self) -> list[dict]:
        return [{"policy_id": p.policy_id, "title": p.title, "type": p.ptype,
                 "clauses": len(p.clauses)} for p in self.policies.values()]

    def search(self, query: str, top_k: int = 5, policy_id: str | None = None,
               kinds: set[str] | None = None) -> list[tuple[Clause, float]]:
        q = tokenize(query)
        scored: list[tuple[Clause, float]] = []
        for i, c in enumerate(self.clauses):
            if policy_id and c.policy_id != policy_id:
                continue
            if kinds and c.kind not in kinds:
                continue
            tf, dl = self._tf[i], self._len[i]
            s = 0.0
            for t in set(q):
                f = tf.get(t, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avg or 1))
                s += self._idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            if s > 0:
                scored.append((c, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]
