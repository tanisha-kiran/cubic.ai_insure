"""Convert a text-bearing PDF policy into the Markdown clause format PolicyStore consumes.

The existing cubic.ai PDF normalizer uses PyMuPDF and preserves page text/page images. This
module uses PyMuPDF's same page text extraction for stable clause citations; scanned pages with
no embedded text are rejected with a clear error until OCR output is available.
"""
from __future__ import annotations

import re
from pathlib import Path

import fitz

KINDS = {"coverage", "exclusion", "condition", "limit", "claims", "definition"}
_HEADING = re.compile(r"^\s*((?:\d+\.)*\d+[.)]?)\s+(.{2,160}?)\s*$")


def classify_clause(title: str, text: str) -> str:
    value = f"{title} {text}".lower()
    if re.search(r"\b(exclusion|excluded|we do not cover|not covered|will not pay|does not cover)\b", value):
        return "exclusion"
    if re.search(r"\b(waiting period|subject to|condition|must be|provided that|only if)\b", value):
        return "condition"
    if re.search(r"\b(documents required|claim form|claims?|notify|notification|within \d+ days)\b", value):
        return "claims"
    if re.search(r"\b(limit|maximum|sum insured|deductible|excess|up to \$|not exceed)\b", value):
        return "limit"
    if re.search(r"\b(means|defined as|definition)\b", value):
        return "definition"
    return "coverage"


def _page_lines(pdf_path: Path) -> list[tuple[int, str]]:
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise ValueError(f"Could not open policy PDF: {exc}") from exc
    if not len(doc):
        raise ValueError("Policy PDF has no pages")
    lines = []
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        lines.extend((page_number, line.strip()) for line in text.splitlines() if line.strip())
    if not lines:
        raise ValueError("No embedded text found; OCR this scanned PDF before importing")
    return lines


def parse_policy_pdf(pdf_path: str | Path, policy_id: str, title: str, ptype: str) -> dict:
    """Return a parse_policy_file-compatible policy and clause review records."""
    path = Path(pdf_path)
    if not re.fullmatch(r"[A-Z0-9_-]{2,40}", policy_id):
        raise ValueError("policy_id must contain 2-40 uppercase letters, digits, _ or -")
    if ptype not in {"travel", "health", "motor"}:
        raise ValueError("ptype must be travel, health or motor")
    title = title.strip()
    if not title or len(title) > 120 or "|" in title or "\n" in title:
        raise ValueError("title must contain 1-120 characters and cannot contain | or newlines")

    source = _page_lines(path)
    clauses: list[dict] = []
    current: dict | None = None
    fallback_index = 0
    for page, line in source:
        match = _HEADING.match(line)
        if match:
            if current:
                clauses.append(current)
            section, heading = match.groups()
            section = section.rstrip(".)")
            current = {"clause_id": f"{policy_id}-{section}", "page": page,
                       "title": heading.strip(" .:"), "text_parts": []}
        elif current is None:
            fallback_index += 1
            current = {"clause_id": f"{policy_id}-{fallback_index}", "page": page,
                       "title": line[:100].strip(" .:"), "text_parts": [line]}
        else:
            current["text_parts"].append(line)
    if current:
        clauses.append(current)

    # Merge repeated heading numbers instead of emitting duplicate clause IDs.
    unique: list[dict] = []
    seen: set[str] = set()
    for clause in clauses:
        if clause["clause_id"] in seen:
            unique[-1]["text_parts"].extend([clause["title"], *clause["text_parts"]])
            continue
        seen.add(clause["clause_id"])
        unique.append(clause)

    rows = []
    for clause in unique:
        body = " ".join(clause["text_parts"]).strip()
        if not body:
            body = clause["title"]
        kind = classify_clause(clause["title"], body)
        rows.append({"clause_id": clause["clause_id"], "page": clause["page"],
                     "kind": kind, "title": clause["title"], "text": body})
    return {"policy_id": policy_id, "title": title, "type": ptype, "clauses": rows}


def write_policy_markdown(parsed: dict, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# POLICY | {parsed['policy_id']} | {parsed['title']} | {parsed['type']}", ""]
    for clause in parsed["clauses"]:
        if clause["kind"] not in KINDS:
            raise ValueError(f"Unknown clause kind: {clause['kind']}")
        lines.extend([f"### [{clause['clause_id']}] | {clause['page']} | {clause['kind']} | {clause['title']}",
                      clause["text"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def ingest_policy_pdf(pdf_path: str | Path, output_path: str | Path, policy_id: str, title: str, ptype: str) -> dict:
    parsed = parse_policy_pdf(pdf_path, policy_id, title, ptype)
    write_policy_markdown(parsed, output_path)
    pdf = fitz.open(pdf_path)
    try:
        for page_number, page in enumerate(pdf, start=1):
            page.get_pixmap(dpi=120, alpha=False).save(Path(output_path).parent / f"{policy_id}_page_{page_number}.png")
    finally:
        pdf.close()
    return parsed

