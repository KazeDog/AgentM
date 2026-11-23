from __future__ import annotations

import re
from typing import Any, List, Mapping


SYNTHETIC_EVIDENCE_METADATA_KEYS = {
    "source",
    "task",
    "query",
    "setid",
    "url",
    "published_date",
    "retrieved_at",
    "license_note",
    "source_note",
    "journal",
    "year",
    "section_error",
}


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def source_body_text(record: Mapping[str, Any]) -> str:
    """Return source-authored content without task/query indexing headers.

    The local Markdown corpora prepend routing metadata such as ``task`` and
    ``query`` to every source record. Those fields remain useful for search
    and traceability, but they must not satisfy semantic endpoint checks.
    Newly parsed records carry an explicit ``source_body``. The fallback
    removal keeps the same boundary for records already present in an index.
    """

    if "source_body" in record and record.get("source_body") is not None:
        return _normalize_text(record.get("source_body"))

    raw = "\n".join(
        str(record.get(key) or "")
        for key in ("text", "abstract", "summary")
        if record.get(key)
    )
    if not raw.strip():
        return ""

    # Legacy indexed Markdown is whitespace-normalized and may no longer
    # expose line boundaries or individual metadata fields. Its final routing
    # field is ``retrieved_at``; source-authored content begins afterward.
    prefix_start = re.search(r"(?:^|\s)-\s*source:\s*", raw, flags=re.I)
    prefix_end = re.search(r"(?:^|\s)-\s*retrieved_at:\s*\S+\s*", raw, flags=re.I)
    if prefix_start and prefix_end and prefix_start.start() < prefix_end.start():
        raw = raw[prefix_end.end() :]

    if "\n" in raw:
        kept_lines: List[str] = []
        for line in raw.splitlines():
            match = re.match(r"^\s*-\s*([A-Za-z0-9_ -]+):\s*", line)
            if match:
                key = match.group(1).strip().lower().replace(" ", "_").replace("-", "_")
                if key in SYNTHETIC_EVIDENCE_METADATA_KEYS:
                    continue
            kept_lines.append(line)
        cleaned = "\n".join(kept_lines)
    else:
        cleaned = raw

    # If legacy records retained structured metadata values, remove their
    # exact rendered fragments without guessing prose boundaries.
    for key in SYNTHETIC_EVIDENCE_METADATA_KEYS:
        value = record.get(key)
        if value is None or isinstance(value, (list, tuple, set, dict)):
            continue
        fragment = f"- {key}: {value}"
        cleaned = re.sub(re.escape(fragment), " ", cleaned, flags=re.I)

    dataset = str(record.get("dataset_name") or record.get("task") or "").strip()
    if dataset:
        generic_headers = (
            f"DailyMed Label Evidence for {dataset}",
            f"DailyMed DDI Label Evidence for {dataset}",
            f"PubMed Evidence for {dataset}",
            f"PubChem Evidence for {dataset}",
        )
        for header in generic_headers:
            cleaned = re.sub(re.escape(header), " ", cleaned, flags=re.I)
    return _normalize_text(cleaned)


def source_title_text(record: Mapping[str, Any]) -> str:
    """Return a source title only when it is not a corpus-routing heading."""

    title = _normalize_text(record.get("title"))
    lowered = title.lower()
    if not title:
        return ""
    if "/" in title and lowered.endswith((".md", ".json", ".jsonl", ".txt")):
        return ""
    if re.match(r"^(?:#\s*)?(?:dailymed|pubmed|pubchem|chembl).+evidence for\b", title, flags=re.I):
        return ""
    return title
