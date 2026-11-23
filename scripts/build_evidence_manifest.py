#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentm.runtime.compat import RESOURCE_ROOT
from agentm.runtime.retrieval.evidence_schema import endpoint_from_dataset, unique_strings


DEFAULT_EVIDENCE_JSONL = Path("/tmp/agentm_evidence_full_plus_backfill.jsonl")
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "evidence_targets"
DDI_DATASETS = {"drugbank", "twosides"}
SKIP_DATASETS = {"raw"}
MAX_COUNT_KEYS_PER_TARGET = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the full AgentM ADMET/DDI evidence target manifest and audit "
            "coverage against a structured evidence JSONL."
        )
    )
    parser.add_argument("--tdc-root", default=str(RESOURCE_ROOT / "data" / "tdc"))
    parser.add_argument("--evidence-jsonl", default=str(DEFAULT_EVIDENCE_JSONL) if DEFAULT_EVIDENCE_JSONL.exists() else "")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--include-admet-datasets", default="", help="Comma-separated allowlist; default is every non-DDI TDC dataset.")
    parser.add_argument("--include-ddi-datasets", default="DrugBank,TWOSIDES", help="Comma-separated DDI datasets to include.")
    parser.add_argument("--max-admet-per-dataset", type=int, default=0, help="Smoke cap per ADMET dataset; 0 means all rows.")
    parser.add_argument("--max-ddi-pairs", type=int, default=0, help="Smoke cap per DDI dataset; 0 means all unique pairs.")
    parser.add_argument("--sample-missing", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_docs = load_evidence_jsonl(Path(args.evidence_jsonl)) if args.evidence_jsonl else []
    evidence_index = build_evidence_index(evidence_docs)
    include_admet = _parse_allowlist(args.include_admet_datasets)
    include_ddi = _parse_allowlist(args.include_ddi_datasets)
    admet_targets = list(
        iter_admet_targets(
            Path(args.tdc_root),
            include_datasets=include_admet,
            max_per_dataset=args.max_admet_per_dataset,
        )
    )
    ddi_targets = list(
        iter_ddi_targets(
            Path(args.tdc_root),
            include_datasets=include_ddi,
            max_pairs_per_dataset=args.max_ddi_pairs,
        )
    )
    coverage_rows = [attach_coverage(target, evidence_index) for target in [*admet_targets, *ddi_targets]]
    manifest = build_manifest(
        tdc_root=Path(args.tdc_root),
        evidence_jsonl=Path(args.evidence_jsonl) if args.evidence_jsonl else None,
        admet_targets=admet_targets,
        ddi_targets=ddi_targets,
        coverage_rows=coverage_rows,
        sample_missing=args.sample_missing,
        caps={
            "max_admet_per_dataset": args.max_admet_per_dataset,
            "max_ddi_pairs": args.max_ddi_pairs,
        },
    )
    write_jsonl(output_dir / "admet_targets.jsonl", admet_targets)
    write_jsonl(output_dir / "ddi_pair_targets.jsonl", ddi_targets)
    write_jsonl(output_dir / "coverage_targets.jsonl", coverage_rows)
    write_json(output_dir / "evidence_target_manifest.json", manifest)
    write_json(output_dir / "coverage_summary.json", manifest["coverage_summary"])
    (output_dir / "coverage_summary.md").write_text(render_markdown_summary(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(output_dir),
                "admet_targets": len(admet_targets),
                "ddi_pair_targets": len(ddi_targets),
                "evidence_documents": len(evidence_docs),
                "coverage_status_counts": manifest["coverage_summary"]["status_counts"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def iter_admet_targets(
    tdc_root: Path,
    *,
    include_datasets: Optional[set[str]] = None,
    max_per_dataset: int = 0,
) -> Iterator[Dict[str, Any]]:
    for dataset_dir in iter_dataset_dirs(tdc_root):
        dataset_name = dataset_dir.name
        if include_datasets and dataset_name.lower() not in include_datasets:
            continue
        metadata = load_metadata(dataset_dir)
        if is_ddi_dataset(dataset_name, metadata):
            continue
        csv_path = normalized_csv_path(dataset_dir, metadata)
        if not csv_path:
            continue
        seen: Dict[str, Dict[str, Any]] = {}
        for row_idx, row in enumerate(read_csv_rows(csv_path), start=1):
            smiles = _first(row, "smiles", "Drug", "drug", "drug_smiles", "canonical_smiles")
            if not smiles:
                continue
            drug_id = _clean_identifier(_first(row, "drug_id", "Drug_ID", "id", "molecule_id"))
            drug_name = drug_id if _looks_like_drug_name(drug_id) else ""
            label = _first(row, "label", "Y", "y")
            endpoint = endpoint_from_dataset(dataset_name, task_type="admet_prediction")
            key = f"{dataset_name.lower()}::{_normalize_smiles_key(smiles) or drug_name.lower() or drug_id.lower()}"
            if key not in seen:
                target = {
                    "target_id": stable_id("admet", key),
                    "kind": "admet_case",
                    "task_type": "admet_prediction",
                    "dataset_name": dataset_name,
                    "endpoint": endpoint,
                    "smiles": smiles,
                    "drug_id": drug_id,
                    "drug_name": drug_name,
                    "entity_terms": target_terms([smiles, drug_id, drug_name]),
                    "label_counts": {},
                    "n_rows": 0,
                    "row_examples": [],
                    "source_csv": str(csv_path),
                    "metadata": {
                        "collection": _nested_get(metadata, ["task", "collection"]) or row.get("tdc_collection"),
                        "mode": _nested_get(metadata, ["task", "mode"]) or row.get("task_mode"),
                        "label_description": _nested_get(metadata, ["task", "label_description"]),
                    },
                }
                seen[key] = target
            target = seen[key]
            target["n_rows"] += 1
            if label != "":
                increment_limited_count(target["label_counts"], str(label))
            if len(target["row_examples"]) < 3:
                target["row_examples"].append(row_idx)
            if max_per_dataset and len(seen) >= max_per_dataset:
                break
        for target in seen.values():
            yield target


def iter_ddi_targets(
    tdc_root: Path,
    *,
    include_datasets: Optional[set[str]] = None,
    max_pairs_per_dataset: int = 0,
) -> Iterator[Dict[str, Any]]:
    for dataset_dir in iter_dataset_dirs(tdc_root):
        dataset_name = dataset_dir.name
        metadata = load_metadata(dataset_dir)
        if not is_ddi_dataset(dataset_name, metadata):
            continue
        if include_datasets and dataset_name.lower() not in include_datasets:
            continue
        csv_path = normalized_csv_path(dataset_dir, metadata)
        if not csv_path:
            continue
        seen: Dict[str, Dict[str, Any]] = {}
        for row_idx, row in enumerate(read_csv_rows(csv_path), start=1):
            drug_a_smiles = _first(row, "drug_a_smiles", "Drug1", "drug1_smiles")
            drug_b_smiles = _first(row, "drug_b_smiles", "Drug2", "drug2_smiles")
            drug_a_name = _clean_identifier(_first(row, "drug_a_name", "Drug1_Name", "drug_a"))
            drug_b_name = _clean_identifier(_first(row, "drug_b_name", "Drug2_Name", "drug_b"))
            drug_a_id = _clean_identifier(_first(row, "drug_a_id", "Drug1_ID", "drug1_id"))
            drug_b_id = _clean_identifier(_first(row, "drug_b_id", "Drug2_ID", "drug2_id"))
            if not (drug_a_smiles or drug_a_name or drug_a_id) or not (drug_b_smiles or drug_b_name or drug_b_id):
                continue
            left_key = _best_pair_key(drug_a_id, drug_a_name, drug_a_smiles)
            right_key = _best_pair_key(drug_b_id, drug_b_name, drug_b_smiles)
            pair_key = "__".join(sorted([left_key, right_key]))
            label = _first(row, "label", "Y", "y", "label_original")
            if pair_key not in seen:
                target = {
                    "target_id": stable_id("ddi", f"{dataset_name.lower()}::{pair_key}"),
                    "kind": "ddi_pair",
                    "task_type": "ddi_prediction",
                    "dataset_name": dataset_name,
                    "endpoint": "ddi",
                    "pair_key": pair_key,
                    "drug_a": {
                        "name": drug_a_name,
                        "drug_id": drug_a_id,
                        "smiles": drug_a_smiles,
                        "entity_terms": target_terms([drug_a_name, drug_a_id, drug_a_smiles]),
                    },
                    "drug_b": {
                        "name": drug_b_name,
                        "drug_id": drug_b_id,
                        "smiles": drug_b_smiles,
                        "entity_terms": target_terms([drug_b_name, drug_b_id, drug_b_smiles]),
                    },
                    "entity_terms": target_terms([drug_a_name, drug_a_id, drug_a_smiles, drug_b_name, drug_b_id, drug_b_smiles, pair_key]),
                    "label_counts": {},
                    "interaction_type_counts": {},
                    "n_rows": 0,
                    "row_examples": [],
                    "source_csv": str(csv_path),
                    "metadata": {
                        "label_mode": metadata.get("label_mode"),
                        "description": metadata.get("description"),
                    },
                }
                seen[pair_key] = target
            target = seen[pair_key]
            target["n_rows"] += 1
            if label != "":
                increment_limited_count(target["label_counts"], str(label))
            interaction_type = _first(row, "interaction_type", "mechanism_text")
            if interaction_type != "":
                increment_limited_count(target["interaction_type_counts"], str(interaction_type))
            if len(target["row_examples"]) < 3:
                target["row_examples"].append(row_idx)
            if max_pairs_per_dataset and len(seen) >= max_pairs_per_dataset:
                break
        for target in seen.values():
            yield target


def attach_coverage(target: Mapping[str, Any], evidence_docs: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> Dict[str, Any]:
    evidence_index = evidence_docs if isinstance(evidence_docs, Mapping) and "_agentm_evidence_index" in evidence_docs else build_evidence_index(evidence_docs)
    endpoint = str(target.get("endpoint") or "").lower()
    endpoint_entries = evidence_index["entries_by_endpoint"].get(endpoint, [])
    if target.get("kind") == "ddi_pair":
        entity_docs, single_side_docs = _ddi_entity_docs(target, endpoint_entries)
    else:
        terms = set(target.get("entity_terms", []) or [])
        entity_docs = [doc for doc, doc_terms in endpoint_entries if terms and terms.intersection(doc_terms)]
        single_side_docs = []
    entity_endpoint_docs = [
        doc
        for doc in entity_docs
        if bool(doc.get("compound_specific")) and bool(doc.get("endpoint_specific"))
    ]
    entity_background_docs = [doc for doc in entity_docs if doc not in entity_endpoint_docs]
    endpoint_background_docs = evidence_index["endpoint_background_by_endpoint"].get(endpoint, [])
    if entity_endpoint_docs:
        status = "covered"
    elif entity_docs:
        status = "entity_background_only"
    elif endpoint_background_docs:
        status = "endpoint_background_only"
    else:
        status = "missing"
    return {
        "target_id": target.get("target_id"),
        "kind": target.get("kind"),
        "dataset_name": target.get("dataset_name"),
        "endpoint": endpoint,
        "status": status,
        "entity_documents": len(entity_docs),
        "entity_endpoint_documents": len(entity_endpoint_docs),
        "entity_background_documents": len(entity_background_docs),
        "endpoint_background_documents": len(endpoint_background_docs),
        "single_side_documents": len(single_side_docs),
        "source_counts": dict(Counter(str(doc.get("source") or "unknown") for doc in entity_docs)),
        "scope_counts": dict(Counter(str(doc.get("evidence_scope") or "unknown") for doc in entity_docs)),
        "examples": coverage_examples([*entity_endpoint_docs, *entity_background_docs, *endpoint_background_docs]),
    }


def build_manifest(
    *,
    tdc_root: Path,
    evidence_jsonl: Optional[Path],
    admet_targets: Sequence[Mapping[str, Any]],
    ddi_targets: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
    sample_missing: int,
    caps: Mapping[str, int],
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    summary = coverage_summary(
        admet_targets=admet_targets,
        ddi_targets=ddi_targets,
        coverage_rows=coverage_rows,
        sample_missing=sample_missing,
    )
    return {
        "created_at": now,
        "tdc_root": str(tdc_root),
        "evidence_jsonl": str(evidence_jsonl) if evidence_jsonl else None,
        "caps": dict(caps),
        "target_files": {
            "admet": "admet_targets.jsonl",
            "ddi": "ddi_pair_targets.jsonl",
            "coverage": "coverage_targets.jsonl",
        },
        "coverage_summary": summary,
    }


def coverage_summary(
    *,
    admet_targets: Sequence[Mapping[str, Any]],
    ddi_targets: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
    sample_missing: int,
) -> Dict[str, Any]:
    status_counts = Counter(str(row.get("status") or "unknown") for row in coverage_rows)
    by_kind: Dict[str, Counter[str]] = defaultdict(Counter)
    by_dataset: Dict[str, Counter[str]] = defaultdict(Counter)
    by_endpoint: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in coverage_rows:
        status = str(row.get("status") or "unknown")
        by_kind[str(row.get("kind") or "unknown")][status] += 1
        by_dataset[str(row.get("dataset_name") or "unknown")][status] += 1
        by_endpoint[str(row.get("endpoint") or "unknown")][status] += 1
    missing_like = [row for row in coverage_rows if row.get("status") != "covered"]
    return {
        "n_targets": len(coverage_rows),
        "n_admet_targets": len(admet_targets),
        "n_ddi_pair_targets": len(ddi_targets),
        "status_counts": dict(status_counts),
        "status_by_kind": {key: dict(value) for key, value in sorted(by_kind.items())},
        "status_by_dataset": {key: dict(value) for key, value in sorted(by_dataset.items())},
        "status_by_endpoint": {key: dict(value) for key, value in sorted(by_endpoint.items())},
        "missing_or_partial_examples": missing_like[: max(sample_missing, 0)],
    }


def load_evidence_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    docs: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            docs.append(payload)
    return docs


def build_evidence_index(evidence_docs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    entries_by_endpoint: Dict[str, List[Tuple[Mapping[str, Any], set[str]]]] = defaultdict(list)
    endpoint_background_by_endpoint: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for doc in evidence_docs:
        endpoint = str(doc.get("endpoint") or "").lower()
        terms = document_terms(doc)
        entries_by_endpoint[endpoint].append((doc, terms))
        if bool(doc.get("endpoint_specific")) and not bool(doc.get("compound_specific")):
            endpoint_background_by_endpoint[endpoint].append(doc)
    return {
        "_agentm_evidence_index": True,
        "entries_by_endpoint": entries_by_endpoint,
        "endpoint_background_by_endpoint": endpoint_background_by_endpoint,
    }


def iter_dataset_dirs(tdc_root: Path) -> Iterator[Path]:
    for dataset_dir in sorted(Path(tdc_root).iterdir()):
        if dataset_dir.name.lower() in SKIP_DATASETS:
            continue
        if dataset_dir.is_dir():
            yield dataset_dir


def load_metadata(dataset_dir: Path) -> Dict[str, Any]:
    path = dataset_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def is_ddi_dataset(dataset_name: str, metadata: Mapping[str, Any]) -> bool:
    return (
        dataset_name.lower() in DDI_DATASETS
        or str(metadata.get("task_type") or "").lower() == "ddi_prediction"
        or "ddi" in str(metadata.get("tdc_task") or "").lower()
    )


def normalized_csv_path(dataset_dir: Path, metadata: Mapping[str, Any]) -> Optional[Path]:
    files = metadata.get("files") if isinstance(metadata.get("files"), Mapping) else {}
    for key in ["normalized", "full"]:
        candidate = files.get(key)
        if candidate:
            resolved = resolve_metadata_path(candidate, dataset_dir)
            if resolved.exists():
                return resolved
    direct = dataset_dir / f"{dataset_dir.name}.csv"
    if direct.exists():
        return direct
    for candidate in sorted(dataset_dir.glob("*.csv")):
        if candidate.name.lower() in {"train.csv", "valid.csv", "test.csv"}:
            continue
        if "raw" in candidate.name.lower() or "invalid" in candidate.name.lower():
            continue
        return candidate
    return None


def resolve_metadata_path(value: Any, dataset_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    text = str(value)
    if text.startswith("data/tdc/"):
        return RESOURCE_ROOT / text
    return dataset_dir / path


def read_csv_rows(path: Path) -> Iterator[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield {str(key or ""): str(value or "").strip() for key, value in row.items()}


def document_terms(doc: Mapping[str, Any]) -> set[str]:
    values: List[Any] = []
    for key in [
        "entity_terms",
        "drug_names",
        "compound_names",
        "compound_synonyms",
        "compound_cids",
        "chembl_ids",
        "canonical_smiles",
        "pair_drug_names",
        "pair_drug_ids",
        "ddi_pair_key",
    ]:
        raw = doc.get(key)
        if isinstance(raw, str):
            values.append(raw)
        else:
            values.extend(raw or [])
    return set(target_terms(values))


def target_terms(values: Iterable[Any]) -> List[str]:
    expanded: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        if re.match(r"^drug\s*\d+$", text, flags=re.I):
            continue
        expanded.append(text)
        cleaned = _clean_identifier(text)
        if cleaned and cleaned != text:
            expanded.append(cleaned)
        compact = re.sub(r"[^A-Za-z0-9]+", "", cleaned or text)
        if compact:
            expanded.append(compact)
        cid = re.match(r"^CID0*(\d+)$", compact, flags=re.I)
        if cid:
            expanded.append(cid.group(1))
        if re.match(r"^\d+\.0$", text):
            expanded.append(text[:-2])
    return unique_strings(expanded, lower=True, limit=160)


def coverage_examples(docs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    examples = []
    for doc in docs[:3]:
        examples.append(
            {
                "chunk_id": doc.get("chunk_id"),
                "source": doc.get("source"),
                "scope": doc.get("evidence_scope"),
                "title": doc.get("title"),
                "endpoint_specific": bool(doc.get("endpoint_specific")),
                "compound_specific": bool(doc.get("compound_specific")),
            }
        )
    return examples


def increment_limited_count(counter: Dict[str, int], key: str, *, max_keys: int = MAX_COUNT_KEYS_PER_TARGET) -> None:
    if key in counter or len(counter) < max_keys:
        counter[key] = int(counter.get(key, 0)) + 1
    else:
        counter["__other__"] = int(counter.get("__other__", 0)) + 1


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def render_markdown_summary(manifest: Mapping[str, Any]) -> str:
    summary = manifest["coverage_summary"]
    lines = [
        "# AgentM Evidence Target Coverage",
        "",
        f"- Targets: {summary['n_targets']}",
        f"- ADMET targets: {summary['n_admet_targets']}",
        f"- DDI pair targets: {summary['n_ddi_pair_targets']}",
        f"- Status counts: `{json.dumps(summary['status_counts'], ensure_ascii=False)}`",
        "",
        "## Dataset Status",
        "",
        "| Dataset | Covered | Entity Background | Endpoint Background | Missing |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset, counts in summary["status_by_dataset"].items():
        lines.append(
            "| {dataset} | {covered} | {entity_background_only} | {endpoint_background_only} | {missing} |".format(
                dataset=dataset,
                covered=counts.get("covered", 0),
                entity_background_only=counts.get("entity_background_only", 0),
                endpoint_background_only=counts.get("endpoint_background_only", 0),
                missing=counts.get("missing", 0),
            )
        )
    return "\n".join(lines) + "\n"


def _ddi_entity_docs(
    target: Mapping[str, Any],
    endpoint_entries: Sequence[Tuple[Mapping[str, Any], set[str]]],
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    pair_key = str(target.get("pair_key") or "").lower()
    drug_a_terms = set(((target.get("drug_a") or {}).get("entity_terms") or []))
    drug_b_terms = set(((target.get("drug_b") or {}).get("entity_terms") or []))
    pair_docs: List[Mapping[str, Any]] = []
    single_side_docs: List[Mapping[str, Any]] = []
    for doc, doc_terms in endpoint_entries:
        doc_pair_key = str(doc.get("ddi_pair_key") or "").lower()
        matches_pair_key = bool(pair_key and (doc_pair_key == pair_key or doc_pair_key in pair_key or pair_key in doc_pair_key))
        matches_a = bool(drug_a_terms.intersection(doc_terms))
        matches_b = bool(drug_b_terms.intersection(doc_terms))
        if matches_pair_key or (matches_a and matches_b):
            pair_docs.append(doc)
        elif matches_a or matches_b:
            single_side_docs.append(doc)
    return pair_docs, single_side_docs


def _parse_allowlist(value: str) -> Optional[set[str]]:
    items = {item.strip().lower() for item in str(value or "").split(",") if item.strip()}
    return items or None


def _first(row: Mapping[str, Any], *keys: str) -> str:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
        value = lowered.get(key.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _clean_identifier(value: Any) -> str:
    text = str(value or "").strip()
    if re.match(r"^\d+\.0$", text):
        return text[:-2]
    return text


def _looks_like_drug_name(value: str) -> bool:
    text = str(value or "").strip()
    if not text or _looks_like_smiles(text):
        return False
    if re.match(r"^drug\s*\d+$", text, flags=re.I):
        return False
    if re.match(r"^\d+(\.\d+)?$", text):
        return False
    if re.match(r"^(DB|CID)\d+", text, flags=re.I):
        return False
    return True


def _looks_like_smiles(value: str) -> bool:
    text = str(value or "").strip()
    if not text or " " in text:
        return False
    return any(char in text for char in "=#@[]()/\\") or bool(re.search(r"[cnops]\d", text, flags=re.I))


def _normalize_smiles_key(smiles: str) -> str:
    return re.sub(r"\s+", "", str(smiles or "").strip()).lower()


def _best_pair_key(drug_id: str, drug_name: str, smiles: str) -> str:
    for value in [drug_id, drug_name, smiles]:
        cleaned = _clean_identifier(value)
        if cleaned:
            return re.sub(r"\s+", "", cleaned).lower()
    return "unknown"


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _nested_get(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


if __name__ == "__main__":
    raise SystemExit(main())
