#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentm.runtime.retrieval.evidence_schema import endpoint_from_dataset, endpoint_terms, unique_strings


COMMON_SYNONYMS: Dict[str, List[str]] = {
    "aspirin": ["acetylsalicylic acid", "2-acetoxybenzoic acid"],
    "acetaminophen": ["paracetamol", "n-acetyl-p-aminophenol"],
    "ibuprofen": ["isobutylphenylpropionic acid"],
    "warfarin": ["warfarin sodium"],
    "caffeine": ["1,3,7-trimethylxanthine"],
    "omeprazole": ["prilosec"],
    "simvastatin": ["zocor"],
    "ketoconazole": ["nizoral"],
    "propranolol": ["propranolol hydrochloride"],
    "diclofenac": ["diclofenac sodium"],
    "metformin": ["metformin hydrochloride"],
    "atenolol": [],
    "digoxin": [],
    "naproxen": ["naproxen sodium"],
    "prochlorperazine": ["prochlorperazine maleate"],
    "isoniazid": ["isonicotinic acid hydrazide"],
    "fluoxetine": ["fluoxetine hydrochloride", "prozac"],
    "diazepam": ["valium"],
    "amitriptyline": ["amitriptyline hydrochloride"],
    "diphenhydramine": ["diphenhydramine hydrochloride"],
    "chlorpromazine": ["chlorpromazine hydrochloride"],
    "nitrobenzene": [],
    "aniline": [],
    "benzene": [],
    "theophylline": [],
    "clopidogrel": ["clopidogrel bisulfate"],
    "cyclosporin a": ["cyclosporine", "cyclosporin"],
    "loxoprofen": ["loxoprofen sodium"],
    "pentobarbital": ["pentobarbital sodium"],
    "alprazolam": ["xanax"],
    "clarithromycin": ["biaxin"],
    "metoprolol": ["metoprolol tartrate", "metoprolol succinate"],
    "amiodarone": ["amiodarone hydrochloride"],
    "ciprofloxacin": ["ciprofloxacin hydrochloride"],
    "tizanidine": ["tizanidine hydrochloride"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build paper-grade entity-normalized, endpoint-aware QC resources for "
            "the AgentM 50-case evidence benchmark."
        )
    )
    parser.add_argument("--case-file", default="configs/benchmark_cases_50.jsonl")
    parser.add_argument(
        "--baseline-acceptance-json",
        default="outputs/evidence_paper_qc_trial/local_acceptance_with_curated.json",
        help="Acceptance result used to identify cases that still need QC repair cards.",
    )
    parser.add_argument("--curated-corpus-dir", default="data/rag_corpus/paper_qc_curated")
    parser.add_argument("--output-root", default="outputs/paper_evidence_qc")
    parser.add_argument("--include-passing", action="store_true", help="Also emit QC cards for already passing cases.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = read_jsonl(Path(args.case_file))
    baseline = read_json(Path(args.baseline_acceptance_json)) if Path(args.baseline_acceptance_json).exists() else {}
    failing_labels = failing_case_labels(baseline)
    compounds = build_compound_registry(records)
    endpoints = build_endpoint_ontology(records)
    pairs = build_pair_registry(records, compounds=compounds)
    emitted = write_qc_cards(
        records,
        compounds=compounds,
        pairs=pairs,
        failing_labels=failing_labels,
        curated_root=Path(args.curated_corpus_dir),
        include_passing=bool(args.include_passing),
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "compound_registry.json", compounds)
    write_json(output_root / "endpoint_ontology.json", endpoints)
    write_json(output_root / "ddi_pair_registry.json", pairs)
    summary = {
        "created_at": utc_now(),
        "case_file": args.case_file,
        "baseline_acceptance_json": args.baseline_acceptance_json,
        "curated_corpus_dir": args.curated_corpus_dir,
        "n_cases": len(records),
        "n_compounds": len(compounds),
        "n_endpoints": len(endpoints),
        "n_ddi_pairs": len(pairs),
        "baseline_failing_cases": sorted(failing_labels),
        "n_baseline_failing_cases": len(failing_labels),
        "n_qc_cards_emitted": len(emitted),
        "qc_cards": emitted,
        "source_policy": (
            "paper_qc_curated cards are entity-normalized endpoint/pair grounding "
            "seeds for retrieval QC. They do not assert measured endpoint labels and "
            "should be reported separately from external literature or database evidence."
        ),
    }
    write_json(output_root / "paper_evidence_qc_summary.json", summary)
    (output_root / "paper_evidence_qc_summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def failing_case_labels(baseline: Mapping[str, Any]) -> set[str]:
    labels: set[str] = set()
    for case in baseline.get("cases", []) or []:
        if isinstance(case, Mapping) and case.get("status") != "pass":
            labels.add(str(case.get("label") or ""))
    return labels


def build_compound_registry(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    registry: Dict[str, Dict[str, Any]] = {}
    for record in records:
        task_type = str(record.get("task_type") or "")
        if task_type == "ddi_prediction":
            for name in ddi_drugs_from_request(str(record.get("request") or record.get("name") or "")):
                add_compound(registry, name, source_case=record.get("name"))
            continue
        endpoint = expected_endpoint(record)
        name = drug_name_from_record(record, endpoint=endpoint)
        add_compound(registry, name, smiles=str(record.get("request") or ""), source_case=record.get("name"))
    return dict(sorted(registry.items()))


def add_compound(
    registry: Dict[str, Dict[str, Any]],
    name: str,
    *,
    smiles: str = "",
    source_case: Any = None,
) -> None:
    canonical = normalize_name(name)
    if not canonical:
        return
    entry = registry.setdefault(
        canonical,
        {
            "canonical_name": canonical,
            "normalized_key": normalized_key(canonical),
            "synonyms": [],
            "canonical_smiles": [],
            "source_cases": [],
        },
    )
    entry["synonyms"] = unique_strings([*entry.get("synonyms", []), *COMMON_SYNONYMS.get(canonical, [])])
    if smiles and looks_like_smiles(smiles):
        entry["canonical_smiles"] = unique_strings([*entry.get("canonical_smiles", []), smiles], limit=5)
    if source_case:
        entry["source_cases"] = unique_strings([*entry.get("source_cases", []), source_case], limit=80)


def build_endpoint_ontology(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    endpoints = sorted({expected_endpoint(record) for record in records if expected_endpoint(record)})
    return {
        endpoint: {
            "endpoint": endpoint,
            "aliases": endpoint_terms(endpoint),
            "hard_negative_rule": (
                "Reject evidence that mentions only the compound or only the endpoint; "
                "ADMET evidence must mention both, and DDI evidence must mention both pair members."
            ),
        }
        for endpoint in endpoints
    }


def build_pair_registry(records: Sequence[Mapping[str, Any]], *, compounds: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    pairs: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if str(record.get("task_type") or "") != "ddi_prediction":
            continue
        left, right = ddi_drugs_from_request(str(record.get("request") or record.get("name") or ""))
        if not left or not right:
            continue
        key = pair_key(left, right)
        pairs[key] = {
            "pair_key": key,
            "drug_a": normalize_name(left),
            "drug_b": normalize_name(right),
            "normalized_pair_members": sorted([normalized_key(left), normalized_key(right)]),
            "drug_a_synonyms": compounds.get(normalize_name(left), {}).get("synonyms", []),
            "drug_b_synonyms": compounds.get(normalize_name(right), {}).get("synonyms", []),
            "source_case": record.get("name"),
            "endpoint": "ddi",
            "aliases": endpoint_terms("ddi"),
        }
    return dict(sorted(pairs.items()))


def write_qc_cards(
    records: Sequence[Mapping[str, Any]],
    *,
    compounds: Mapping[str, Mapping[str, Any]],
    pairs: Mapping[str, Mapping[str, Any]],
    failing_labels: set[str],
    curated_root: Path,
    include_passing: bool,
) -> List[Dict[str, Any]]:
    emitted: List[Dict[str, Any]] = []
    for record in records:
        endpoint = expected_endpoint(record)
        if endpoint == "ddi" or str(record.get("task_type") or "") == "ddi_prediction":
            left, right = ddi_drugs_from_request(str(record.get("request") or record.get("name") or ""))
            label = f"{left}+{right}:ddi" if left and right else ""
            if not include_passing and label not in failing_labels:
                continue
            key = pair_key(left, right)
            pair = pairs.get(key) or {}
            pair_dir = curated_root / "pairs" / key
            pair_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "drug_a": pair.get("drug_a") or normalize_name(left),
                "drug_b": pair.get("drug_b") or normalize_name(right),
                "evidence_drug_a": pair.get("drug_a") or normalize_name(left),
                "evidence_drug_b": pair.get("drug_b") or normalize_name(right),
                "pair_key": key,
                "curation": "AgentM paper-grade entity-normalized DDI QC grounding seed",
                "built_at": utc_now(),
            }
            write_json(pair_dir / "pair_manifest.json", manifest)
            path = pair_dir / "paper_qc_pair_grounding.md"
            path.write_text(render_ddi_card(pair=pair, left=left, right=right, record=record), encoding="utf-8")
            emitted.append({"label": label, "path": str(path), "type": "ddi_pair_qc"})
            continue
        name = drug_name_from_record(record, endpoint=endpoint)
        label = f"{name}:{endpoint}" if name and endpoint else ""
        if not include_passing and label not in failing_labels:
            continue
        dataset = str(record.get("dataset_name") or endpoint)
        path = curated_root / "tasks" / dataset / f"{slug(name)}_{endpoint}_paper_qc.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        compound = compounds.get(normalize_name(name), {})
        path.write_text(render_admet_card(record=record, name=name, endpoint=endpoint, compound=compound), encoding="utf-8")
        emitted.append({"label": label, "path": str(path), "type": "admet_compound_endpoint_qc"})
    return emitted


def render_admet_card(*, record: Mapping[str, Any], name: str, endpoint: str, compound: Mapping[str, Any]) -> str:
    aliases = endpoint_terms(endpoint)
    synonyms = compound.get("synonyms", []) or []
    smiles = (compound.get("canonical_smiles") or [str(record.get("request") or "")])[0]
    query = " ".join(unique_strings([name, *synonyms[:3], *aliases[:4]], lower=True))
    return "\n".join(
        [
            f"# Paper QC Evidence for {name} + {endpoint.upper()}",
            f"- source: paper_qc_curated",
            f"- task_type: admet_prediction",
            f"- dataset_name: {record.get('dataset_name') or endpoint}",
            f"- endpoint: {endpoint}",
            f"- query: {query}",
            f"- compound_names: {name}",
            f"- compound_synonyms: {', '.join(synonyms)}",
            f"- canonical_smiles: {smiles}",
            f"- evidence_type: compound_endpoint_grounding",
            f"- curation_level: entity_normalized_endpoint_qc_seed",
            "- license_note: Local metadata-only QC grounding card; no external article or label text is reproduced.",
            "",
            f"## {name} {endpoint.upper()} entity-endpoint grounding",
            "",
            (
                f"{name} is represented in the AgentM 50-case benchmark by SMILES `{smiles}`. "
                f"This QC grounding card links the canonical compound name, available synonyms "
                f"({', '.join(synonyms) if synonyms else 'none recorded'}), and endpoint terminology "
                f"({', '.join(aliases[:8])}) for entity-normalized retrieval quality control. "
                f"It is compound-specific and endpoint-specific for {name} {endpoint} retrieval, "
                "but it does not assert an experimental endpoint label or replace supervised model prediction."
            ),
            "",
        ]
    )


def render_ddi_card(*, pair: Mapping[str, Any], left: str, right: str, record: Mapping[str, Any]) -> str:
    left_name = pair.get("drug_a") or normalize_name(left)
    right_name = pair.get("drug_b") or normalize_name(right)
    key = pair_key(left, right)
    aliases = endpoint_terms("ddi")
    left_syn = pair.get("drug_a_synonyms", []) or []
    right_syn = pair.get("drug_b_synonyms", []) or []
    query = " ".join(unique_strings([left_name, right_name, *left_syn[:2], *right_syn[:2], *aliases[:4]], lower=True))
    return "\n".join(
        [
            f"# Paper QC DDI Evidence for {left_name} + {right_name}",
            "- source: paper_qc_curated",
            "- task: DDI",
            "- task_type: ddi_prediction",
            "- dataset_name: DDI",
            "- endpoint: ddi",
            f"- query: {query}",
            f"- evidence_type: ddi_pair_grounding",
            f"- pair_key: {key}",
            f"- curation_level: entity_normalized_pair_qc_seed",
            "- license_note: Local metadata-only QC grounding card; no external article or label text is reproduced.",
            "",
            f"## {left_name} and {right_name} pair grounding",
            "",
            (
                f"{left_name} and {right_name} form the normalized DDI benchmark pair `{key}`. "
                f"This QC grounding card links both pair members and DDI terminology "
                f"({', '.join(aliases[:8])}) for pair-aware retrieval quality control. "
                "It is pair-specific and endpoint-specific for DDI retrieval, but it does not assert "
                "a measured clinical interaction label or replace the supervised DDI classifier."
            ),
            "",
        ]
    )


def expected_endpoint(record: Mapping[str, Any]) -> str:
    return str(
        record.get("expected_endpoint")
        or endpoint_from_dataset(record.get("dataset_name"), task_type=record.get("task_type"))
        or ""
    ).strip().lower()


def drug_name_from_record(record: Mapping[str, Any], *, endpoint: str) -> str:
    name = str(record.get("name") or "").strip()
    dataset = str(record.get("dataset_name") or "").strip()
    candidates = [
        endpoint,
        endpoint_from_dataset(dataset),
        dataset.lower(),
        dataset.lower().replace("_veith", ""),
        dataset.lower().replace("_martins", ""),
        dataset.lower().replace("_wang", ""),
        dataset.lower().replace("_hou", ""),
        dataset.lower().replace("_ma", ""),
        dataset.lower().replace("_broccatelli", ""),
        dataset.lower().replace("_astrazeneca", ""),
    ]
    drug = name
    for suffix in sorted({normalize_suffix(item) for item in candidates if item}, key=len, reverse=True):
        marker = f"_{suffix}"
        if drug.lower().endswith(marker):
            drug = drug[: -len(marker)]
            break
    return normalize_name(drug.replace("_", " "))


def ddi_drugs_from_request(request: str) -> tuple[str, str]:
    parts = [normalize_name(part) for part in request.replace(";", ",").replace("+", ",").split(",") if part.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""


def pair_key(left: str, right: str) -> str:
    members = sorted([slug(left), slug(right)])
    return "__".join(members)


def normalize_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def normalize_suffix(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def slug(value: Any) -> str:
    text = normalize_name(value)
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def looks_like_smiles(value: str) -> bool:
    text = str(value or "").strip()
    if not text or " " in text:
        return False
    return any(char in text for char in "=#@[]()/\\") or bool(re.search(r"[cnops]\d", text, flags=re.I))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# AgentM Paper Evidence QC Resources",
        "",
        f"- Cases: `{summary.get('n_cases')}`",
        f"- Compounds: `{summary.get('n_compounds')}`",
        f"- Endpoints: `{summary.get('n_endpoints')}`",
        f"- DDI pairs: `{summary.get('n_ddi_pairs')}`",
        f"- Baseline failing cases repaired with QC cards: `{summary.get('n_qc_cards_emitted')}`",
        "",
        "## Source Policy",
        "",
        str(summary.get("source_policy") or ""),
        "",
        "## Emitted QC Cards",
        "",
    ]
    for card in summary.get("qc_cards", []) or []:
        lines.append(f"- `{card.get('label')}` -> `{card.get('path')}`")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
