from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

from agentm.runtime.tools.scientific import ScientificToolRuntime
from experiments.agentm_reliability.common import source_tree_hash, utc_now, write_json
from experiments.agentm_reliability.evidence_provenance import (
    audit_evidence_provenance,
    classify_evidence_layer,
    evidence_is_direct,
    normalize_evidence_record,
)
from experiments.agentm_reliability.runtime_guards import is_reaction_case


def _objects(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _objects(child)


def _fixed_llm_jobs(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_run: dict[str, dict[str, Any]] = {}
    for item in _objects(manifest.get("phases", {})):
        if item.get("orchestrator") != "llm" or not item.get("job_path"):
            continue
        run_id = str(item.get("run_id") or "")
        if run_id:
            by_run[run_id] = dict(item)
    return [by_run[key] for key in sorted(by_run)]


def _runtime_case(case: Mapping[str, Any]) -> dict[str, Any]:
    base = case.get("base_case")
    return dict(base) if isinstance(base, Mapping) else dict(case)


def _resolve_context(runtime: ScientificToolRuntime, case: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    active = _runtime_case(case)
    context: dict[str, Any] = {
        "request": str(active.get("request") or active.get("instruction") or ""),
        "task_type": str(active.get("task_type") or case.get("task_type") or ""),
        "dataset_name": str(active.get("dataset_name") or case.get("dataset_name") or ""),
        "metadata": {
            "enable_fallbacks": False,
            "require_real_rag": True,
            "use_external_evidence": False,
        },
    }
    route = runtime.route_task({}, context)
    if route.status != "completed":
        return context, f"route_failed:{route.error or route.status}"
    resolver = runtime.resolve_ddi_pair if context.get("task_type") == "ddi_prediction" else runtime.resolve_molecule
    resolved = resolver({}, context)
    if resolved.status != "completed":
        return context, f"resolve_failed:{resolved.error or resolved.status}"
    return context, "completed"


def scan(
    manifest_path: Path,
    *,
    es_url: str,
    index_name: str | None,
    top_k: int,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixed_jobs = _fixed_llm_jobs(manifest)
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="agentm_direct_evidence_scan_") as tmp:
        runtime = ScientificToolRuntime(
            output_dir=tmp,
            feature_set="fingerprint_only",
            require_real_model=False,
            require_real_rag=True,
            min_evidence=1,
            use_external_evidence=False,
            use_smiles_encoder=False,
            use_mllm_visual_semantic=False,
            retrieval_backend="es",
            es_url=es_url,
            use_llm_reasoning=False,
            require_llm_reasoning=False,
            llm_profile=None,
        )
        if index_name:
            runtime.hybrid_retriever.index_name = index_name
        for fixed in fixed_jobs:
            job_path = Path(str(fixed["job_path"]))
            job = json.loads(job_path.read_text(encoding="utf-8"))
            case = job.get("case", {}) or {}
            row: dict[str, Any] = {
                "run_id": fixed.get("run_id"),
                "case_id": fixed.get("case_id"),
                "dataset_group": fixed.get("dataset_group"),
                "task_type": fixed.get("task_type"),
                "job_path": str(job_path),
                "llm_calls": 0,
            }
            if is_reaction_case(case):
                row.update({"scan_status": "reaction_local_evidence_not_es_scanned", "eligible_direct_canary": False})
                records.append(row)
                continue
            active = _runtime_case(case)
            if str(fixed.get("dataset_group")) == "multi_turn" or str(active.get("task_type")) == "multi_turn":
                row.update({"scan_status": "multi_turn_not_single_query_scanned", "eligible_direct_canary": False})
                records.append(row)
                continue
            context, resolution_status = _resolve_context(runtime, case)
            row["resolution_status"] = resolution_status
            row["resolved_names"] = runtime._molecule_names(context)
            if resolution_status != "completed":
                row.update({"scan_status": resolution_status, "eligible_direct_canary": False})
                records.append(row)
                continue
            query = runtime._rag_query(context)
            metadata = runtime._retrieval_metadata(context)
            result = runtime.hybrid_retriever.retrieve(query, metadata=metadata, backend="es", top_k=top_k)
            audit = audit_evidence_provenance({"evidence": result.evidence_items})
            direct_items = []
            for item in result.evidence_items:
                normalized = normalize_evidence_record(item)
                if evidence_is_direct(normalized):
                    direct_items.append(
                        {
                            "chunk_id": normalized.get("chunk_id"),
                            "source": normalized.get("source"),
                            "layer": classify_evidence_layer(normalized),
                            "title": normalized.get("title"),
                            "source_path": normalized.get("source_path"),
                        }
                    )
            execution_valid = bool(
                result.provider == "elasticsearch_hybrid"
                and result.backend == "elasticsearch_hybrid_bm25_dense"
                and not result.warnings
            )
            row.update(
                {
                    "scan_status": "completed" if execution_valid else "retrieval_invalid",
                    "query": query,
                    "provider": result.provider,
                    "backend": result.backend,
                    "warnings": list(result.warnings),
                    "retrieved_count": len(result.evidence_items),
                    "external_evidence_count": audit.get("external_evidence_count"),
                    "direct_evidence_count": audit.get("direct_evidence_count"),
                    "qc_grounding_count": audit.get("qc_grounding_count"),
                    "direct_evidence_coverage": audit.get("direct_evidence_coverage"),
                    "direct_items": direct_items,
                    "eligible_direct_canary": bool(execution_valid and direct_items),
                }
            )
            records.append(row)
    eligible = [row for row in records if row.get("eligible_direct_canary")]
    eligible.sort(
        key=lambda row: (
            -int(row.get("direct_evidence_count", 0) or 0),
            str(row.get("dataset_group") or ""),
            str(row.get("case_id") or ""),
        )
    )
    return {
        "generated_at": utc_now(),
        "scan_mode": "offline_no_llm_runtime_like_dense_hybrid",
        "llm_calls": 0,
        "source_tree_hash": source_tree_hash(),
        "elasticsearch_index_name": runtime.hybrid_retriever.index_name,
        "manifest_path": str(manifest_path),
        "fixed_llm_condition_count": len(fixed_jobs),
        "scanned_nonreaction_single_query_count": sum(row.get("scan_status") in {"completed", "retrieval_invalid"} for row in records),
        "eligible_direct_canary_count": len(eligible),
        "eligible_candidates": eligible,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="No-LLM scan of fixed-pilot direct-evidence candidates.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--es-url", default="http://127.0.0.1:9200")
    parser.add_argument("--index-name", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    payload = scan(
        args.manifest.resolve(),
        es_url=args.es_url,
        index_name=args.index_name,
        top_k=args.top_k,
    )
    write_json(args.output.resolve(), payload, overwrite=True)
    print(json.dumps({key: payload[key] for key in ("fixed_llm_condition_count", "scanned_nonreaction_single_query_count", "eligible_direct_canary_count", "llm_calls", "source_tree_hash")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
