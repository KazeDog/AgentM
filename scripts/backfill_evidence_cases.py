#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote
import xml.etree.ElementTree as ET

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentm.runtime.retrieval.evidence_schema import endpoint_terms, unique_strings


PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov"
DAILYMED_BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
PUBMED_EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

ENDPOINT_DATASETS = {
    "bbbp": "BBB_Martins",
    "dili": "DILI",
    "herg": "hERG",
    "ames": "AMES",
    "cyp1a2": "CYP1A2_Veith",
    "cyp2c9": "CYP2C9_Veith",
    "cyp2c19": "CYP2C19_Veith",
    "cyp2d6": "CYP2D6_Veith",
    "cyp3a4": "CYP3A4_Veith",
}

DAILYMED_SECTION_KEYWORDS = (
    "boxed warning",
    "contraindications",
    "warnings and precautions",
    "warnings",
    "adverse reactions",
    "drug interactions",
    "clinical pharmacology",
    "indications and usage",
)


class CachedNegativeResponse(RuntimeError):
    pass


class PublicDatabaseCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.stats: Dict[str, Any] = {
            "enabled": True,
            "root": str(self.root),
            "hits": 0,
            "misses": 0,
            "writes": 0,
            "negative_hits": 0,
        }

    def get(self, namespace: str, key: str) -> Optional[Dict[str, Any]]:
        path = self._path(namespace, key)
        if not path.exists():
            self.stats["misses"] += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self.stats["misses"] += 1
            return None
        self.stats["hits"] += 1
        if not payload.get("ok"):
            self.stats["negative_hits"] += 1
        return payload

    def set(self, namespace: str, key: str, payload: Mapping[str, Any]) -> None:
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(payload)
        record["cached_at"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.stats["writes"] += 1

    def _path(self, namespace: str, key: str) -> Path:
        digest = hashlib.sha1(str(key).encode("utf-8")).hexdigest()
        safe_namespace = re.sub(r"[^a-zA-Z0-9_.-]+", "_", namespace).strip("_") or "default"
        return self.root / safe_namespace / f"{digest}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill AgentM structured evidence for missing ADMET/DDI cases.")
    parser.add_argument("--output-dir", default="data/evidence_backfill")
    parser.add_argument("--case", action="append", default=[], help="ADMET case: drug:endpoint or drug:endpoint:dataset_name")
    parser.add_argument("--ddi-pair", action="append", default=[], help="DDI pair: drug_a:drug_b")
    parser.add_argument("--manifest-jsonl", action="append", default=[], help="Target JSONL from build_evidence_manifest.py.")
    parser.add_argument("--coverage-jsonl", default="", help="Optional coverage_targets.jsonl used to select missing/partial targets.")
    parser.add_argument("--coverage-status", default="missing,entity_background_only,endpoint_background_only")
    parser.add_argument("--target-kind", default="admet_case,ddi_pair")
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sources", default="pubchem,chembl,dailymed,pubmed")
    parser.add_argument("--max-pubmed", type=int, default=5)
    parser.add_argument("--max-dailymed", type=int, default=2)
    parser.add_argument("--max-chembl-activities", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=float(os.getenv("AGENTM_PUBCHEM_TIMEOUT", "20")))
    parser.add_argument("--request-delay", type=float, default=0.1)
    parser.add_argument("--ncbi-api-key", default=os.getenv("NCBI_API_KEY"))
    parser.add_argument("--http-retries", type=int, default=int(os.getenv("AGENTM_HTTP_RETRIES", "2")))
    parser.add_argument("--http-backoff-factor", type=float, default=float(os.getenv("AGENTM_HTTP_BACKOFF_FACTOR", "0.5")))
    parser.add_argument("--no-env-proxy", action="store_true", help="Ignore HTTP(S)_PROXY/ALL_PROXY environment variables for public database requests.")
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "public_db_cache"), help="Persistent public database response cache shared across batches.")
    parser.add_argument("--no-cache", action="store_true", help="Disable persistent public database cache.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = {item.strip().lower() for item in args.sources.split(",") if item.strip()}
    builder = EvidenceBackfillBuilder(
        output_dir=Path(args.output_dir),
        sources=sources,
        timeout=args.timeout,
        request_delay=args.request_delay,
        max_pubmed=args.max_pubmed,
        max_dailymed=args.max_dailymed,
        max_chembl_activities=args.max_chembl_activities,
        ncbi_api_key=args.ncbi_api_key,
        resume=args.resume,
        dry_run=args.dry_run,
        http_retries=args.http_retries,
        http_backoff_factor=args.http_backoff_factor,
        no_env_proxy=args.no_env_proxy,
        cache_dir=None if args.no_cache else Path(args.cache_dir),
    )
    summaries = []
    for raw_case in args.case:
        drug, endpoint, dataset = _parse_admet_case(raw_case)
        summaries.append(builder.backfill_admet_case(drug=drug, endpoint=endpoint, dataset_name=dataset))
    for raw_pair in args.ddi_pair:
        drug_a, drug_b = _parse_ddi_pair(raw_pair)
        summaries.append(builder.backfill_ddi_pair(drug_a=drug_a, drug_b=drug_b))
    coverage = _coverage_status_map(Path(args.coverage_jsonl)) if args.coverage_jsonl else {}
    manifest_targets = list(
        _iter_manifest_targets(
            [Path(path) for path in args.manifest_jsonl],
            coverage=coverage,
            coverage_statuses={item.strip() for item in args.coverage_status.split(",") if item.strip()},
            target_kinds={item.strip() for item in args.target_kind.split(",") if item.strip()},
            start_offset=args.start_offset,
            max_targets=args.max_targets,
        )
    )
    for target in manifest_targets:
        if target.get("kind") == "admet_case":
            summaries.append(
                builder.backfill_admet_case(
                    drug=_admet_query_from_target(target),
                    endpoint=str(target.get("endpoint") or ""),
                    dataset_name=str(target.get("dataset_name") or ""),
                    case_id=str(target.get("target_id") or ""),
                )
            )
        elif target.get("kind") == "ddi_pair":
            drug_a, drug_b = _ddi_queries_from_target(target)
            summaries.append(
                builder.backfill_ddi_pair(
                    drug_a=drug_a,
                    drug_b=drug_b,
                    pair_key=str(target.get("pair_key") or target.get("target_id") or ""),
                    target_id=str(target.get("target_id") or ""),
                )
            )
    manifest = builder.write_manifest(summaries)
    print(
        json.dumps(
            {
                "status": "completed",
                "manifest_path": str(manifest),
                "manifest_targets_selected": len(manifest_targets),
                "dry_run": bool(args.dry_run),
                "cases": summaries,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


class EvidenceBackfillBuilder:
    def __init__(
        self,
        *,
        output_dir: Path,
        sources: set[str],
        timeout: float,
        request_delay: float,
        max_pubmed: int,
        max_dailymed: int,
        max_chembl_activities: int,
        ncbi_api_key: Optional[str],
        resume: bool = True,
        dry_run: bool = False,
        http_retries: int = 2,
        http_backoff_factor: float = 0.5,
        no_env_proxy: bool = False,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.output_dir = output_dir
        self.sources = sources
        self.timeout = float(timeout)
        self.request_delay = float(request_delay)
        self.max_pubmed = int(max_pubmed)
        self.max_dailymed = int(max_dailymed)
        self.max_chembl_activities = int(max_chembl_activities)
        self.ncbi_api_key = ncbi_api_key
        self.resume = bool(resume)
        self.dry_run = bool(dry_run)
        self.http_retries = max(0, int(http_retries))
        self.http_backoff_factor = float(http_backoff_factor)
        self.no_env_proxy = bool(no_env_proxy)
        self.session = _new_http_session(
            http_retries=self.http_retries,
            http_backoff_factor=self.http_backoff_factor,
            no_env_proxy=self.no_env_proxy,
        )
        self.cache = PublicDatabaseCache(cache_dir) if cache_dir else None
        self.retrieved_at = datetime.now(timezone.utc).isoformat()
        self.failures: List[Dict[str, Any]] = []

    def backfill_admet_case(self, *, drug: str, endpoint: str, dataset_name: str, case_id: str = "") -> Dict[str, Any]:
        endpoint = endpoint.strip().lower()
        dataset_name = dataset_name or ENDPOINT_DATASETS.get(endpoint, endpoint)
        task_dir = self.output_dir / "tasks" / dataset_name
        task_dir.mkdir(parents=True, exist_ok=True)
        output = task_dir / f"backfill_{_slug(case_id or drug)}_{endpoint}.jsonl"
        if self.dry_run or (self.resume and output.exists()):
            return {
                "kind": "admet_case",
                "target_id": case_id or None,
                "drug": drug,
                "endpoint": endpoint,
                "dataset_name": dataset_name,
                "output_jsonl": str(output),
                "status": "dry_run" if self.dry_run else "skipped_existing",
                "n_documents": _count_jsonl(output) if output.exists() else 0,
                "source_counts": {},
            }
        failures_before = len(self.failures)
        identity = self._compound_identity(drug)
        docs: List[Dict[str, Any]] = []
        docs.extend(self._identity_docs(drug=drug, endpoint=endpoint, dataset_name=dataset_name, identity=identity, task_type="admet_prediction"))
        if "chembl" in self.sources:
            docs.extend(self._chembl_docs(drug=drug, endpoint=endpoint, dataset_name=dataset_name, identity=identity, task_type="admet_prediction"))
        if "dailymed" in self.sources:
            docs.extend(self._dailymed_docs(drug=drug, endpoint=endpoint, dataset_name=dataset_name, identity=identity, task_type="admet_prediction"))
        if "pubmed" in self.sources:
            docs.extend(self._pubmed_docs(drug=drug, endpoint=endpoint, dataset_name=dataset_name, identity=identity, task_type="admet_prediction"))
        self._write_jsonl(output, docs)
        case_failures = self.failures[failures_before:]
        return {
            "kind": "admet_case",
            "target_id": case_id or None,
            "drug": drug,
            "endpoint": endpoint,
            "dataset_name": dataset_name,
            "status": _case_status(docs, case_failures),
            "output_jsonl": str(output),
            "n_documents": len(docs),
            "n_failures": len(case_failures),
            "source_counts": _source_counts(docs),
        }

    def backfill_ddi_pair(self, *, drug_a: str, drug_b: str, pair_key: str = "", target_id: str = "") -> Dict[str, Any]:
        pair_key = _slug(pair_key) if pair_key else f"{_slug(drug_a)}__{_slug(drug_b)}"
        pair_dir = self.output_dir / "pairs" / pair_key
        pair_dir.mkdir(parents=True, exist_ok=True)
        output = pair_dir / "backfill_pair.jsonl"
        if self.dry_run or (self.resume and output.exists()):
            return {
                "kind": "ddi_pair",
                "target_id": target_id or None,
                "drug_a": drug_a,
                "drug_b": drug_b,
                "endpoint": "ddi",
                "dataset_name": "DDI",
                "output_jsonl": str(output),
                "pair_manifest": str(pair_dir / "pair_manifest.json"),
                "status": "dry_run" if self.dry_run else "skipped_existing",
                "n_documents": _count_jsonl(output) if output.exists() else 0,
                "source_counts": {},
            }
        failures_before = len(self.failures)
        identity_a = self._compound_identity(drug_a)
        identity_b = self._compound_identity(drug_b)
        docs: List[Dict[str, Any]] = []
        for role, drug, identity in [("drug_a", drug_a, identity_a), ("drug_b", drug_b, identity_b)]:
            docs.extend(
                self._identity_docs(
                    drug=drug,
                    endpoint="ddi",
                    dataset_name="DDI",
                    identity=identity,
                    task_type="ddi_prediction",
                    pair_drug_names=[drug_a, drug_b],
                    ddi_pair_key=pair_key,
                    role=role,
                )
            )
            if "chembl" in self.sources:
                docs.extend(
                    self._chembl_docs(
                        drug=drug,
                        endpoint="ddi",
                        dataset_name="DDI",
                        identity=identity,
                        task_type="ddi_prediction",
                        pair_drug_names=[drug_a, drug_b],
                        ddi_pair_key=pair_key,
                        role=role,
                    )
                )
            if "dailymed" in self.sources:
                docs.extend(
                    self._dailymed_docs(
                        drug=drug,
                        endpoint="ddi",
                        dataset_name="DDI",
                        identity=identity,
                        task_type="ddi_prediction",
                        pair_drug_names=[drug_a, drug_b],
                        ddi_pair_key=pair_key,
                        role=role,
                    )
                )
        if "pubmed" in self.sources:
            docs.extend(self._pubmed_pair_docs(drug_a=drug_a, drug_b=drug_b, identity_a=identity_a, identity_b=identity_b, pair_key=pair_key))
        self._write_jsonl(output, docs)
        case_failures = self.failures[failures_before:]
        manifest = {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "pair_key": pair_key,
            "pair_dir": str(pair_dir),
            "evidence_drug_a": identity_a.get("title") or drug_a,
            "evidence_drug_b": identity_b.get("title") or drug_b,
            "built_at": self.retrieved_at,
            "backfill_documents": len(docs),
        }
        (pair_dir / "pair_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {
            "kind": "ddi_pair",
            "target_id": target_id or None,
            "drug_a": drug_a,
            "drug_b": drug_b,
            "endpoint": "ddi",
            "dataset_name": "DDI",
            "status": _case_status(docs, case_failures),
            "output_jsonl": str(output),
            "pair_manifest": str(pair_dir / "pair_manifest.json"),
            "n_documents": len(docs),
            "n_failures": len(case_failures),
            "source_counts": _source_counts(docs),
        }

    def write_manifest(self, summaries: Sequence[Dict[str, Any]]) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "backfill_manifest.json"
        payload = {
            "created_at": self.retrieved_at,
            "sources": sorted(self.sources),
            "cache": self.cache.stats if self.cache else {"enabled": False},
            "summaries": list(summaries),
            "failures": self.failures,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if self.failures:
            failures_path = self.output_dir / "backfill_failures.jsonl"
            failures_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in self.failures) + "\n", encoding="utf-8")
        return path

    def _compound_identity(self, drug: str) -> Dict[str, Any]:
        identity = {"query": drug, "retrieval_status": "not_attempted", "compound_names": [drug], "compound_synonyms": [], "compound_cids": [], "chembl_ids": [], "canonical_smiles": []}
        if "pubchem" in self.sources:
            try:
                identity.update(self._pubchem_identity(drug))
            except Exception as exc:
                self._failure("pubchem", drug, exc)
                identity["retrieval_status"] = "pubchem_failed"
        if "chembl" in self.sources:
            try:
                chembl = self._chembl_identity(drug)
                if chembl:
                    identity["chembl_ids"] = unique_strings([*identity.get("chembl_ids", []), chembl.get("chembl_id")], lower=False)
                    identity["chembl_pref_name"] = chembl.get("pref_name")
                    identity["compound_names"] = unique_strings([*identity.get("compound_names", []), chembl.get("pref_name")], limit=20)
            except Exception as exc:
                self._failure("chembl_identity", drug, exc)
        return identity

    def _pubchem_identity(self, drug: str) -> Dict[str, Any]:
        namespace = "smiles" if _looks_like_smiles(drug) else "name"
        fields = "Title,IUPACName,MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,SMILES,ConnectivitySMILES,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount"
        payload = self._get_json(
            namespace="pubchem_property",
            cache_key=json.dumps({"namespace": namespace, "drug": drug, "fields": fields}, sort_keys=True),
            url=
            f"{PUBCHEM_BASE_URL}/rest/pug/compound/{namespace}/{quote(drug, safe='')}/property/{fields}/JSON",
        )
        rows = payload.get("PropertyTable", {}).get("Properties", []) or []
        if not rows:
            raise ValueError("PubChem returned no properties.")
        prop = rows[0]
        cid = prop.get("CID")
        synonyms = self._pubchem_synonyms(cid) if cid else []
        self._sleep()
        return {
            "retrieval_status": "success",
            "compound_names": unique_strings([prop.get("Title"), drug], limit=20),
            "compound_synonyms": unique_strings(synonyms, limit=80),
            "compound_cids": unique_strings([cid], lower=True),
            "canonical_smiles": unique_strings([prop.get("CanonicalSMILES"), prop.get("SMILES"), prop.get("ConnectivitySMILES"), prop.get("IsomericSMILES")], limit=10),
            "title": prop.get("Title") or drug,
            "iupac_name": prop.get("IUPACName"),
            "molecular_formula": prop.get("MolecularFormula"),
            "molecular_weight": prop.get("MolecularWeight"),
            "xlogp": prop.get("XLogP"),
            "tpsa": prop.get("TPSA"),
            "hbond_donor_count": prop.get("HBondDonorCount"),
            "hbond_acceptor_count": prop.get("HBondAcceptorCount"),
            "url": f"{PUBCHEM_BASE_URL}/compound/{cid}" if cid else None,
        }

    def _pubchem_synonyms(self, cid: Any) -> List[str]:
        if not cid:
            return []
        payload = self._get_json(
            namespace="pubchem_synonyms",
            cache_key=str(cid),
            url=f"{PUBCHEM_BASE_URL}/rest/pug/compound/cid/{quote(str(cid), safe='')}/synonyms/JSON",
        )
        rows = payload.get("InformationList", {}).get("Information", []) or []
        return [str(item) for item in (rows[0].get("Synonym", []) if rows else []) if item]

    def _chembl_identity(self, drug: str) -> Dict[str, Any]:
        params = {"q": drug, "limit": 1}
        payload = self._get_json(namespace="chembl_molecule_search", cache_key=json.dumps(params, sort_keys=True), url=f"{CHEMBL_BASE_URL}/molecule/search.json", params=params)
        rows = payload.get("molecules", []) or []
        if not rows:
            return {}
        row = rows[0]
        return {"chembl_id": row.get("molecule_chembl_id"), "pref_name": row.get("pref_name") or drug}

    def _identity_docs(
        self,
        *,
        drug: str,
        endpoint: str,
        dataset_name: str,
        identity: Mapping[str, Any],
        task_type: str,
        pair_drug_names: Optional[List[str]] = None,
        ddi_pair_key: Optional[str] = None,
        role: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if "pubchem" not in self.sources:
            return []
        if identity.get("retrieval_status") != "success":
            return []
        title = identity.get("title") or drug
        text = (
            f"{title} identity from PubChem. CID {', '.join(identity.get('compound_cids', []) or [])}. "
            f"Formula {identity.get('molecular_formula')}; molecular weight {identity.get('molecular_weight')}; "
            f"TPSA {identity.get('tpsa')}; XLogP {identity.get('xlogp')}. "
            f"Synonyms include {', '.join((identity.get('compound_synonyms') or [])[:8])}."
        )
        return [
            self._doc(
                source="pubchem",
                evidence_type="compound_identity",
                title=f"{title}: PubChem identity",
                text=text,
                endpoint=endpoint,
                dataset_name=dataset_name,
                task_type=task_type,
                identity=identity,
                url=identity.get("url"),
                pair_drug_names=pair_drug_names,
                ddi_pair_key=ddi_pair_key,
                role=role,
            )
        ]

    def _chembl_docs(
        self,
        *,
        drug: str,
        endpoint: str,
        dataset_name: str,
        identity: Mapping[str, Any],
        task_type: str,
        pair_drug_names: Optional[List[str]] = None,
        ddi_pair_key: Optional[str] = None,
        role: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        chembl_id = (identity.get("chembl_ids") or [None])[0]
        if not chembl_id:
            return []
        docs: List[Dict[str, Any]] = []
        mechanisms = self._chembl_endpoint("mechanism", {"molecule_chembl_id": chembl_id, "limit": 5}, "mechanisms")
        indications = self._chembl_endpoint("drug_indication", {"molecule_chembl_id": chembl_id, "limit": 5}, "drug_indications")
        activities = self._chembl_endpoint("activity", {"molecule_chembl_id": chembl_id, "limit": self.max_chembl_activities}, "activities")
        parts = [f"ChEMBL profile for {identity.get('chembl_pref_name') or drug} ({chembl_id})."]
        if mechanisms:
            labels = [item.get("mechanism_of_action") or item.get("action_type") or item.get("target_pref_name") for item in mechanisms if isinstance(item, dict)]
            parts.append("Mechanism records: " + "; ".join(str(item) for item in labels[:5] if item))
        if indications:
            labels = [item.get("mesh_heading") or item.get("efo_term") for item in indications if isinstance(item, dict)]
            parts.append("Indication records: " + "; ".join(str(item) for item in labels[:5] if item))
        if activities:
            labels = [item.get("target_pref_name") or item.get("standard_type") for item in activities if isinstance(item, dict)]
            parts.append("Bioactivity records include: " + "; ".join(str(item) for item in labels[:8] if item))
        docs.append(
            self._doc(
                source="chembl",
                evidence_type="bioactivity",
                title=f"{drug}: ChEMBL profile {chembl_id}",
                text=" ".join(parts),
                endpoint=endpoint,
                dataset_name=dataset_name,
                task_type=task_type,
                identity=identity,
                url=f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/",
                pair_drug_names=pair_drug_names,
                ddi_pair_key=ddi_pair_key,
                role=role,
                extra={"chembl_ids": [chembl_id], "targets": unique_strings(_chembl_targets([*mechanisms, *activities]), limit=40)},
            )
        )
        return docs

    def _chembl_endpoint(self, endpoint: str, params: Mapping[str, Any], payload_key: str) -> List[Dict[str, Any]]:
        try:
            payload = self._get_json(namespace=f"chembl_{endpoint}", cache_key=json.dumps(dict(params), sort_keys=True), url=f"{CHEMBL_BASE_URL}/{endpoint}.json", params=params)
            self._sleep()
            rows = payload.get(payload_key, []) or []
            return [row for row in rows if isinstance(row, dict)]
        except Exception as exc:
            self._failure(f"chembl_{endpoint}", json.dumps(params, ensure_ascii=False), exc)
            return []

    def _dailymed_docs(
        self,
        *,
        drug: str,
        endpoint: str,
        dataset_name: str,
        identity: Mapping[str, Any],
        task_type: str,
        pair_drug_names: Optional[List[str]] = None,
        ddi_pair_key: Optional[str] = None,
        role: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        for candidate in self._candidate_names(drug, identity):
            try:
                payload = self._get_json(
                    namespace="dailymed_spls",
                    cache_key=json.dumps({"drug_name": candidate, "page": 1, "pageSize": self.max_dailymed}, sort_keys=True),
                    url=
                    f"{DAILYMED_BASE_URL}/spls.json",
                    params={"drug_name": candidate, "page": 1, "pageSize": self.max_dailymed},
                )
                rows = payload.get("data") or payload.get("results") or []
            except Exception as exc:
                self._failure("dailymed", candidate, exc)
                rows = []
            self._sleep()
            for row in rows[: self.max_dailymed] if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                setid = row.get("setid") or row.get("set_id")
                sections = self._dailymed_sections(setid) if setid else []
                text = " ".join([f"{item.get('title')}: {item.get('text')}" for item in sections]) or f"DailyMed label metadata for {candidate}."
                docs.append(
                    self._doc(
                        source="dailymed",
                        evidence_type="drug_label",
                        title=f"{candidate}: {row.get('title') or setid}",
                        text=text,
                        endpoint=endpoint,
                        dataset_name=dataset_name,
                        task_type=task_type,
                        identity=identity,
                        url=f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}" if setid else None,
                        pair_drug_names=pair_drug_names,
                        ddi_pair_key=ddi_pair_key,
                        role=role,
                        extra={"setid": setid, "published_date": row.get("published_date")},
                    )
                )
            if docs:
                break
        return docs

    def _dailymed_sections(self, setid: str) -> List[Dict[str, str]]:
        try:
            text = self._get_text(namespace="dailymed_spl_xml", cache_key=str(setid), url=f"{DAILYMED_BASE_URL}/spls/{quote(str(setid), safe='')}.xml")
            root = ET.fromstring(text)
        except Exception as exc:
            self._failure("dailymed_sections", str(setid), exc)
            return []
        sections: List[Dict[str, str]] = []
        seen = set()
        for section in root.iter():
            if _local_name(section.tag) != "section":
                continue
            title = _section_title(section)
            key = title.lower()
            if not title or key in seen or not any(token in key for token in DAILYMED_SECTION_KEYWORDS):
                continue
            text = _section_text(section, title)
            if text:
                sections.append({"title": title, "text": text[:1200]})
                seen.add(key)
            if len(sections) >= 5:
                break
        self._sleep()
        return sections

    def _pubmed_docs(
        self,
        *,
        drug: str,
        endpoint: str,
        dataset_name: str,
        identity: Mapping[str, Any],
        task_type: str,
    ) -> List[Dict[str, Any]]:
        aliases = [drug, *(identity.get("compound_synonyms") or [])[:3]]
        query = _pubmed_admet_query(aliases, endpoint)
        return self._pubmed_query_docs(
            query=query,
            endpoint=endpoint,
            dataset_name=dataset_name,
            task_type=task_type,
            identity=identity,
            title_prefix=f"{drug} {endpoint}",
        )

    def _pubmed_pair_docs(self, *, drug_a: str, drug_b: str, identity_a: Mapping[str, Any], identity_b: Mapping[str, Any], pair_key: str) -> List[Dict[str, Any]]:
        query = _pubmed_ddi_query(self._candidate_names(drug_a, identity_a), self._candidate_names(drug_b, identity_b))
        pair_terms = [drug_a, drug_b, *(identity_a.get("compound_synonyms") or [])[:3], *(identity_b.get("compound_synonyms") or [])[:3]]
        merged_identity = {
            "compound_names": unique_strings([drug_a, drug_b, *identity_a.get("compound_names", []), *identity_b.get("compound_names", [])], limit=30),
            "compound_synonyms": unique_strings([*identity_a.get("compound_synonyms", []), *identity_b.get("compound_synonyms", [])], limit=120),
            "compound_cids": unique_strings([*identity_a.get("compound_cids", []), *identity_b.get("compound_cids", [])], lower=True),
            "chembl_ids": unique_strings([*identity_a.get("chembl_ids", []), *identity_b.get("chembl_ids", [])], lower=False),
            "canonical_smiles": unique_strings([*identity_a.get("canonical_smiles", []), *identity_b.get("canonical_smiles", [])], limit=20),
        }
        return self._pubmed_query_docs(
            query=query,
            endpoint="ddi",
            dataset_name="DDI",
            task_type="ddi_prediction",
            identity=merged_identity,
            title_prefix=f"{drug_a} {drug_b} DDI",
            pair_drug_names=pair_terms,
            ddi_pair_key=pair_key,
        )

    def _pubmed_query_docs(
        self,
        *,
        query: str,
        endpoint: str,
        dataset_name: str,
        task_type: str,
        identity: Mapping[str, Any],
        title_prefix: str,
        pair_drug_names: Optional[List[str]] = None,
        ddi_pair_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        try:
            pmids = self._pubmed_search(query)
            records = self._pubmed_fetch(pmids)
        except Exception as exc:
            self._failure("pubmed", query, exc)
            return []
        docs = []
        for record in records:
            docs.append(
                self._doc(
                    source="pubmed",
                    evidence_type="literature",
                    title=f"PMID {record.get('pmid')}: {record.get('title') or title_prefix}",
                    text=str(record.get("abstract") or "No abstract available."),
                    endpoint=endpoint,
                    dataset_name=dataset_name,
                    task_type=task_type,
                    identity=identity,
                    url=record.get("url"),
                    pair_drug_names=pair_drug_names,
                    ddi_pair_key=ddi_pair_key,
                    extra={"pmid": record.get("pmid"), "journal": record.get("journal"), "year": record.get("year"), "query": query},
                )
            )
        return docs

    def _pubmed_search(self, query: str) -> List[str]:
        params = {"db": "pubmed", "retmode": "json", "retmax": self.max_pubmed, "term": query}
        if self.ncbi_api_key:
            params["api_key"] = self.ncbi_api_key
        payload = self._get_json(namespace="pubmed_esearch", cache_key=json.dumps(params, sort_keys=True), url=f"{PUBMED_EUTILS_BASE_URL}/esearch.fcgi", params=params)
        self._sleep()
        return payload.get("esearchresult", {}).get("idlist", [])

    def _pubmed_fetch(self, pmids: Sequence[str]) -> List[Dict[str, Any]]:
        if not pmids:
            return []
        params = {"db": "pubmed", "retmode": "xml", "id": ",".join(str(item) for item in pmids)}
        if self.ncbi_api_key:
            params["api_key"] = self.ncbi_api_key
        text = self._get_text(namespace="pubmed_efetch", cache_key=json.dumps(params, sort_keys=True), url=f"{PUBMED_EUTILS_BASE_URL}/efetch.fcgi", params=params)
        self._sleep()
        return _parse_pubmed_xml(text, retrieved_at=self.retrieved_at)

    def _get_json(self, *, namespace: str, cache_key: str, url: str, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        payload = self._get_cached_response(namespace=namespace, cache_key=cache_key, url=url, params=params, response_kind="json")
        return payload if isinstance(payload, dict) else {}

    def _get_text(self, *, namespace: str, cache_key: str, url: str, params: Optional[Mapping[str, Any]] = None) -> str:
        payload = self._get_cached_response(namespace=namespace, cache_key=cache_key, url=url, params=params, response_kind="text")
        return str(payload or "")

    def _get_cached_response(
        self,
        *,
        namespace: str,
        cache_key: str,
        url: str,
        params: Optional[Mapping[str, Any]],
        response_kind: str,
    ) -> Any:
        if self.cache:
            cached = self.cache.get(namespace, cache_key)
            if cached is not None:
                if cached.get("ok"):
                    return cached.get("payload")
                raise CachedNegativeResponse(str(cached.get("error") or "cached negative response"))
        response = self.session.get(url, params=params, timeout=self.timeout)
        if response.status_code in {400, 404}:
            error = f"HTTP {response.status_code}: {response.text[:300]}"
            if self.cache:
                self.cache.set(namespace, cache_key, {"ok": False, "error": error, "status_code": response.status_code, "url": response.url})
            response.raise_for_status()
        response.raise_for_status()
        payload = response.json() if response_kind == "json" else response.text
        if self.cache:
            self.cache.set(namespace, cache_key, {"ok": True, "payload": payload, "status_code": response.status_code, "url": response.url})
        return payload

    def _doc(
        self,
        *,
        source: str,
        evidence_type: str,
        title: str,
        text: str,
        endpoint: str,
        dataset_name: str,
        task_type: str,
        identity: Mapping[str, Any],
        url: Optional[str] = None,
        pair_drug_names: Optional[List[str]] = None,
        ddi_pair_key: Optional[str] = None,
        role: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        doc = {
            "source": source,
            "evidence_type": evidence_type,
            "title": title,
            "text": text,
            "task_type": task_type,
            "dataset_name": dataset_name,
            "endpoint": endpoint,
            "endpoint_aliases": endpoint_terms(endpoint),
            "endpoint_terms": endpoint_terms(endpoint),
            "compound_names": unique_strings(identity.get("compound_names", []), limit=40),
            "compound_synonyms": unique_strings(identity.get("compound_synonyms", []), limit=120),
            "compound_cids": unique_strings(identity.get("compound_cids", []), lower=True),
            "chembl_ids": unique_strings(identity.get("chembl_ids", []), lower=False),
            "canonical_smiles": unique_strings(identity.get("canonical_smiles", []), limit=20),
            "drug_names": unique_strings(identity.get("compound_names", []), limit=40),
            "pair_drug_names": unique_strings(pair_drug_names or [], limit=80),
            "ddi_pair_key": ddi_pair_key,
            "role": role,
            "url": url,
            "retrieved_at": self.retrieved_at,
        }
        doc["entity_terms"] = unique_strings(
            [
                *doc["compound_names"],
                *doc["compound_synonyms"],
                *doc["compound_cids"],
                *doc["chembl_ids"],
                *doc["canonical_smiles"],
                *doc["pair_drug_names"],
            ],
            lower=True,
            limit=160,
        )
        if extra:
            doc.update({key: value for key, value in extra.items() if value not in (None, [], {})})
        doc["compound_specific"] = _compound_specific(source=source, evidence_type=evidence_type, text=f"{title} {text}", entity_terms=doc["entity_terms"])
        doc["endpoint_specific"] = _endpoint_specific(endpoint=endpoint, evidence_type=evidence_type, text=f"{title} {text}", task_type=task_type)
        doc["evidence_scope"] = _evidence_scope(
            task_type=task_type,
            compound_specific=doc["compound_specific"],
            endpoint_specific=doc["endpoint_specific"],
            has_pair=bool(ddi_pair_key and pair_drug_names),
        )
        return doc

    def _candidate_names(self, drug: str, identity: Mapping[str, Any]) -> List[str]:
        return unique_strings([drug, *identity.get("compound_names", []), *identity.get("compound_synonyms", [])], limit=8)

    def _write_jsonl(self, path: Path, docs: Sequence[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(doc, ensure_ascii=False) for doc in docs) + ("\n" if docs else ""), encoding="utf-8")

    def _failure(self, source: str, query: str, exc: Exception) -> None:
        self.failures.append({"source": source, "query": query, "error": str(exc), "retrieved_at": self.retrieved_at})

    def _sleep(self) -> None:
        if self.request_delay > 0:
            time.sleep(self.request_delay)


def _parse_admet_case(raw: str) -> Tuple[str, str, str]:
    parts = [item.strip() for item in raw.split(":")]
    if len(parts) < 2:
        raise SystemExit(f"Invalid --case {raw!r}; expected drug:endpoint or drug:endpoint:dataset_name")
    drug, endpoint = parts[0], parts[1].lower()
    dataset = parts[2] if len(parts) >= 3 and parts[2] else ENDPOINT_DATASETS.get(endpoint, endpoint)
    return drug, endpoint, dataset


def _parse_ddi_pair(raw: str) -> Tuple[str, str]:
    parts = [item.strip() for item in raw.split(":")]
    if len(parts) != 2:
        raise SystemExit(f"Invalid --ddi-pair {raw!r}; expected drug_a:drug_b")
    return parts[0], parts[1]


def _iter_manifest_targets(
    paths: Sequence[Path],
    *,
    coverage: Mapping[str, str],
    coverage_statuses: set[str],
    target_kinds: set[str],
    start_offset: int,
    max_targets: int,
) -> Iterable[Dict[str, Any]]:
    emitted = 0
    seen = 0
    for path in paths:
        for target in _read_jsonl(path):
            kind = str(target.get("kind") or "")
            if target_kinds and kind not in target_kinds:
                continue
            target_id = str(target.get("target_id") or "")
            status = coverage.get(target_id)
            if coverage and coverage_statuses and status not in coverage_statuses:
                continue
            if seen < start_offset:
                seen += 1
                continue
            target["coverage_status"] = status
            yield target
            emitted += 1
            if max_targets and emitted >= max_targets:
                return


def _coverage_status_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise SystemExit(f"--coverage-jsonl not found: {path}")
    out: Dict[str, str] = {}
    for row in _read_jsonl(path):
        target_id = str(row.get("target_id") or "")
        if target_id:
            out[target_id] = str(row.get("status") or "")
    return out


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"JSONL not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            yield payload


def _admet_query_from_target(target: Mapping[str, Any]) -> str:
    for key in ["drug_name", "name", "drug_id"]:
        value = str(target.get(key) or "").strip()
        if value and _is_public_database_query(value):
            return value
    return str(target.get("smiles") or target.get("drug_id") or target.get("target_id") or "").strip()


def _ddi_queries_from_target(target: Mapping[str, Any]) -> Tuple[str, str]:
    return _ddi_side_query(target.get("drug_a") or {}), _ddi_side_query(target.get("drug_b") or {})


def _ddi_side_query(side: Mapping[str, Any]) -> str:
    for key in ["name", "drug_name", "drug_id"]:
        value = str(side.get(key) or "").strip()
        if value and _is_public_database_query(value):
            return value
    return str(side.get("smiles") or side.get("drug_id") or side.get("name") or "").strip()


def _is_public_database_query(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if _looks_like_smiles(text):
        return True
    if re.match(r"^drug\s*\d+$", text, flags=re.I):
        return False
    if re.match(r"^(DB|CID)\d+", text, flags=re.I):
        return False
    if re.match(r"^\d+(\.\d+)?$", text):
        return False
    return True


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())


def _pubmed_admet_query(aliases: Sequence[str], endpoint: str) -> str:
    drug_terms = " OR ".join(f'"{item}"' for item in unique_strings(aliases, limit=4) if item)
    endpoint_query = " OR ".join(f'"{item}"' for item in endpoint_terms(endpoint)[:6])
    return f"({drug_terms}) AND ({endpoint_query}) AND (drug OR clinical OR pharmacokinetics OR toxicity OR safety)"


def _pubmed_ddi_query(drug_a_aliases: Sequence[str], drug_b_aliases: Sequence[str]) -> str:
    a_terms = " OR ".join(f'"{item}"' for item in unique_strings(drug_a_aliases, limit=4) if item)
    b_terms = " OR ".join(f'"{item}"' for item in unique_strings(drug_b_aliases, limit=4) if item)
    return f"({a_terms}) AND ({b_terms}) AND (interaction OR coadministration OR pharmacokinetic OR bleeding OR toxicity)"


def _compound_specific(*, source: str, evidence_type: str, text: str, entity_terms: Sequence[str]) -> bool:
    source = str(source or "").lower()
    evidence_type = str(evidence_type or "").lower()
    if source in {"pubchem", "dailymed", "chembl"} and evidence_type in {"compound_identity", "drug_label", "bioactivity"}:
        return True
    return _mentions_any(text, entity_terms, min_chars=5, compact_min_chars=8)


def _endpoint_specific(*, endpoint: str, evidence_type: str, text: str, task_type: str) -> bool:
    evidence_type = str(evidence_type or "").lower()
    if evidence_type == "compound_identity":
        return False
    endpoint = str(endpoint or "").lower()
    terms = endpoint_terms(endpoint)
    if endpoint == "ddi" or task_type == "ddi_prediction":
        terms = unique_strings(
            [
                *terms,
                "cyp3a4",
                "cyp inhibitor",
                "p-gp",
                "rhabdomyolysis",
                "myopathy",
                "contraindicated",
                "bleeding",
                "anticoagulant",
                "antiplatelet",
                "dual therapy",
                "triple therapy",
                "combined therapy",
            ],
            lower=True,
        )
    return _mentions_any(text, terms, min_chars=3, compact_min_chars=5)


def _evidence_scope(*, task_type: str, compound_specific: bool, endpoint_specific: bool, has_pair: bool) -> str:
    if task_type == "ddi_prediction" and has_pair and compound_specific and endpoint_specific:
        return "ddi_pair"
    if task_type == "ddi_prediction" and has_pair and compound_specific:
        return "ddi_pair_background"
    if compound_specific and endpoint_specific:
        return "compound_endpoint"
    if compound_specific:
        return "compound_background"
    if endpoint_specific:
        return "endpoint_background"
    return "background"


def _mentions_any(text: str, terms: Sequence[Any], *, min_chars: int, compact_min_chars: int) -> bool:
    norm_text = _norm_for_match(text)
    padded_text = f" {norm_text} "
    compact_text = re.sub(r"[^a-z0-9]+", "", norm_text)
    for term in terms or []:
        norm = _norm_for_match(str(term))
        if not norm:
            continue
        compact = re.sub(r"[^a-z0-9]+", "", norm)
        if len(norm) < min_chars and not any(char.isdigit() for char in compact):
            continue
        if f" {norm} " in padded_text:
            return True
        if len(compact) >= compact_min_chars and compact in compact_text:
            return True
    return False


def _norm_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())).strip()


def _parse_pubmed_xml(xml_text: str, *, retrieved_at: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    records = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _node_text(article.find(".//PMID"))
        title = _node_text(article.find(".//ArticleTitle"))
        journal = _node_text(article.find(".//Journal/Title"))
        year = _node_text(article.find(".//PubDate/Year"))
        abstracts = [_node_text(node) for node in article.findall(".//Abstract/AbstractText")]
        records.append(
            {
                "pmid": pmid,
                "title": title,
                "journal": journal,
                "year": year,
                "abstract": " ".join(item for item in abstracts if item),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                "retrieved_at": retrieved_at,
            }
        )
    return records


def _node_text(node: Any) -> str:
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def _local_name(tag: str) -> str:
    return str(tag).split("}", 1)[-1]


def _section_title(section: Any) -> str:
    for child in section:
        if _local_name(child.tag) == "title":
            return " ".join("".join(child.itertext()).split())
    return ""


def _section_text(section: Any, title: str) -> str:
    text = " ".join(" ".join(section.itertext()).split())
    if title and text.lower().startswith(title.lower()):
        text = text[len(title) :].strip(" :-")
    return text


def _chembl_targets(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    values: List[Any] = []
    for row in rows:
        values.extend(row.get(key) for key in ["target_pref_name", "target_chembl_id", "standard_type"])
    return unique_strings(values, limit=50)


def _source_counts(docs: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for doc in docs:
        source = str(doc.get("source") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def _case_status(docs: Sequence[Mapping[str, Any]], failures: Sequence[Mapping[str, Any]]) -> str:
    if docs and failures:
        return "completed_with_warnings"
    if docs:
        return "completed"
    if failures:
        return "failed_no_documents"
    return "no_documents"


def _new_http_session(*, http_retries: int, http_backoff_factor: float, no_env_proxy: bool) -> requests.Session:
    session = requests.Session()
    session.trust_env = not bool(no_env_proxy)
    retry = Retry(
        total=max(0, int(http_retries)),
        connect=max(0, int(http_retries)),
        read=max(0, int(http_retries)),
        status=max(0, int(http_retries)),
        backoff_factor=float(http_backoff_factor),
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return text or "unknown"


def _looks_like_smiles(value: str) -> bool:
    text = str(value or "")
    return any(ch in text for ch in "=#@[]()/\\") and bool(re.search(r"[A-Za-z]", text))


if __name__ == "__main__":
    raise SystemExit(main())
