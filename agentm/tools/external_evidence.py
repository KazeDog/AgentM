from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests


PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov"
PUBMED_EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DAILYMED_BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"


def external_evidence_enabled(default: bool = False) -> bool:
    value = os.getenv("AGENTM_ENABLE_EXTERNAL_EVIDENCE")
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class ExternalEvidenceClient:
    def __init__(self, *, enabled: Optional[bool] = None, timeout: Optional[float] = None):
        self.enabled = external_evidence_enabled(default=False) if enabled is None else bool(enabled)
        self.timeout = float(timeout if timeout is not None else os.getenv("AGENTM_EXTERNAL_EVIDENCE_TIMEOUT", "12"))

    def retrieve_ddi(self, drug_a: str | None, drug_b: str | None, *, max_results: int = 3) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        evidence: List[Dict[str, Any]] = []
        evidence.extend(self.pubchem_compound_summary(drug_a))
        evidence.extend(self.pubchem_compound_summary(drug_b))
        evidence.extend(self.dailymed_label_summary(drug_a, max_results=2))
        evidence.extend(self.dailymed_label_summary(drug_b, max_results=2))
        evidence.extend(self.pubmed_pair_search(drug_a, drug_b, max_results=max_results))
        return evidence

    def retrieve_single(self, drug_name: str | None, *, dataset_name: str | None = None, max_results: int = 3) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        name = (drug_name or "").strip()
        if not name:
            return []
        evidence: List[Dict[str, Any]] = []
        evidence.extend(self.pubchem_compound_summary(name))
        evidence.extend(self.dailymed_label_summary(name, max_results=2))
        evidence.extend(self.pubmed_single_search(name, dataset_name=dataset_name, max_results=max_results))
        return evidence

    def pubchem_compound_summary(self, drug_name: str | None) -> List[Dict[str, Any]]:
        name = (drug_name or "").strip()
        if not name:
            return []
        try:
            response = requests.get(
                f"{PUBCHEM_BASE_URL}/rest/pug/compound/name/{requests.utils.quote(name)}/cids/JSON",
                timeout=self.timeout,
            )
            response.raise_for_status()
            cids = response.json().get("IdentifierList", {}).get("CID", [])
            if not cids:
                return []
            cid = cids[0]
            return [
                {
                    "source": "pubchem",
                    "claim": "PubChem compound identifier found for drug name.",
                    "drug_name": name,
                    "value": {"cid": cid, "all_cids": cids[:5], "url": f"{PUBCHEM_BASE_URL}/compound/{cid}"},
                }
            ]
        except Exception as exc:
            return [self._error("pubchem", name, exc)]

    def dailymed_label_summary(self, drug_name: str | None, *, max_results: int = 2) -> List[Dict[str, Any]]:
        name = (drug_name or "").strip()
        if not name:
            return []
        try:
            response = requests.get(
                f"{DAILYMED_BASE_URL}/spls.json",
                params={"drug_name": name, "page": 1, "pageSize": max_results},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            rows = data.get("data") or data.get("results") or []
            if not isinstance(rows, list) or not rows:
                return []
            labels = []
            for row in rows[:max_results]:
                if not isinstance(row, dict):
                    continue
                setid = row.get("setid") or row.get("set_id")
                labels.append(
                    {
                        "setid": setid,
                        "title": row.get("title"),
                        "published_date": row.get("published_date"),
                        "url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}" if setid else None,
                    }
                )
            return [
                {
                    "source": "dailymed",
                    "claim": "DailyMed label records found for drug name.",
                    "drug_name": name,
                    "value": {"labels": labels},
                }
            ]
        except Exception as exc:
            return [self._error("dailymed", name, exc)]

    def pubmed_pair_search(self, drug_a: str | None, drug_b: str | None, *, max_results: int = 3) -> List[Dict[str, Any]]:
        a = (drug_a or "").strip()
        b = (drug_b or "").strip()
        if not a or not b:
            return []
        query = f'"{a}" "{b}" drug interaction'
        try:
            response = requests.get(
                f"{PUBMED_EUTILS_BASE_URL}/esearch.fcgi",
                params={"db": "pubmed", "retmode": "json", "retmax": max_results, "term": query},
                timeout=self.timeout,
            )
            response.raise_for_status()
            ids = response.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []
            return [
                {
                    "source": "pubmed",
                    "claim": "PubMed records found for drug-pair interaction query.",
                    "drug_a": a,
                    "drug_b": b,
                    "value": {
                        "query": query,
                        "pmids": ids,
                        "urls": [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in ids],
                    },
                }
            ]
        except Exception as exc:
            return [self._error("pubmed", f"{a} + {b}", exc)]

    def pubmed_single_search(self, drug_name: str | None, *, dataset_name: str | None = None, max_results: int = 3) -> List[Dict[str, Any]]:
        name = (drug_name or "").strip()
        if not name:
            return []
        context = _dataset_context(dataset_name)
        query = f'"{name}" {context}'.strip()
        try:
            response = requests.get(
                f"{PUBMED_EUTILS_BASE_URL}/esearch.fcgi",
                params={"db": "pubmed", "retmode": "json", "retmax": max_results, "term": query},
                timeout=self.timeout,
            )
            response.raise_for_status()
            ids = response.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []
            return [
                {
                    "source": "pubmed",
                    "claim": "PubMed records found for drug-property or ADMET query.",
                    "drug_name": name,
                    "dataset_name": dataset_name,
                    "value": {
                        "query": query,
                        "pmids": ids,
                        "urls": [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in ids],
                    },
                }
            ]
        except Exception as exc:
            return [self._error("pubmed", query, exc)]

    def _error(self, source: str, query: str, exc: Exception) -> Dict[str, Any]:
        return {
            "source": "external_evidence_error",
            "provider": source,
            "claim": f"{source} lookup failed.",
            "query": query,
            "error": str(exc),
        }


def _dataset_context(dataset_name: str | None) -> str:
    dataset = (dataset_name or "").strip().lower()
    if dataset == "bbbp":
        return "blood brain barrier permeability"
    if dataset == "bace":
        return "BACE inhibition"
    if dataset == "hiv":
        return "HIV activity"
    if dataset == "esol":
        return "aqueous solubility"
    if dataset == "freesolv":
        return "hydration free energy"
    if dataset == "lipophilicity":
        return "lipophilicity"
    if dataset in {"clintox", "tox21", "sider", "herg", "ames", "dili"}:
        return f"{dataset_name} toxicity ADMET"
    if dataset.startswith("cyp"):
        return f"{dataset_name} inhibition cytochrome P450"
    return "drug property ADMET"
