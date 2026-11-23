from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import quote
import xml.etree.ElementTree as ET

import requests

from agentm.agents.rag_evidence_agent import RAGEvidenceAgent, citation_pointers_from_text
from agentm.defaults import DEFAULT_DDI_RAG_CORPUS_DIR, DEFAULT_RAG_BACKEND, DEFAULT_RAG_EMBEDDING_MODEL_PATH, DEFAULT_RAG_TOP_K
from agentm.schema import DDIPairFeatureBundle
from agentm.utils.io import ensure_dir, write_json, write_text


PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov"
PUBMED_EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DAILYMED_BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

DAILYMED_DDI_SECTION_KEYWORDS = (
    "boxed warning",
    "contraindications",
    "warnings and precautions",
    "warnings",
    "drug interactions",
    "adverse reactions",
    "clinical pharmacology",
)


@dataclass
class DDIPairCorpusSummary:
    drug_a: str
    drug_b: str
    pair_key: str
    pair_dir: str
    pubchem_records: int = 0
    dailymed_labels: int = 0
    pubmed_records: int = 0
    failures: int = 0
    built_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DDIRAGEvidenceAgent:
    """Build and retrieve pair-level DDI evidence from PubChem, DailyMed, and PubMed."""

    def __init__(
        self,
        output_dir: str | Path = "outputs/ddi_rag",
        *,
        corpus_dir: str | Path = DEFAULT_DDI_RAG_CORPUS_DIR,
        build_on_demand: bool = True,
        timeout: float = 12.0,
        request_delay: float = 0.1,
        max_pubmed: int = 8,
        max_labels_per_drug: int = 2,
        top_k: int = DEFAULT_RAG_TOP_K,
        rag_backend: str = DEFAULT_RAG_BACKEND,
        embedding_model_name: str = DEFAULT_RAG_EMBEDDING_MODEL_PATH,
        embedding_backend: str = "auto",
        embedding_device: str = "auto",
        embedding_local_files_only: bool = True,
        embedding_batch_size: int = 16,
        use_llm: bool = False,
        llm_profile: Optional[str] = None,
    ):
        self.output_dir = ensure_dir(output_dir)
        self.corpus_dir = ensure_dir(corpus_dir)
        self.build_on_demand = bool(build_on_demand)
        self.timeout = float(timeout)
        self.request_delay = float(request_delay)
        self.max_pubmed = int(max_pubmed)
        self.max_labels_per_drug = int(max_labels_per_drug)
        self.top_k = int(top_k)
        self.rag_backend = rag_backend
        self.embedding_model_name = embedding_model_name
        self.embedding_backend = embedding_backend
        self.embedding_device = embedding_device
        self.embedding_local_files_only = bool(embedding_local_files_only)
        self.embedding_batch_size = int(embedding_batch_size)
        self.use_llm = bool(use_llm)
        self.llm_profile = llm_profile
        self.session = requests.Session()
        self.failures_path = self.corpus_dir / "failures.jsonl"

    def retrieve_for_pair(
        self,
        pair: DDIPairFeatureBundle,
        *,
        query: str,
        context_metadata: Optional[Mapping[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        drug_a = _query_name(pair.drug_a_features.molecule)
        drug_b = _query_name(pair.drug_b_features.molecule)
        return self.retrieve(
            drug_a=drug_a,
            drug_b=drug_b,
            drug_a_smiles=pair.drug_a_features.molecule.canonical_smiles,
            drug_b_smiles=pair.drug_b_features.molecule.canonical_smiles,
            query=query,
            context_metadata=context_metadata,
            top_k=top_k,
        )

    def retrieve(
        self,
        *,
        drug_a: str,
        drug_b: str,
        query: str,
        drug_a_smiles: Optional[str] = None,
        drug_b_smiles: Optional[str] = None,
        context_metadata: Optional[Mapping[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        warnings: List[str] = []
        pair_key = pair_key_for_drugs(drug_a, drug_b)
        pair_dir = self.corpus_dir / "pairs" / pair_key
        if not self._pair_has_corpus(pair_dir):
            if self.build_on_demand:
                summary = self.build_pair_corpus(
                    drug_a=drug_a,
                    drug_b=drug_b,
                    drug_a_smiles=drug_a_smiles,
                    drug_b_smiles=drug_b_smiles,
                )
                warnings.extend(self._summary_warnings(summary))
            else:
                warnings.append(f"ddi_rag_pair_corpus_missing:{pair_key}")
        if not self._pair_has_corpus(pair_dir):
            return self._write_result(
                {
                    "query": query,
                    "provider": "ddi_rag_unavailable",
                    "backend": self.rag_backend,
                    "pair_key": pair_key,
                    "pair_dir": str(pair_dir),
                    "evidence_items": [],
                    "retrieved_context": [],
                    "warnings": warnings or ["ddi_rag_corpus_unavailable"],
                    "artifacts": {},
                }
            )

        rag = RAGEvidenceAgent(
            output_dir=self.output_dir / "retrieval",
            corpus_dirs=[pair_dir],
            top_k=top_k or self.top_k,
            rag_backend=self.rag_backend,
            embedding_model_name=self.embedding_model_name,
            embedding_backend=self.embedding_backend,
            embedding_device=self.embedding_device,
            embedding_local_files_only=self.embedding_local_files_only,
            embedding_batch_size=self.embedding_batch_size,
            use_llm=self.use_llm,
            llm_profile=self.llm_profile,
        )
        result = rag.retrieve(
            query=query,
            context_metadata={**dict(context_metadata or {}), "dataset_name": "DDI", "drug_a": drug_a, "drug_b": drug_b},
            top_k=top_k or self.top_k,
        )
        retrieved = list(result.get("retrieved_context", []) or [])
        evidence_items = self._evidence_items(
            query=query,
            retrieved=retrieved,
            drug_a=drug_a,
            drug_b=drug_b,
            pair_key=pair_key,
            pair_dir=pair_dir,
        )
        warnings.extend(str(item) for item in result.get("warnings", []) or [])
        return self._write_result(
            {
                "query": query,
                "provider": f"ddi_rag:{result.get('provider')}",
                "backend": result.get("backend", self.rag_backend),
                "pair_key": pair_key,
                "pair_dir": str(pair_dir),
                "n_documents": result.get("n_documents", 0),
                "n_chunks": result.get("n_chunks", 0),
                "evidence_items": evidence_items,
                "retrieved_context": retrieved,
                "warnings": warnings,
                "artifacts": {},
            }
        )

    def build_pair_corpus(
        self,
        *,
        drug_a: str,
        drug_b: str,
        drug_a_smiles: Optional[str] = None,
        drug_b_smiles: Optional[str] = None,
    ) -> Dict[str, Any]:
        retrieved_at = datetime.now(timezone.utc).isoformat()
        pair_key = pair_key_for_drugs(drug_a, drug_b)
        pair_dir = ensure_dir(self.corpus_dir / "pairs" / pair_key)
        failures_before = self._count_pair_failures(pair_key)
        pubchem_records = []
        for name, smiles, role in [(drug_a, drug_a_smiles, "drug_a"), (drug_b, drug_b_smiles, "drug_b")]:
            try:
                pubchem_records.append(self._pubchem_identity(name=name, smiles=smiles, role=role, retrieved_at=retrieved_at))
            except Exception as exc:
                self._failure(pair_key, "pubchem", name or smiles or role, exc)
            self._sleep()
        write_text(pair_dir / "pubchem_identity.md", self._pubchem_markdown(drug_a, drug_b, pubchem_records, retrieved_at))
        evidence_names = self._evidence_names_from_pubchem(drug_a, drug_b, pubchem_records)

        dailymed_labels = []
        for role, candidates in [("drug_a", evidence_names["drug_a_candidates"]), ("drug_b", evidence_names["drug_b_candidates"])]:
            role_labels: List[Dict[str, Any]] = []
            for name in candidates:
                try:
                    role_labels = self._dailymed_labels(name, role=role, retrieved_at=retrieved_at)
                    if role_labels:
                        dailymed_labels.extend(role_labels)
                        break
                except Exception as exc:
                    self._failure(pair_key, "dailymed", name, exc)
                self._sleep()
        write_text(pair_dir / "dailymed_interactions.md", self._dailymed_markdown(drug_a, drug_b, dailymed_labels, retrieved_at))

        pubmed_records = []
        pubmed_queries = ddi_pubmed_query_plan(
            evidence_names["drug_a"],
            evidence_names["drug_b"],
            drug_a_aliases=evidence_names.get("drug_a_candidates"),
            drug_b_aliases=evidence_names.get("drug_b_candidates"),
        )
        searched_queries: List[Dict[str, Any]] = []
        seen_pmids = set()
        for query_info in pubmed_queries:
            if len(seen_pmids) >= self.max_pubmed:
                break
            pubmed_query = str(query_info.get("query") or "")
            try:
                pmids = self._pubmed_search(pubmed_query, retmax=max(1, self.max_pubmed - len(seen_pmids)))
                new_pmids = [pmid for pmid in pmids if pmid not in seen_pmids]
                seen_pmids.update(new_pmids)
                searched_queries.append({**query_info, "n_pmids": len(pmids), "n_new_pmids": len(new_pmids), "pmids": pmids})
                if new_pmids:
                    pubmed_records.extend(self._pubmed_fetch(new_pmids, query=pubmed_query, retrieved_at=retrieved_at))
            except Exception as exc:
                searched_queries.append({**query_info, "n_pmids": 0, "n_new_pmids": 0, "error": str(exc)})
                self._failure(pair_key, "pubmed", pubmed_query, exc)
            self._sleep()
        write_json(pair_dir / "query_expansion.json", {"pubmed_queries": searched_queries, "retrieved_at": retrieved_at})
        pubmed_query_summary = " ; ".join(str(item.get("query")) for item in searched_queries) or ddi_pubmed_query(evidence_names["drug_a"], evidence_names["drug_b"])
        write_text(pair_dir / "pubmed_pair_literature.md", self._pubmed_markdown(drug_a, drug_b, pubmed_query_summary, pubmed_records, retrieved_at))

        summary = DDIPairCorpusSummary(
            drug_a=drug_a,
            drug_b=drug_b,
            pair_key=pair_key,
            pair_dir=str(pair_dir),
            pubchem_records=len(pubchem_records),
            dailymed_labels=len(dailymed_labels),
            pubmed_records=len(pubmed_records),
            failures=self._count_pair_failures(pair_key) - failures_before,
            built_at=retrieved_at,
        ).to_dict()
        summary["evidence_drug_a"] = evidence_names["drug_a"]
        summary["evidence_drug_b"] = evidence_names["drug_b"]
        summary["pubmed_query_expansion"] = searched_queries
        write_json(pair_dir / "pair_manifest.json", summary)
        self._write_global_manifest()
        return summary

    def _pubchem_identity(self, *, name: str, smiles: Optional[str], role: str, retrieved_at: str) -> Dict[str, Any]:
        cids: List[Any] = []
        query_value = (name or "").strip()
        if query_value:
            try:
                response = self.session.get(
                    f"{PUBCHEM_BASE_URL}/rest/pug/compound/name/{quote(query_value)}/cids/JSON",
                    timeout=self.timeout,
                )
                response.raise_for_status()
                cids = response.json().get("IdentifierList", {}).get("CID", []) or []
            except Exception:
                cids = []
        if not cids and smiles:
            response = self.session.get(
                f"{PUBCHEM_BASE_URL}/rest/pug/compound/smiles/{quote(smiles, safe='')}/cids/JSON",
                timeout=self.timeout,
            )
            response.raise_for_status()
            cids = response.json().get("IdentifierList", {}).get("CID", []) or []
        if not cids:
            raise ValueError("PubChem returned no CID.")
        cid = cids[0]
        properties = self._pubchem_properties(cid)
        synonyms = self._pubchem_synonyms(cid)
        return {
            "source": "pubchem",
            "role": role,
            "query": query_value or smiles,
            "cid": cid,
            "url": f"{PUBCHEM_BASE_URL}/compound/{cid}",
            "properties": properties,
            "synonyms": synonyms[:20],
            "retrieved_at": retrieved_at,
            "source_note": "PubChem PUG REST identifiers, synonyms, and computed properties.",
        }

    def _evidence_names_from_pubchem(self, drug_a: str, drug_b: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_role = {record.get("role"): record for record in records}
        a_candidates = self._candidate_names(drug_a, by_role.get("drug_a"))
        b_candidates = self._candidate_names(drug_b, by_role.get("drug_b"))
        return {
            "drug_a": a_candidates[0] if a_candidates else drug_a,
            "drug_b": b_candidates[0] if b_candidates else drug_b,
            "drug_a_candidates": a_candidates or [drug_a],
            "drug_b_candidates": b_candidates or [drug_b],
        }

    def _candidate_names(self, original: str, record: Optional[Dict[str, Any]]) -> List[str]:
        candidates: List[str] = []
        if original and not _looks_like_database_id(original):
            candidates.append(original)
        if record:
            for name in record.get("synonyms", []) or []:
                text = str(name).strip()
                if _looks_like_drug_name(text) and text not in candidates:
                    candidates.append(text)
        if original and original not in candidates:
            candidates.append(original)
        return candidates[:5]

    def _pubchem_properties(self, cid: Any) -> Dict[str, Any]:
        names = "CanonicalSMILES,IUPACName,MolecularFormula,MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount"
        response = self.session.get(f"{PUBCHEM_BASE_URL}/rest/pug/compound/cid/{cid}/property/{names}/JSON", timeout=self.timeout)
        response.raise_for_status()
        rows = response.json().get("PropertyTable", {}).get("Properties", []) or []
        return rows[0] if rows else {}

    def _pubchem_synonyms(self, cid: Any) -> List[str]:
        response = self.session.get(f"{PUBCHEM_BASE_URL}/rest/pug/compound/cid/{cid}/synonyms/JSON", timeout=self.timeout)
        response.raise_for_status()
        rows = response.json().get("InformationList", {}).get("Information", []) or []
        return [str(item) for item in (rows[0].get("Synonym", []) if rows else []) if item]

    def _dailymed_labels(self, drug_name: str, *, role: str, retrieved_at: str) -> List[Dict[str, Any]]:
        response = self.session.get(
            f"{DAILYMED_BASE_URL}/spls.json",
            params={"drug_name": drug_name, "page": 1, "pageSize": self.max_labels_per_drug},
            timeout=self.timeout,
        )
        response.raise_for_status()
        rows = response.json().get("data") or response.json().get("results") or []
        labels = []
        for row in rows[: self.max_labels_per_drug] if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            setid = row.get("setid") or row.get("set_id")
            sections: List[Dict[str, str]] = []
            section_error = None
            if setid:
                try:
                    sections = self._dailymed_sections(setid)
                except Exception as exc:
                    section_error = str(exc)
            labels.append(
                {
                    "source": "dailymed",
                    "role": role,
                    "query": drug_name,
                    "setid": setid,
                    "title": row.get("title"),
                    "published_date": row.get("published_date"),
                    "url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}" if setid else None,
                    "sections": sections,
                    "section_error": section_error,
                    "retrieved_at": retrieved_at,
                    "source_note": "DailyMed public SPL label metadata and selected DDI-relevant section excerpts.",
                }
            )
        return labels

    def _dailymed_sections(self, setid: str) -> List[Dict[str, str]]:
        response = self.session.get(f"{DAILYMED_BASE_URL}/spls/{setid}.xml", timeout=self.timeout)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        sections: List[Dict[str, str]] = []
        seen = set()
        for section in root.iter():
            if _local_name(section.tag) != "section":
                continue
            title = _section_title(section)
            normalized = title.lower()
            if not title or normalized in seen:
                continue
            if not any(keyword in normalized for keyword in DAILYMED_DDI_SECTION_KEYWORDS):
                continue
            text = _section_text(section, title)
            if not text:
                continue
            seen.add(normalized)
            sections.append({"title": title, "text": text[:1600]})
            if len(sections) >= 8:
                break
        return sections

    def _pubmed_search(self, query: str, *, retmax: Optional[int] = None) -> List[str]:
        response = self.session.get(
            f"{PUBMED_EUTILS_BASE_URL}/esearch.fcgi",
            params={"db": "pubmed", "retmode": "json", "retmax": retmax or self.max_pubmed, "term": query},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("esearchresult", {}).get("idlist", []) or []

    def _pubmed_fetch(self, pmids: List[str], *, query: str, retrieved_at: str) -> List[Dict[str, Any]]:
        if not pmids:
            return []
        response = self.session.get(
            f"{PUBMED_EUTILS_BASE_URL}/efetch.fcgi",
            params={"db": "pubmed", "retmode": "xml", "id": ",".join(pmids)},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return parse_pubmed_pair_xml(response.text, query=query, retrieved_at=retrieved_at)

    def _pubchem_markdown(self, drug_a: str, drug_b: str, records: List[Dict[str, Any]], retrieved_at: str) -> str:
        lines = [
            f"# PubChem Identity Evidence for {drug_a} + {drug_b}",
            "",
            "- source: PubChem",
            "- task: DDI",
            f"- query: {drug_a} + {drug_b}",
            f"- retrieved_at: {retrieved_at}",
            "- license_note: PubChem public compound identifiers, synonyms, and computed properties are stored for local DDI retrieval.",
            "",
        ]
        for record in records:
            props = record.get("properties") or {}
            lines.extend(
                [
                    f"## {record.get('role')}: CID {record.get('cid')}",
                    "",
                    f"- source: PubChem",
                    f"- task: DDI",
                    f"- query: {record.get('query')}",
                    f"- url: {record.get('url')}",
                    f"- retrieved_at: {record.get('retrieved_at')}",
                    f"- canonical_smiles: `{props.get('CanonicalSMILES')}`",
                    f"- properties: MW={props.get('MolecularWeight')}, XLogP={props.get('XLogP')}, TPSA={props.get('TPSA')}, HBD={props.get('HBondDonorCount')}, HBA={props.get('HBondAcceptorCount')}",
                    f"- synonyms: {', '.join(record.get('synonyms', [])[:12])}",
                    "",
                ]
            )
        return "\n".join(lines).rstrip() + "\n"

    def _dailymed_markdown(self, drug_a: str, drug_b: str, labels: List[Dict[str, Any]], retrieved_at: str) -> str:
        lines = [
            f"# DailyMed DDI Label Evidence for {drug_a} + {drug_b}",
            "",
            "- source: DailyMed",
            "- task: DDI",
            f"- query: {drug_a} + {drug_b}",
            f"- retrieved_at: {retrieved_at}",
            "- license_note: DailyMed public SPL label metadata and selected interaction/warning section excerpts are stored for retrieval.",
            "",
        ]
        if not labels:
            lines.append("No DailyMed labels were found for the pair drug names.")
            return "\n".join(lines).rstrip() + "\n"
        for label in labels:
            lines.extend(
                [
                    f"## {label.get('role')}: {label.get('title') or label.get('setid')}",
                    "",
                    f"- source: DailyMed",
                    f"- task: DDI",
                    f"- query: {label.get('query')}",
                    f"- setid: {label.get('setid')}",
                    f"- url: {label.get('url')}",
                    f"- published_date: {label.get('published_date') or ''}",
                    f"- retrieved_at: {label.get('retrieved_at')}",
                    "",
                ]
            )
            for section in label.get("sections", [])[:8]:
                lines.extend([f"### {section.get('title')}", "", str(section.get("text") or "").strip(), ""])
            if label.get("section_error"):
                lines.extend([f"- section_error: {label.get('section_error')}", ""])
        return "\n".join(lines).rstrip() + "\n"

    def _pubmed_markdown(self, drug_a: str, drug_b: str, query: str, records: List[Dict[str, Any]], retrieved_at: str) -> str:
        lines = [
            f"# PubMed Pair Literature for {drug_a} + {drug_b}",
            "",
            "- source: PubMed",
            "- task: DDI",
            f"- query: {query}",
            f"- url: https://pubmed.ncbi.nlm.nih.gov/?term={quote(query)}",
            f"- retrieved_at: {retrieved_at}",
            "- license_note: PubMed titles and abstracts are stored as bibliographic metadata/abstract snippets for retrieval; no full-text article body is stored.",
            "",
        ]
        for record in records:
            lines.extend(
                [
                    f"## PMID {record.get('pmid')}: {record.get('title') or 'Untitled'}",
                    "",
                    f"- source: PubMed",
                    "- task: DDI",
                    f"- query: {query}",
                    f"- url: {record.get('url')}",
                    f"- journal: {record.get('journal') or ''}",
                    f"- year: {record.get('year') or ''}",
                    f"- retrieved_at: {record.get('retrieved_at')}",
                    "",
                    str(record.get("abstract") or "No abstract available.").strip(),
                    "",
                ]
            )
        return "\n".join(lines).rstrip() + "\n"

    def _evidence_items(
        self,
        *,
        query: str,
        retrieved: List[Dict[str, Any]],
        drug_a: str,
        drug_b: str,
        pair_key: str,
        pair_dir: Path,
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for item in retrieved:
            source = _source_from_path(item.get("source_path"))
            snippet = re.sub(r"\s+", " ", str(item.get("snippet") or "")).strip()
            pointers = citation_pointers_from_text(snippet, source_path=item.get("source_path"))
            items.append(
                {
                    "source": source,
                    "claim": f"DDI RAG retrieved {source} context: {snippet[:240]}",
                    "drug_a": drug_a,
                    "drug_b": drug_b,
                    "value": {
                        "query": query,
                        "query_expansion": _query_expansion_from_pair_dir(pair_dir),
                        "rank": item.get("rank"),
                        "score": item.get("score"),
                        "chunk_id": item.get("chunk_id"),
                        "doc_id": item.get("doc_id"),
                        "source_path": item.get("source_path"),
                        "snippet": snippet,
                        "pair_key": pair_key,
                        "pair_dir": str(pair_dir),
                        **pointers,
                    },
                    "evidence_direction": "neutral",
                    "evidence_weight": 0.0,
                    "source_reliability": 0.55,
                    "evidence_scoring_rationale": "Retrieved by pair-level DDI RAG; directional scoring is applied later when possible.",
                }
            )
        return items

    def _write_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        json_path = self.output_dir / "ddi_rag_evidence.json"
        md_path = self.output_dir / "ddi_rag_evidence.md"
        result.setdefault("artifacts", {})
        result["artifacts"].update({"rag_evidence_path": str(json_path), "rag_evidence_markdown_path": str(md_path)})
        write_json(json_path, result)
        lines = [
            "# DDI RAG Evidence",
            "",
            f"**Query:** {result.get('query') or 'n/a'}",
            f"**Provider:** {result.get('provider')}",
            f"**Backend:** {result.get('backend')}",
            f"**Pair key:** `{result.get('pair_key')}`",
            f"**Pair corpus:** `{result.get('pair_dir')}`",
            "",
            "## Retrieved Context",
        ]
        context = list(result.get("retrieved_context", []) or [])
        if context:
            lines.append("| Rank | Score | Source | Chunk | Snippet |")
            lines.append("|---:|---:|---|---|---|")
            for item in context:
                lines.append(
                    "| {rank} | {score:.4f} | `{source}` | `{chunk}` | {snippet} |".format(
                        rank=item.get("rank") or 0,
                        score=float(item.get("score") or 0.0),
                        source=_pipe(item.get("source_path")),
                        chunk=_pipe(item.get("chunk_id")),
                        snippet=_pipe(str(item.get("snippet") or "")[:240]),
                    )
                )
        else:
            lines.append("- No retrieved DDI context.")
        warnings = list(result.get("warnings", []) or [])
        if warnings:
            lines.extend(["", "## Warnings"])
            lines.extend(f"- {warning}" for warning in warnings)
        write_text(md_path, "\n".join(lines).rstrip() + "\n")
        return result

    def _write_global_manifest(self) -> Path:
        pairs = []
        for path in sorted((self.corpus_dir / "pairs").glob("*/pair_manifest.json")) if (self.corpus_dir / "pairs").exists() else []:
            try:
                pairs.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return write_json(
            self.corpus_dir / "manifest.json",
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "source_set": ["PubChem", "DailyMed", "PubMed"],
                "task": "DDI",
                "pairs": pairs,
            },
        )

    def _pair_has_corpus(self, pair_dir: Path) -> bool:
        return any((pair_dir / name).exists() for name in ["pubchem_identity.md", "dailymed_interactions.md", "pubmed_pair_literature.md"])

    def _summary_warnings(self, summary: Mapping[str, Any]) -> List[str]:
        warnings = []
        if int(summary.get("pubchem_records", 0) or 0) == 0:
            warnings.append("ddi_rag_pubchem_empty")
        if int(summary.get("dailymed_labels", 0) or 0) == 0:
            warnings.append("ddi_rag_dailymed_empty")
        if int(summary.get("pubmed_records", 0) or 0) == 0:
            warnings.append("ddi_rag_pubmed_empty")
        if int(summary.get("failures", 0) or 0) > 0:
            warnings.append(f"ddi_rag_build_failures:{summary.get('failures')}")
        return warnings

    def _failure(self, pair_key: str, source: str, query: str, exc: Exception) -> None:
        ensure_dir(self.failures_path.parent)
        payload = {
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "pair_key": pair_key,
            "source": source,
            "query": query,
            "error": str(exc),
        }
        with self.failures_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _count_pair_failures(self, pair_key: str) -> int:
        if not self.failures_path.exists():
            return 0
        count = 0
        for line in self.failures_path.read_text(encoding="utf-8").splitlines():
            try:
                count += 1 if json.loads(line).get("pair_key") == pair_key else 0
            except Exception:
                continue
        return count

    def _sleep(self) -> None:
        if self.request_delay > 0:
            time.sleep(self.request_delay)


def ddi_pubmed_query(drug_a: str, drug_b: str) -> str:
    return ddi_pubmed_query_plan(drug_a, drug_b)[0]["query"]


def ddi_pubmed_query_plan(
    drug_a: str,
    drug_b: str,
    *,
    drug_a_aliases: Optional[Iterable[str]] = None,
    drug_b_aliases: Optional[Iterable[str]] = None,
) -> List[Dict[str, str]]:
    a = _pubmed_alias_clause(drug_a, drug_a_aliases)
    b = _pubmed_alias_clause(drug_b, drug_b_aliases)
    return [
        {
            "query_type": "direct_pair_evidence",
            "query": f"({a} AND {b}) AND (drug interaction OR drug interactions OR contraindication OR coadministration)",
        },
        {
            "query_type": "pharmacokinetic_mechanism",
            "query": f"({a} AND {b}) AND (pharmacokinetic OR pharmacodynamics OR exposure OR AUC OR clearance OR metabolism)",
        },
        {
            "query_type": "cyp_transporter_mechanism",
            "query": f"({a} AND {b}) AND (CYP OR cytochrome OR CYP3A4 OR CYP2C9 OR CYP2D6 OR P-glycoprotein OR P-gp OR transporter)",
        },
        {
            "query_type": "adverse_event_overlap",
            "query": f"({a} AND {b}) AND (toxicity OR adverse event OR warning OR QT OR bleeding OR hepatic OR renal)",
        },
        {
            "query_type": "broad_pair_literature",
            "query": f"{a} AND {b}",
        },
    ]


def _pubmed_alias_clause(primary: str, aliases: Optional[Iterable[str]] = None) -> str:
    terms: List[str] = []
    for value in [primary, *(aliases or [])]:
        text = str(value or "").strip()
        if not text or len(text) > 80 or text in terms:
            continue
        if _looks_like_database_id(text):
            continue
        terms.append(text)
        if len(terms) >= 3:
            break
    if not terms:
        terms = [str(primary or "").strip() or "unknown"]
    return "(" + " OR ".join(f'"{term}"' for term in terms) + ")"


def _query_expansion_from_pair_dir(pair_dir: Path) -> List[Dict[str, Any]]:
    path = pair_dir / "query_expansion.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        queries = payload.get("pubmed_queries")
        return queries if isinstance(queries, list) else []
    except Exception:
        return []


def pair_key_for_drugs(drug_a: str, drug_b: str) -> str:
    left, right = sorted([_slug(drug_a), _slug(drug_b)])
    return f"{left}__{right}"[:160]


def _looks_like_database_id(value: str) -> bool:
    text = str(value).strip()
    return bool(re.fullmatch(r"(DB|CHEMBL|CID)[:_-]?\d+[A-Z0-9]*", text, flags=re.IGNORECASE))


def _looks_like_drug_name(value: str) -> bool:
    text = str(value).strip()
    if not text or len(text) < 3 or len(text) > 64:
        return False
    if _looks_like_database_id(text):
        return False
    if any(char in text for char in ["=", "#", "@", "[", "]", "\\", "/"]):
        return False
    if len(re.findall(r"[A-Za-z]", text)) < 3:
        return False
    if text.lower().startswith(("cas-", "sc-", "cid ")):
        return False
    return True


def parse_pubmed_pair_xml(xml_text: str, *, query: str, retrieved_at: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    records: List[Dict[str, Any]] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _xml_text(article, ".//PMID")
        title = _xml_text(article, ".//ArticleTitle")
        journal = _xml_text(article, ".//Journal/Title") or _xml_text(article, ".//Journal/ISOAbbreviation")
        year = _xml_text(article, ".//PubDate/Year") or _xml_text(article, ".//PubDate/MedlineDate")
        abstract_parts = [
            _clean_xml_text(" ".join(part.itertext()))
            for part in article.findall(".//AbstractText")
            if _clean_xml_text(" ".join(part.itertext()))
        ]
        records.append(
            {
                "source": "pubmed",
                "task": "DDI",
                "query": query,
                "pmid": pmid,
                "title": _clean_xml_text(title),
                "journal": _clean_xml_text(journal),
                "year": _clean_xml_text(year),
                "abstract": _clean_xml_text(" ".join(abstract_parts)),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                "retrieved_at": retrieved_at,
                "source_note": "PubMed title and abstract metadata for a drug-pair DDI query.",
            }
        )
    return records


def _query_name(molecule: Any) -> str:
    for value in [getattr(molecule, "name", None), getattr(molecule, "original_input", None), getattr(molecule, "canonical_smiles", None)]:
        text = str(value or "").strip()
        if text:
            return text
    return "unknown"


def _source_from_path(path: Any) -> str:
    text = str(path or "").lower()
    if "dailymed" in text:
        return "dailymed"
    if "pubmed" in text:
        return "pubmed"
    if "pubchem" in text:
        return "pubchem"
    return "rag_literature"


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip().lower()).strip("_")
    return text[:72] or "unknown"


def _xml_text(node: ET.Element, path: str) -> str:
    found = node.find(path)
    return "".join(found.itertext()) if found is not None else ""


def _local_name(tag: Any) -> str:
    text = str(tag)
    return text.rsplit("}", 1)[-1] if "}" in text else text


def _section_title(section: ET.Element) -> str:
    for child in list(section):
        if _local_name(child.tag) == "title":
            return _clean_xml_text(" ".join(child.itertext()))
    return ""


def _section_text(section: ET.Element, title: str) -> str:
    chunks = []
    for child in list(section):
        if _local_name(child.tag) == "title":
            continue
        text = _clean_xml_text(" ".join(child.itertext()))
        if text:
            chunks.append(text)
    text = _clean_xml_text(" ".join(chunks))
    if title and text.lower().startswith(title.lower()):
        text = _clean_xml_text(text[len(title) :])
    return text


def _clean_xml_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _pipe(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()
