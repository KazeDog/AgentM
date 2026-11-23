from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import requests

from agentm.utils.io import ensure_dir, write_json, write_text


UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


@dataclass
class BiomedicalQueryContext:
    raw_request: str
    target: Optional[str] = None
    disease: Optional[str] = None
    species: str = "Homo sapiens"
    organism_id: int = 9606
    literature_queries: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BiomedicalQueryBuilder:
    """Extract target/disease hints and construct literature queries."""

    STOPWORDS = {
        "agent",
        "and",
        "drug",
        "drugs",
        "disease",
        "for",
        "in",
        "known",
        "molecule",
        "molecules",
        "protein",
        "target",
        "with",
    }

    def build(self, request: str, *, target: Optional[str] = None, disease: Optional[str] = None) -> BiomedicalQueryContext:
        text = (request or "").strip()
        inferred_target = target or self._infer_target(text)
        inferred_disease = disease or self._infer_disease(text)
        queries = self._literature_queries(inferred_target, inferred_disease, text)
        return BiomedicalQueryContext(
            raw_request=text,
            target=inferred_target,
            disease=inferred_disease,
            literature_queries=queries,
        )

    def _infer_target(self, text: str) -> Optional[str]:
        patterns = [
            r"(?:target|protein|gene|靶点|蛋白|基因)\s*[:：]?\s*([A-Za-z0-9][A-Za-z0-9_-]{1,20})",
            r"([A-Za-z0-9][A-Za-z0-9_-]{1,20})\s*(?:target|protein|gene|靶点|蛋白|基因)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip("-_ ")
                if candidate and candidate.lower() not in self.STOPWORDS:
                    return candidate

        candidates = re.findall(r"\b[A-Z][A-Z0-9]{1,8}(?:-[A-Z0-9]{1,8})?\b", text)
        for candidate in candidates:
            if candidate.lower() not in self.STOPWORDS:
                return candidate
        return None

    def _infer_disease(self, text: str) -> Optional[str]:
        patterns = [
            r"(?:disease|疾病|适应症)\s*[:：]?\s*([^,，。；;]+)",
            r"\bin\s+([^,，。；;]+)",
            r"(?:治疗|用于)\s+([^,，。；;]+)",
            r"\bfor\s+([^,，。；;]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                candidate = self._clean_phrase(match.group(1))
                if candidate:
                    return candidate
        return None

    def _literature_queries(self, target: Optional[str], disease: Optional[str], request: str) -> List[str]:
        target_text = target or ""
        disease_text = disease or ""
        base = " ".join(part for part in [target_text, disease_text] if part).strip()
        if not base:
            return [request] if request else []
        queries = [
            f"{base} drug discovery",
            f"{base} inhibitor mechanism",
            f"{base} therapeutic target",
        ]
        return list(dict.fromkeys(queries))

    def _clean_phrase(self, text: str) -> str:
        cleaned = re.sub(r"\b(?:known|drugs?|molecules?|target|protein|gene)\b", " ", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;，。；：")
        return cleaned


class UniProtClient:
    def __init__(self, *, timeout: float = 20):
        self.timeout = timeout

    def search_fasta(self, target: str | None, *, organism_id: int = 9606) -> Dict[str, Any]:
        if not target:
            return {"status": "skipped", "reason": "missing_target"}
        query = f'((gene_exact:{target}) OR (protein_name:"{target}")) AND (organism_id:{organism_id})'
        response = requests.get(
            UNIPROT_SEARCH_URL,
            params={"query": query, "format": "fasta", "size": 1},
            timeout=self.timeout,
        )
        response.raise_for_status()
        fasta = response.text.strip()
        return {
            "status": "success" if fasta else "empty",
            "target": target,
            "organism_id": organism_id,
            "query": query,
            "fasta": fasta,
            "source_url": response.url,
        }


class ChEMBLClient:
    def __init__(self, *, timeout: float = 20):
        self.timeout = timeout

    def search_target(self, target: str | None, *, limit: int = 3) -> Dict[str, Any]:
        if not target:
            return {"status": "skipped", "reason": "missing_target", "targets": []}
        response = requests.get(
            f"{CHEMBL_BASE_URL}/target/search.json",
            params={"q": target, "limit": limit},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        targets = payload.get("targets", [])
        return {"status": "success", "target": target, "targets": targets, "source_url": response.url}

    def known_molecules_for_target(self, target_chembl_id: str | None, *, limit: int = 10) -> Dict[str, Any]:
        if not target_chembl_id:
            return {"status": "skipped", "reason": "missing_target_chembl_id", "molecules": []}
        response = requests.get(
            f"{CHEMBL_BASE_URL}/activity.json",
            params={
                "target_chembl_id": target_chembl_id,
                "limit": limit,
                "pchembl_value__isnull": "false",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        activities = response.json().get("activities", [])
        molecules = []
        seen = set()
        for item in activities:
            molecule_id = item.get("molecule_chembl_id")
            if not molecule_id or molecule_id in seen:
                continue
            seen.add(molecule_id)
            molecules.append(
                {
                    "molecule_chembl_id": molecule_id,
                    "standard_type": item.get("standard_type"),
                    "standard_value": item.get("standard_value"),
                    "standard_units": item.get("standard_units"),
                    "pchembl_value": item.get("pchembl_value"),
                }
            )
        return {
            "status": "success",
            "target_chembl_id": target_chembl_id,
            "molecules": molecules,
            "source_url": response.url,
        }

    def molecule_smiles(self, molecule_chembl_id: str | None) -> Dict[str, Any]:
        if not molecule_chembl_id:
            return {"status": "skipped", "reason": "missing_molecule_chembl_id"}
        response = requests.get(
            f"{CHEMBL_BASE_URL}/molecule/{molecule_chembl_id}.json",
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        structures = payload.get("molecule_structures") or {}
        return {
            "status": "success",
            "molecule_chembl_id": molecule_chembl_id,
            "pref_name": payload.get("pref_name"),
            "canonical_smiles": structures.get("canonical_smiles"),
            "standard_inchi_key": structures.get("standard_inchi_key"),
            "source_url": response.url,
        }


class SemanticScholarClient:
    def __init__(self, *, timeout: float = 20, api_key: Optional[str] = None):
        self.timeout = timeout
        self.api_key = api_key

    def search_papers(self, query: str, *, limit: int = 5) -> Dict[str, Any]:
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        response = requests.get(
            SEMANTIC_SCHOLAR_SEARCH_URL,
            params={
                "query": query,
                "limit": limit,
                "fields": "title,abstract,url,year,authors,openAccessPdf,citationCount,externalIds",
            },
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return {
            "status": "success",
            "query": query,
            "papers": payload.get("data", []),
            "source_url": response.url,
        }

    def write_literature_records(
        self,
        paper_results: List[Dict[str, Any]],
        output_dir: str | Path,
        *,
        download_pdfs: bool = False,
    ) -> List[Dict[str, Any]]:
        target_dir = ensure_dir(output_dir)
        records: List[Dict[str, Any]] = []
        counter = 0
        for result in paper_results:
            query = result.get("query")
            for paper in result.get("papers", []):
                if not isinstance(paper, dict):
                    continue
                counter += 1
                paper_id = self._paper_id(paper, counter)
                text_path = target_dir / f"{counter:03d}_{paper_id}.txt"
                text = self._paper_text(paper, query=query)
                write_text(text_path, text)
                record = {
                    "paper_id": paper.get("paperId"),
                    "title": paper.get("title"),
                    "year": paper.get("year"),
                    "url": paper.get("url"),
                    "query": query,
                    "text_path": str(text_path),
                    "open_access_pdf": paper.get("openAccessPdf"),
                    "citation_count": paper.get("citationCount"),
                }
                pdf_url = ((paper.get("openAccessPdf") or {}) if isinstance(paper.get("openAccessPdf"), dict) else {}).get("url")
                if download_pdfs and pdf_url:
                    pdf_path = target_dir / f"{counter:03d}_{paper_id}.pdf"
                    pdf_result = self._download_pdf(pdf_url, pdf_path)
                    record.update(pdf_result)
                records.append(record)
        write_json(target_dir / "literature_records.json", {"records": records})
        return records

    def _paper_id(self, paper: Dict[str, Any], counter: int) -> str:
        raw = str(paper.get("paperId") or paper.get("externalIds", {}).get("DOI") or counter)
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)[:80]

    def _paper_text(self, paper: Dict[str, Any], *, query: Optional[str]) -> str:
        authors = paper.get("authors") or []
        author_text = ", ".join(str(item.get("name")) for item in authors if isinstance(item, dict) and item.get("name"))
        return "\n".join(
            [
                f"Title: {paper.get('title') or ''}",
                f"Year: {paper.get('year') or ''}",
                f"Authors: {author_text}",
                f"URL: {paper.get('url') or ''}",
                f"Query: {query or ''}",
                "",
                str(paper.get("abstract") or ""),
            ]
        ).strip() + "\n"

    def _download_pdf(self, pdf_url: str, pdf_path: Path) -> Dict[str, Any]:
        try:
            response = requests.get(pdf_url, timeout=self.timeout)
            response.raise_for_status()
            pdf_path.write_bytes(response.content)
            return {"pdf_path": str(pdf_path), "pdf_download_status": "success", "pdf_url": pdf_url}
        except Exception as exc:
            return {"pdf_download_status": "failed", "pdf_url": pdf_url, "pdf_error": str(exc)}
