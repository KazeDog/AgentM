from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import quote
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from agentm.data.tdc_datasets import TDC_TASKS, TDCTaskSpec, expand_task_names, task_spec
from agentm.utils.io import ensure_dir, write_json, write_text


PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov"
PUBMED_EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DAILYMED_BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

DAILYMED_SUMMARY_SECTION_KEYWORDS = (
    "boxed warning",
    "warnings and precautions",
    "warnings",
    "adverse reactions",
    "drug interactions",
    "indications and usage",
)


TASK_SYNONYMS: Dict[str, List[str]] = {
    "BBB_Martins": ["blood brain barrier", "BBB permeability", "CNS penetration"],
    "HIA_Hou": ["human intestinal absorption", "oral absorption", "intestinal permeability"],
    "Pgp_Broccatelli": ["P-glycoprotein", "Pgp inhibition", "MDR1 transporter"],
    "Bioavailability_Ma": ["oral bioavailability", "systemic exposure"],
    "CYP1A2_Veith": ["CYP1A2 inhibition", "cytochrome P450 1A2"],
    "CYP2C9_Veith": ["CYP2C9 inhibition", "cytochrome P450 2C9"],
    "CYP2C19_Veith": ["CYP2C19 inhibition", "cytochrome P450 2C19"],
    "CYP2D6_Veith": ["CYP2D6 inhibition", "cytochrome P450 2D6"],
    "CYP3A4_Veith": ["CYP3A4 inhibition", "cytochrome P450 3A4"],
    "Caco2_Wang": ["Caco-2 permeability", "intestinal epithelial permeability"],
    "Lipophilicity_AstraZeneca": ["lipophilicity", "logD", "logP"],
    "Solubility_AqSolDB": ["aqueous solubility", "drug solubility"],
    "HydrationFreeEnergy_FreeSolv": ["hydration free energy", "solvation free energy"],
    "Half_Life_Obach": ["drug half-life", "pharmacokinetic half-life"],
    "Clearance_Hepatocyte_AZ": ["hepatocyte clearance", "intrinsic clearance"],
    "Clearance_Microsome_AZ": ["microsomal clearance", "liver microsome clearance"],
    "hERG": ["hERG blockade", "KCNH2", "QT prolongation"],
    "AMES": ["Ames mutagenicity", "bacterial mutagenicity"],
    "DILI": ["drug induced liver injury", "hepatotoxicity"],
    "ClinTox": ["clinical toxicity", "drug toxicity"],
    "LD50_Zhu": ["acute toxicity", "LD50"],
}


@dataclass
class RealRAGCorpusConfig:
    output_dir: Path = Path("data/rag_corpus/tdc_real")
    data_root: Path = Path("data/tdc")
    tasks: List[str] = field(default_factory=lambda: list(TDC_TASKS))
    max_pubmed: int = 20
    max_compounds: int = 8
    max_dailymed_labels: int = 2
    timeout: float = 15.0
    request_delay: float = 0.1
    random_seed: int = 13
    ncbi_api_key: Optional[str] = None


@dataclass
class TaskCorpusSummary:
    task: str
    collection: str
    mode: str
    label_description: str
    pubmed_records: int = 0
    pubchem_compounds: int = 0
    dailymed_labels: int = 0
    failures: int = 0
    pubmed_path: Optional[str] = None
    pubchem_path: Optional[str] = None
    dailymed_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RealRAGCorpusBuilder:
    """Build a local, source-grounded RAG corpus for TDC ADMET/Tox tasks."""

    def __init__(self, config: RealRAGCorpusConfig):
        self.config = config
        self.session = requests.Session()
        self.retrieved_at = datetime.now(timezone.utc).isoformat()
        self.output_dir = ensure_dir(config.output_dir)
        self.failures_path = self.output_dir / "failures.jsonl"

    def build(self) -> Dict[str, Any]:
        summaries: List[TaskCorpusSummary] = []
        for task_name in self.config.tasks:
            spec = task_spec(task_name)
            task_dir = ensure_dir(self.output_dir / "tasks" / task_name)
            summary = TaskCorpusSummary(
                task=spec.name,
                collection=spec.collection,
                mode=spec.mode,
                label_description=spec.label_description,
            )
            pubmed_records = self._safe_pubmed(spec, task_dir)
            summary.pubmed_records = len(pubmed_records)
            summary.pubmed_path = str(task_dir / "pubmed_literature.md")
            compounds = self._safe_pubchem(spec, task_dir)
            summary.pubchem_compounds = len(compounds)
            summary.pubchem_path = str(task_dir / "pubchem_compounds.md")
            labels = self._safe_dailymed(spec, task_dir, compounds)
            summary.dailymed_labels = len(labels)
            summary.dailymed_path = str(task_dir / "dailymed_labels.md")
            summary.failures = self._count_task_failures(spec.name)
            summaries.append(summary)

        manifest = self._write_manifest(summaries)
        csv_path, md_path = self._write_coverage(summaries)
        return {
            "output_dir": str(self.output_dir),
            "manifest_path": str(manifest),
            "coverage_summary_csv": str(csv_path),
            "coverage_summary_md": str(md_path),
            "failures_path": str(self.failures_path),
            "tasks": [summary.to_dict() for summary in summaries],
        }

    def _safe_pubmed(self, spec: TDCTaskSpec, task_dir: Path) -> List[Dict[str, Any]]:
        query = pubmed_query_for_task(spec)
        try:
            pmids = self._pubmed_search(query, retmax=self.config.max_pubmed)
            records = self._pubmed_fetch(pmids, query=query, spec=spec)
        except Exception as exc:
            self._failure(spec.name, "pubmed", query, exc)
            records = []
        write_text(task_dir / "pubmed_literature.md", self._pubmed_markdown(spec, query, records))
        self._sleep()
        return records

    def _safe_pubchem(self, spec: TDCTaskSpec, task_dir: Path) -> List[Dict[str, Any]]:
        selected = select_representative_compounds(
            self.config.data_root / spec.name / f"{spec.name}.csv",
            spec=spec,
            max_compounds=self.config.max_compounds,
            random_seed=self.config.random_seed,
        )
        records = []
        for item in selected:
            smiles = str(item.get("smiles") or "").strip()
            if not smiles:
                continue
            try:
                record = self._pubchem_by_smiles(smiles)
                record.update({"tdc_label": item.get("label"), "tdc_drug_id": item.get("drug_id")})
                records.append(record)
            except Exception as exc:
                self._failure(spec.name, "pubchem", smiles, exc)
            self._sleep()
        write_text(task_dir / "pubchem_compounds.md", self._pubchem_markdown(spec, records))
        return records

    def _safe_dailymed(self, spec: TDCTaskSpec, task_dir: Path, compounds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        labels: List[Dict[str, Any]] = []
        seen_names = set()
        for compound in compounds:
            for name in candidate_drug_names(compound):
                key = name.lower()
                if key in seen_names:
                    continue
                seen_names.add(key)
                try:
                    labels.extend(self._dailymed_labels(name, max_results=self.config.max_dailymed_labels))
                except Exception as exc:
                    self._failure(spec.name, "dailymed", name, exc)
                self._sleep()
                if len(labels) >= self.config.max_compounds * self.config.max_dailymed_labels:
                    break
            if len(labels) >= self.config.max_compounds * self.config.max_dailymed_labels:
                break
        write_text(task_dir / "dailymed_labels.md", self._dailymed_markdown(spec, labels))
        return labels

    def _pubmed_search(self, query: str, *, retmax: int) -> List[str]:
        params = {"db": "pubmed", "retmode": "json", "retmax": retmax, "term": query}
        if self.config.ncbi_api_key:
            params["api_key"] = self.config.ncbi_api_key
        response = self.session.get(f"{PUBMED_EUTILS_BASE_URL}/esearch.fcgi", params=params, timeout=self.config.timeout)
        response.raise_for_status()
        return response.json().get("esearchresult", {}).get("idlist", [])

    def _pubmed_fetch(self, pmids: List[str], *, query: str, spec: TDCTaskSpec) -> List[Dict[str, Any]]:
        if not pmids:
            return []
        params = {"db": "pubmed", "retmode": "xml", "id": ",".join(pmids)}
        if self.config.ncbi_api_key:
            params["api_key"] = self.config.ncbi_api_key
        response = self.session.get(f"{PUBMED_EUTILS_BASE_URL}/efetch.fcgi", params=params, timeout=self.config.timeout)
        response.raise_for_status()
        return parse_pubmed_xml(response.text, query=query, spec=spec, retrieved_at=self.retrieved_at)

    def _pubchem_by_smiles(self, smiles: str) -> Dict[str, Any]:
        property_names = ",".join(
            [
                "CanonicalSMILES",
                "IUPACName",
                "MolecularFormula",
                "MolecularWeight",
                "XLogP",
                "TPSA",
                "HBondDonorCount",
                "HBondAcceptorCount",
            ]
        )
        response = self.session.get(
            f"{PUBCHEM_BASE_URL}/rest/pug/compound/smiles/{quote(smiles, safe='')}/property/{property_names}/JSON",
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        properties = response.json().get("PropertyTable", {}).get("Properties", [])
        if not properties:
            raise ValueError("PubChem returned no compound properties.")
        data = properties[0]
        cid = data.get("CID")
        synonyms = self._pubchem_synonyms(cid) if cid else []
        return {
            "source": "pubchem",
            "query_smiles": smiles,
            "cid": cid,
            "canonical_smiles": data.get("CanonicalSMILES"),
            "iupac_name": data.get("IUPACName"),
            "molecular_formula": data.get("MolecularFormula"),
            "molecular_weight": data.get("MolecularWeight"),
            "xlogp": data.get("XLogP"),
            "tpsa": data.get("TPSA"),
            "hbond_donor_count": data.get("HBondDonorCount"),
            "hbond_acceptor_count": data.get("HBondAcceptorCount"),
            "synonyms": synonyms[:20],
            "url": f"{PUBCHEM_BASE_URL}/compound/{cid}" if cid else None,
            "retrieved_at": self.retrieved_at,
            "source_note": "PubChem PUG REST compound properties and synonyms.",
        }

    def _pubchem_synonyms(self, cid: Any) -> List[str]:
        response = self.session.get(f"{PUBCHEM_BASE_URL}/rest/pug/compound/cid/{cid}/synonyms/JSON", timeout=self.config.timeout)
        response.raise_for_status()
        info = response.json().get("InformationList", {}).get("Information", [])
        if not info:
            return []
        return [str(item) for item in info[0].get("Synonym", []) if item]

    def _dailymed_labels(self, drug_name: str, *, max_results: int) -> List[Dict[str, Any]]:
        response = self.session.get(
            f"{DAILYMED_BASE_URL}/spls.json",
            params={"drug_name": drug_name, "page": 1, "pageSize": max_results},
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        rows = response.json().get("data") or response.json().get("results") or []
        labels = []
        for row in rows[:max_results] if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            setid = row.get("setid") or row.get("set_id")
            sections: List[Dict[str, str]] = []
            section_error = None
            if setid:
                try:
                    sections = self._dailymed_label_sections(setid)
                except Exception as exc:
                    section_error = str(exc)
            labels.append(
                {
                    "source": "dailymed",
                    "query_name": drug_name,
                    "setid": setid,
                    "title": row.get("title"),
                    "published_date": row.get("published_date"),
                    "url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}" if setid else None,
                    "summary_sections": sections,
                    "section_error": section_error,
                    "retrieved_at": self.retrieved_at,
                    "source_note": "DailyMed public SPL label metadata and selected label section summaries.",
                }
            )
        return labels

    def _dailymed_label_sections(self, setid: str) -> List[Dict[str, str]]:
        response = self.session.get(f"{DAILYMED_BASE_URL}/spls/{setid}.xml", timeout=self.config.timeout)
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
            if not any(keyword in normalized for keyword in DAILYMED_SUMMARY_SECTION_KEYWORDS):
                continue
            text = _section_text(section, title)
            if not text:
                continue
            seen.add(normalized)
            sections.append({"title": title, "text": text[:1200]})
            if len(sections) >= 6:
                break
        return sections

    def _pubmed_markdown(self, spec: TDCTaskSpec, query: str, records: List[Dict[str, Any]]) -> str:
        lines = [
            f"# PubMed Evidence for {spec.name}",
            "",
            f"- source: PubMed",
            f"- task: {spec.name}",
            f"- query: {query}",
            f"- url: https://pubmed.ncbi.nlm.nih.gov/?term={quote(query)}",
            f"- retrieved_at: {self.retrieved_at}",
            "- license_note: PubMed titles and abstracts are stored as bibliographic metadata/abstract snippets for retrieval; no full-text article body is stored.",
            "",
        ]
        for record in records:
            lines.extend(
                [
                    f"## PMID {record.get('pmid')}: {record.get('title') or 'Untitled'}",
                    "",
                    f"- source: PubMed",
                    f"- task: {spec.name}",
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

    def _pubchem_markdown(self, spec: TDCTaskSpec, records: List[Dict[str, Any]]) -> str:
        lines = [
            f"# PubChem Compound Evidence for {spec.name}",
            "",
            f"- source: PubChem",
            f"- task: {spec.name}",
            f"- retrieved_at: {self.retrieved_at}",
            "- license_note: PubChem public compound identifiers, names, synonyms, and computed properties are stored for retrieval.",
            "",
        ]
        for record in records:
            lines.extend(
                [
                    f"## CID {record.get('cid')}: {record.get('iupac_name') or record.get('canonical_smiles')}",
                    "",
                    f"- source: PubChem",
                    f"- task: {spec.name}",
                    f"- query: {record.get('query_smiles')}",
                    f"- url: {record.get('url')}",
                    f"- retrieved_at: {record.get('retrieved_at')}",
                    f"- canonical_smiles: `{record.get('canonical_smiles')}`",
                    f"- tdc_label: `{record.get('tdc_label')}`",
                    f"- properties: MW={record.get('molecular_weight')}, XLogP={record.get('xlogp')}, TPSA={record.get('tpsa')}, HBD={record.get('hbond_donor_count')}, HBA={record.get('hbond_acceptor_count')}",
                    f"- synonyms: {', '.join(record.get('synonyms', [])[:12])}",
                    "",
                ]
            )
        return "\n".join(lines).rstrip() + "\n"

    def _dailymed_markdown(self, spec: TDCTaskSpec, labels: List[Dict[str, Any]]) -> str:
        lines = [
            f"# DailyMed Label Evidence for {spec.name}",
            "",
            f"- source: DailyMed",
            f"- task: {spec.name}",
            f"- retrieved_at: {self.retrieved_at}",
            "- license_note: DailyMed public SPL label metadata and selected section excerpts are stored for retrieval.",
            "- source_note: Selected public label summary sections are stored when available.",
            "",
        ]
        for label in labels:
            lines.extend(
                [
                    f"## {label.get('query_name')}: {label.get('title') or label.get('setid')}",
                    "",
                    f"- source: DailyMed",
                    f"- task: {spec.name}",
                    f"- query: {label.get('query_name')}",
                    f"- setid: {label.get('setid')}",
                    f"- url: {label.get('url')}",
                    f"- published_date: {label.get('published_date') or ''}",
                    f"- retrieved_at: {label.get('retrieved_at')}",
                    "",
                ]
            )
            if label.get("summary_sections"):
                lines.append("### Selected label sections")
                lines.append("")
                for section in label.get("summary_sections", [])[:6]:
                    lines.extend(
                        [
                            f"#### {section.get('title')}",
                            "",
                            str(section.get("text") or "").strip(),
                            "",
                        ]
                    )
            elif label.get("section_error"):
                lines.extend([f"- section_error: {label.get('section_error')}", ""])
        if not labels:
            lines.append("No DailyMed labels were found for the sampled PubChem candidate names.")
        return "\n".join(lines).rstrip() + "\n"

    def _write_manifest(self, summaries: List[TaskCorpusSummary]) -> Path:
        return write_json(
            self.output_dir / "manifest.json",
            {
                "created_at": self.retrieved_at,
                "source_set": ["PubMed", "PubChem", "DailyMed"],
                "tasks": [summary.to_dict() for summary in summaries],
                "config": {
                    **asdict(self.config),
                    "output_dir": str(self.config.output_dir),
                    "data_root": str(self.config.data_root),
                },
            },
        )

    def _write_coverage(self, summaries: List[TaskCorpusSummary]) -> tuple[Path, Path]:
        rows = [summary.to_dict() for summary in summaries]
        df = pd.DataFrame(rows)
        csv_path = self.output_dir / "coverage_summary.csv"
        df.to_csv(csv_path, index=False)
        lines = [
            "# Real RAG Corpus Coverage Summary",
            "",
            "| task | mode | pubmed_records | pubchem_compounds | dailymed_labels | failures |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['task']} | {row['mode']} | {row['pubmed_records']} | {row['pubchem_compounds']} | {row['dailymed_labels']} | {row['failures']} |"
            )
        md_path = write_text(self.output_dir / "coverage_summary.md", "\n".join(lines).rstrip() + "\n")
        return csv_path, md_path

    def _failure(self, task: str, source: str, query: str, exc: Exception) -> None:
        ensure_dir(self.failures_path.parent)
        payload = {
            "retrieved_at": self.retrieved_at,
            "task": task,
            "source": source,
            "query": query,
            "error": str(exc),
        }
        with self.failures_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _count_task_failures(self, task: str) -> int:
        if not self.failures_path.exists():
            return 0
        count = 0
        for line in self.failures_path.read_text(encoding="utf-8").splitlines():
            try:
                count += 1 if json.loads(line).get("task") == task else 0
            except Exception:
                continue
        return count

    def _sleep(self) -> None:
        if self.config.request_delay > 0:
            time.sleep(self.config.request_delay)


def pubmed_query_for_task(spec: TDCTaskSpec) -> str:
    synonyms = TASK_SYNONYMS.get(spec.name, [])
    endpoint_terms = [spec.label_description, *synonyms]
    endpoint = " OR ".join(f'"{term}"' for term in endpoint_terms if term)
    context = "ADMET OR pharmacokinetics OR toxicity OR drug discovery OR molecular property"
    return f"({endpoint}) AND ({context})"


def select_representative_compounds(
    path: str | Path,
    *,
    spec: TDCTaskSpec,
    max_compounds: int,
    random_seed: int = 13,
) -> List[Dict[str, Any]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    if df.empty or "smiles" not in df.columns:
        return []
    if "label" not in df.columns:
        selected = df.head(max_compounds)
    elif spec.mode == "classification":
        selected_parts = []
        labels = list(df["label"].dropna().unique())
        per_label = max(1, max_compounds // max(1, len(labels)))
        for label in labels:
            group = df[df["label"] == label]
            selected_parts.append(group.sample(n=min(per_label, len(group)), random_state=random_seed) if len(group) else group)
        selected = pd.concat(selected_parts).drop_duplicates(subset=["smiles"]).head(max_compounds)
        if len(selected) < max_compounds:
            remainder = df[~df["smiles"].isin(selected["smiles"])].head(max_compounds - len(selected))
            selected = pd.concat([selected, remainder])
    else:
        work = df.copy()
        try:
            work["bin"] = pd.qcut(pd.to_numeric(work["label"]), q=min(max_compounds, max(1, len(work))), duplicates="drop")
            selected = work.groupby("bin", observed=False, group_keys=False).head(1).head(max_compounds)
        except Exception:
            selected = work.head(max_compounds)
    rows = []
    for _, row in selected.head(max_compounds).iterrows():
        rows.append(
            {
                "smiles": row.get("smiles"),
                "label": row.get("label"),
                "drug_id": row.get("drug_id"),
            }
        )
    return rows


def candidate_drug_names(compound: Mapping[str, Any], *, limit: int = 3) -> List[str]:
    candidates = []
    for value in compound.get("synonyms", []) or []:
        text = str(value).strip()
        if _looks_like_drug_name(text):
            candidates.append(text)
        if len(candidates) >= limit:
            break
    if not candidates and compound.get("iupac_name"):
        candidates.append(str(compound["iupac_name"])[:80])
    return candidates[:limit]


def parse_pubmed_xml(xml_text: str, *, query: str, spec: TDCTaskSpec, retrieved_at: str) -> List[Dict[str, Any]]:
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
        abstract = " ".join(abstract_parts)
        records.append(
            {
                "source": "pubmed",
                "task": spec.name,
                "query": query,
                "pmid": pmid,
                "title": _clean_xml_text(title),
                "journal": _clean_xml_text(journal),
                "year": _clean_xml_text(year),
                "abstract": _clean_xml_text(abstract),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                "retrieved_at": retrieved_at,
                "source_note": "PubMed title and abstract metadata.",
            }
        )
    return records


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


def _looks_like_drug_name(text: str) -> bool:
    if not (3 <= len(text) <= 60):
        return False
    if not re.search(r"[A-Za-z]", text):
        return False
    bad_tokens = ["acid", "chloride", "hydrate", "sodium", "compound", "inchi", "smiles"]
    lower = text.lower()
    if any(token in lower for token in bad_tokens):
        return False
    if sum(ch.isdigit() for ch in text) > 6:
        return False
    return True


def build_config_from_args(args: Any) -> RealRAGCorpusConfig:
    return RealRAGCorpusConfig(
        output_dir=Path(args.output_dir),
        data_root=Path(args.data_root),
        tasks=expand_task_names(args.tasks),
        max_pubmed=int(args.max_pubmed),
        max_compounds=int(args.max_compounds),
        max_dailymed_labels=int(args.max_dailymed_labels),
        timeout=float(args.timeout),
        request_delay=float(args.request_delay),
        random_seed=int(args.random_seed),
        ncbi_api_key=os.getenv("NCBI_API_KEY"),
    )
