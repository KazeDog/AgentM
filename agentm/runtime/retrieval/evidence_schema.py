from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from experiments.agentm_reliability.evidence_text import source_body_text, source_title_text


ENDPOINT_ALIASES: Dict[str, List[str]] = {
    "bbbp": ["bbbp", "bbb", "blood-brain barrier", "blood brain barrier", "brain penetration", "cns penetration", "bbb permeability"],
    "dili": ["dili", "drug-induced liver injury", "liver injury", "hepatic injury", "hepatotoxicity", "hepatic toxicity", "liver damage"],
    "herg": ["herg", "kcnh2", "qt prolongation", "cardiac arrhythmia", "cardiotoxicity"],
    "ames": ["ames", "mutagenicity", "mutagenic", "genotoxicity"],
    "hia": ["hia", "human intestinal absorption", "intestinal absorption", "oral absorption"],
    "pgp": ["pgp", "p-glycoprotein", "p glycoprotein", "abcb1", "efflux transporter"],
    "bioavailability": ["bioavailability", "oral bioavailability", "systemic exposure"],
    "caco2": ["caco2", "caco-2", "intestinal permeability", "permeability"],
    "solubility": ["solubility", "aqueous solubility", "water solubility"],
    "lipophilicity": ["lipophilicity", "logp", "lipophilic", "partition coefficient"],
    "hydration_free_energy": ["hydration free energy", "free solvation energy", "solvation free energy", "freesolv"],
    "half_life": ["half-life", "half life", "elimination half-life", "pharmacokinetic half-life"],
    "clearance_hepatocyte": ["hepatocyte clearance", "hepatic clearance", "intrinsic clearance", "hepatocyte stability"],
    "clearance_microsome": ["microsome clearance", "microsomal clearance", "microsomal stability", "intrinsic clearance"],
    "clintox": ["clintox", "clinical toxicity", "toxicity", "clinical trial toxicity"],
    "ld50": ["ld50", "acute toxicity", "median lethal dose", "lethal dose"],
    "cyp3a4": ["cyp3a4", "cyp 3a4", "cyp3a4 inhibition", "cyp3a4 inhibitor", "cyp3a4 substrate", "cytochrome p450 3a4", "cytochrome p450 3a4 inhibition"],
    "cyp2d6": ["cyp2d6", "cyp 2d6", "cyp2d6 inhibition", "cyp2d6 inhibitor", "cyp2d6 substrate", "cytochrome p450 2d6", "cytochrome p450 2d6 inhibition"],
    "cyp2c9": ["cyp2c9", "cyp 2c9", "cyp2c9 inhibition", "cyp2c9 inhibitor", "cyp2c9 substrate", "cytochrome p450 2c9", "cytochrome p450 2c9 inhibition"],
    "cyp2c19": ["cyp2c19", "cyp 2c19", "cyp2c19 inhibition", "cyp2c19 inhibitor", "cyp2c19 substrate", "cytochrome p450 2c19", "cytochrome p450 2c19 inhibition"],
    "cyp1a2": ["cyp1a2", "cyp 1a2", "cyp1a2 inhibition", "cyp1a2 inhibitor", "cyp1a2 substrate", "cytochrome p450 1a2", "cytochrome p450 1a2 inhibition"],
    "ddi": [
        "ddi",
        "drug-drug interaction",
        "drug interaction",
        "coadministration",
        "concomitant use",
        "bleeding",
        "anticoagulant",
        "antiplatelet",
        "dual therapy",
        "triple therapy",
        "combined therapy",
    ],
}


DATASET_ENDPOINT_HINTS: Dict[str, str] = {
    "bbb_martins": "bbbp",
    "bbbp": "bbbp",
    "dili": "dili",
    "herg": "herg",
    "ames": "ames",
    "hia_hou": "hia",
    "pgp_broccatelli": "pgp",
    "bioavailability_ma": "bioavailability",
    "caco2_wang": "caco2",
    "solubility_aqsoldb": "solubility",
    "lipophilicity_astrazeneca": "lipophilicity",
    "hydrationfreeenergy_freesolv": "hydration_free_energy",
    "hydration_free_energy_freesolv": "hydration_free_energy",
    "half_life_obach": "half_life",
    "halflife_obach": "half_life",
    "clearance_hepatocyte_az": "clearance_hepatocyte",
    "clearance_microsome_az": "clearance_microsome",
    "clintox": "clintox",
    "ld50_zhu": "ld50",
    "bioavailability": "bioavailability",
    "caco2": "caco2",
    "solubility": "solubility",
    "lipophilicity": "lipophilicity",
    "half_life": "half_life",
    "clearance_hepatocyte": "clearance_hepatocyte",
    "clearance_microsome": "clearance_microsome",
    "ld50": "ld50",
    "pgp": "pgp",
    "hia": "hia",
    "drugbank": "ddi",
    "twosides": "ddi",
    "ddi": "ddi",
}


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_term(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def compact_term(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def unique_strings(values: Iterable[Any], *, lower: bool = False, limit: Optional[int] = None) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalize_text(value)
        if not text:
            continue
        text = _strip_code_ticks(text)
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text.lower() if lower else text)
        if limit and len(out) >= limit:
            break
    return out


def endpoint_from_dataset(dataset_name: Any, *, task_type: Any = None) -> str:
    if str(task_type or "").lower() == "ddi_prediction":
        return "ddi"
    value = str(dataset_name or "").strip()
    compact = compact_term(value)
    for needle, endpoint in DATASET_ENDPOINT_HINTS.items():
        if compact_term(needle) in compact:
            return endpoint
    for endpoint in ENDPOINT_ALIASES:
        if endpoint in compact:
            return endpoint
    return normalize_term(value).split(" ")[0] if value else ""


def endpoint_terms(endpoint: Any) -> List[str]:
    key = compact_term(endpoint)
    aliases = ENDPOINT_ALIASES.get(key, [])
    return unique_strings([endpoint, *aliases], lower=True)


def infer_endpoint_from_context(context: Mapping[str, Any]) -> str:
    task_type = context.get("task_type")
    if task_type == "ddi_prediction":
        return "ddi"
    return endpoint_from_dataset(context.get("dataset_name"), task_type=task_type)


def entity_terms_from_context(context: Mapping[str, Any]) -> List[str]:
    values: List[Any] = []
    for molecule in context.get("molecules", []) or []:
        if isinstance(molecule, Mapping):
            values.extend(molecule.get(key) for key in ["name", "original_input", "canonical_smiles", "smiles"])
            values.extend(molecule.get("synonyms", []) or [])
            metadata = molecule.get("metadata") if isinstance(molecule.get("metadata"), Mapping) else {}
        else:
            values.extend(getattr(molecule, key, None) for key in ["name", "original_input", "canonical_smiles", "smiles"])
            values.extend(getattr(molecule, "synonyms", []) or [])
            metadata = getattr(molecule, "metadata", {}) or {}
        if isinstance(metadata, Mapping):
            values.extend(
                metadata.get(key)
                for key in [
                    "pubchem_cid",
                    "pubchem_title",
                    "pubchem_iupac_name",
                    "chembl_id",
                    "chembl_pref_name",
                ]
            )
            values.extend(metadata.get("synonyms", []) or [])
            values.extend(metadata.get("pubchem_synonyms", []) or [])
    values.extend(context.get("molecule_mentions", []) or [])
    if context.get("ddi_pair") is not None:
        pair = context.get("ddi_pair")
        if isinstance(pair, Mapping):
            values.extend(pair.get(key) for key in ["drug_a", "drug_b", "drug_a_name", "drug_b_name"])
        else:
            values.extend(getattr(pair, key, None) for key in ["drug_a", "drug_b", "drug_a_name", "drug_b_name"])
    return unique_strings(values, lower=True, limit=80)


def iter_structured_evidence_documents(corpus_dirs: Iterable[Path], *, limit: int = 0) -> Iterator[Dict[str, Any]]:
    emitted = 0
    for corpus_dir in corpus_dirs:
        root = Path(corpus_dir)
        for path in sorted(root.rglob("*")) if root.exists() else []:
            if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".txt", ".md"}:
                continue
            if path.name.lower() in {
                "coverage_summary.md",
                "failures.jsonl",
                "manifest.json",
                "pair_manifest.json",
                "backfill_manifest.json",
                "backfill_failures.jsonl",
            }:
                continue
            if path.name.lower().startswith("benchmark_evidence_backfill_summary"):
                continue
            if path.name.lower().startswith("structured_evidence"):
                continue
            if "_audit" in path.name.lower() or path.name.lower().startswith("backfill_queue_audit"):
                continue
            for doc in load_evidence_path(path, corpus_dir=root):
                yield doc
                emitted += 1
                if limit and emitted >= limit:
                    return


def load_evidence_path(path: Path, *, corpus_dir: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    rel = path.relative_to(corpus_dir)
    task_type = "ddi_prediction" if "ddi" in str(corpus_dir).lower() or "pairs" in rel.parts else "admet_prediction"
    dataset_name = _dataset_from_path(rel, path)
    endpoint = endpoint_from_dataset(dataset_name, task_type=task_type)
    manifest = _pair_manifest(path.parent) if task_type == "ddi_prediction" else {}
    base = {
        "source_path": str(path),
        "doc_id": str(rel),
        "source": _source_from_path(path),
        "task_type": task_type,
        "dataset_name": dataset_name,
        "endpoint": endpoint,
        "endpoint_aliases": endpoint_terms(endpoint),
        "evidence_type": _evidence_type_from_path(path),
        "ddi_pair_key": manifest.get("pair_key"),
        "pair_drug_ids": _pair_drug_ids(manifest),
        "pair_drug_names": unique_strings([manifest.get("evidence_drug_a"), manifest.get("evidence_drug_b")]),
        "pair_drug_smiles": unique_strings([manifest.get("drug_a"), manifest.get("drug_b")]),
    }
    if path.suffix.lower() == ".jsonl":
        docs = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            if line.strip():
                docs.append(_doc_from_payload(_json_or_text(line), base=base, chunk_suffix=str(line_no)))
        return docs
    if path.suffix.lower() == ".json":
        return [_doc_from_payload(_json_or_text(path.read_text(encoding="utf-8", errors="ignore")), base=base, chunk_suffix="1")]
    return _docs_from_markdown(path.read_text(encoding="utf-8", errors="ignore"), base=base)


def _docs_from_markdown(text: str, *, base: Mapping[str, Any]) -> List[Dict[str, Any]]:
    sections = _split_markdown_sections(text)
    if not sections:
        return [_doc_from_payload(text, base=base, chunk_suffix="1")]
    preamble_metadata = _metadata_lines(text[: text.find("##")]) if "##" in text else {}
    docs = []
    for idx, (heading, body) in enumerate(sections, start=1):
        payload = {"title": heading, "text": body, **preamble_metadata, **_metadata_lines(body)}
        payload["source_body"] = source_body_text(payload)
        docs.append(_doc_from_payload(payload, base=base, chunk_suffix=str(idx), heading=heading))
    return docs


def _doc_from_payload(payload: Any, *, base: Mapping[str, Any], chunk_suffix: str, heading: str = "") -> Dict[str, Any]:
    doc = dict(base)
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if isinstance(value, (str, int, float, bool, list)) and key not in {"embedding"}:
                doc[key] = value
        text = payload.get("text") or payload.get("snippet") or payload.get("abstract") or payload.get("claim") or json.dumps(payload, ensure_ascii=False)
    else:
        text = str(payload)
    title = normalize_text(doc.get("title") or heading or doc.get("doc_id"))
    doc["title"] = title
    doc["source"] = str(doc.get("source") or "").lower()
    doc["text"] = normalize_text(text)
    if "source_body" in doc:
        doc["source_body"] = normalize_text(doc.get("source_body"))
    doc["url"] = _strip_code_ticks(doc.get("url") or "")
    doc["endpoint"] = compact_term(doc.get("endpoint") or base.get("endpoint") or "")
    if base.get("ddi_pair_key"):
        doc["ddi_pair_key"] = base.get("ddi_pair_key")
        doc["pair_drug_ids"] = list(base.get("pair_drug_ids") or [])
        doc["pair_drug_names"] = list(base.get("pair_drug_names") or [])
        doc["pair_drug_smiles"] = list(base.get("pair_drug_smiles") or [])
    if doc.get("pmid") is None:
        pmid = _pmid_from_heading(title) or _pmid_from_text(doc["text"])
        if pmid:
            doc["pmid"] = pmid
    if doc.get("year") is None:
        year = _year_from_text(doc["text"])
        if year:
            doc["year"] = year
    compound = _compound_metadata(doc, heading=title)
    metadata_entity_terms = _split_metadata_terms(doc.get("entity_terms"))
    doc.update(compound)
    all_entities = [
        *compound.get("compound_names", []),
        *compound.get("compound_synonyms", []),
        *compound.get("compound_cids", []),
        *compound.get("chembl_ids", []),
        *compound.get("canonical_smiles", []),
        *metadata_entity_terms,
        *doc.get("pair_drug_ids", []),
        *doc.get("pair_drug_names", []),
        *doc.get("pair_drug_smiles", []),
    ]
    doc["drug_names"] = unique_strings([*compound.get("compound_names", []), *doc.get("pair_drug_names", [])])
    doc["entity_terms"] = unique_strings(all_entities, lower=True, limit=120)
    doc["text_entity_terms"] = _text_entity_terms(doc, all_entities)
    strict_endpoint_terms = endpoint_terms(doc.get("endpoint"))
    doc["endpoint_terms"] = strict_endpoint_terms
    doc["endpoint_aliases"] = strict_endpoint_terms
    text_compound_specific = _text_mentions_entity(doc)
    text_endpoint_specific = _text_mentions_endpoint(doc)
    doc["compound_specific"] = bool(text_compound_specific)
    doc["endpoint_specific"] = bool(text_endpoint_specific)
    doc["evidence_scope"] = _evidence_scope(doc)
    doc["chunk_id"] = f"{doc['doc_id']}::{chunk_suffix}"
    doc["evidence_id"] = doc["chunk_id"]
    return doc


def _compound_metadata(doc: Mapping[str, Any], *, heading: str) -> Dict[str, List[str]]:
    source = str(doc.get("source") or "").lower()
    names: List[Any] = [*(_as_list(doc.get("compound_names"))), *(_as_list(doc.get("drug_names")))]
    synonyms: List[Any] = _as_list(doc.get("compound_synonyms"))
    cids: List[Any] = _as_list(doc.get("compound_cids"))
    chembl_ids: List[Any] = _as_list(doc.get("chembl_ids"))
    smiles: List[Any] = _as_list(doc.get("canonical_smiles"))
    for term in _split_metadata_terms(doc.get("entity_terms")):
        term_text = str(term).strip()
        term_lower = term_text.lower()
        if not term_text:
            continue
        if re.fullmatch(r"(?:cid\s*)?\d+", term_lower):
            cids.append(re.sub(r"^cid\s*", "", term_lower))
        elif re.fullmatch(r"chembl\d+", term_lower):
            chembl_ids.append(term_text.upper())
        elif _looks_like_smiles(term_text):
            smiles.append(term_text)
        elif not term_lower.startswith("cid "):
            names.append(term_text)
    if source == "pubchem":
        cid_match = re.search(r"\bCID\s+(\d+)\b", heading, flags=re.I)
        if cid_match:
            cids.append(cid_match.group(1))
            after = heading[cid_match.end() :].lstrip(": -")
            if after:
                names.append(after)
        role_match = re.search(r"\bdrug_[ab]\s*:\s*CID\s+(\d+)", heading, flags=re.I)
        if role_match:
            cids.append(role_match.group(1))
        synonyms.extend(_split_list(doc.get("synonyms")))
        query = _strip_code_ticks(doc.get("query") or "")
        if query and not _looks_like_smiles(query) and not query.lower().startswith("db"):
            names.append(query)
    elif source == "dailymed":
        before = heading.split(":", 1)[0].strip()
        if before and not before.lower().startswith("dailymed") and "/" not in before and not before.lower().endswith(".md"):
            names.append(before)
        query = doc.get("query")
        if query and not _looks_like_smiles(query):
            names.append(query)
    elif source in {"chembl", "pubmed"}:
        chembl_ids.extend(re.findall(r"\bCHEMBL\d+\b", f"{heading} {doc.get('text', '')}", flags=re.I))
    for key in ["cid", "CID", "pubchem_cid"]:
        if doc.get(key):
            cids.append(doc.get(key))
    for key in ["chembl_id", "molecule_chembl_id"]:
        if doc.get(key):
            chembl_ids.append(doc.get(key))
    return {
        "compound_names": unique_strings(names, limit=20),
        "compound_synonyms": unique_strings(synonyms, limit=80),
        "compound_cids": unique_strings(cids, lower=True),
        "chembl_ids": unique_strings(chembl_ids, lower=False),
        "canonical_smiles": unique_strings([item for item in smiles if item and str(item).lower() != "none"], limit=10),
    }


def _split_markdown_sections(text: str) -> List[Tuple[str, str]]:
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", text, flags=re.M))
    sections: List[Tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections.append((normalize_text(match.group(1)), text[start:end].strip()))
    return sections


def _metadata_lines(text: str) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for line in text.splitlines():
        match = re.match(r"^-\s*([A-Za-z0-9_ -]+):\s*(.+?)\s*$", line)
        if not match:
            continue
        key = match.group(1).strip().lower().replace(" ", "_").replace("-", "_")
        metadata[key] = _strip_code_ticks(match.group(2).strip())
    return metadata


def _dataset_from_path(rel: Path, path: Path) -> str:
    for parts in [rel.parts, path.parts]:
        if "tasks" in parts:
            idx = parts.index("tasks")
            if idx + 1 < len(parts):
                return str(parts[idx + 1])
    return "DDI" if "pairs" in rel.parts else ""


def _pair_manifest(pair_dir: Path) -> Dict[str, Any]:
    manifest = pair_dir / "pair_manifest.json"
    if not manifest.exists():
        return {}
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _pair_drug_ids(manifest: Mapping[str, Any]) -> List[str]:
    ids: List[Any] = []
    pair_key = str(manifest.get("pair_key") or "")
    if pair_key:
        ids.extend(part for part in re.split(r"[_+|,;/\s]+", pair_key) if part)
    for key in ["drug_a_id", "drug_b_id", "drug_a_db_id", "drug_b_db_id"]:
        ids.append(manifest.get(key))
    return unique_strings(ids, lower=True)


def _source_from_path(path: Path) -> str:
    name = path.name.lower()
    if "pubchem" in name:
        return "pubchem"
    if "dailymed" in name:
        return "dailymed"
    if "pubmed" in name:
        return "pubmed"
    if "chembl" in name:
        return "chembl"
    return "local_rag"


def _evidence_type_from_path(path: Path) -> str:
    name = path.name.lower()
    if "pubchem" in name:
        return "compound_identity"
    if "dailymed" in name:
        return "drug_label"
    if "pubmed" in name:
        return "literature"
    if "interaction" in name:
        return "interaction"
    return "literature"


def _evidence_scope(doc: Mapping[str, Any]) -> str:
    if (
        doc.get("task_type") == "ddi_prediction"
        and (doc.get("pair_drug_names") or doc.get("pair_drug_ids"))
        and doc.get("compound_specific")
        and doc.get("endpoint_specific")
    ):
        return "ddi_pair"
    if doc.get("compound_specific") and doc.get("endpoint_specific"):
        return "compound_endpoint"
    if doc.get("compound_specific"):
        return "compound_background"
    if doc.get("endpoint_specific"):
        return "endpoint_background"
    return "background"


def _text_mentions_endpoint(doc: Mapping[str, Any]) -> bool:
    if str(doc.get("evidence_type") or "").lower() == "compound_identity":
        return False
    terms = endpoint_terms(doc.get("endpoint"))
    if compact_term(doc.get("endpoint")) == "ddi" or doc.get("task_type") == "ddi_prediction":
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
    if not terms:
        return False
    text = normalize_term(" ".join([source_title_text(doc), source_body_text(doc)]))
    compact = compact_term(text)
    for term in terms:
        norm = normalize_term(term)
        if norm and norm in text:
            return True
        if len(norm) > 4 and compact_term(norm) in compact:
            return True
    return False


def _text_mentions_entity(doc: Mapping[str, Any]) -> bool:
    text = " ".join(str(doc.get(key) or "") for key in ["title", "text", "snippet", "abstract"])
    if not text.strip():
        return False
    pair_names = unique_strings(doc.get("pair_drug_names", []) or [], lower=True)
    if doc.get("task_type") == "ddi_prediction" and len(pair_names) >= 2:
        return all(_mentions_text(text, [name]) for name in pair_names[:2])
    terms: List[Any] = []
    for key in [
        "compound_names",
        "drug_names",
        "compound_synonyms",
        "compound_cids",
        "chembl_ids",
        "canonical_smiles",
        "pair_drug_ids",
        "pair_drug_names",
        "pair_drug_smiles",
    ]:
        terms.extend(_as_list(doc.get(key)))
    return _mentions_text(text, terms)


def _text_entity_terms(doc: Mapping[str, Any], terms: Iterable[Any]) -> List[str]:
    text = " ".join(str(doc.get(key) or "") for key in ["title", "text", "snippet", "abstract"])
    if not text.strip():
        return []
    grounded = []
    for term in terms:
        if _mentions_text(text, [term]):
            grounded.append(term)
    return unique_strings(grounded, lower=True, limit=120)


def _mentions_text(text: str, terms: Iterable[Any]) -> bool:
    normalized = normalize_term(text)
    compact = compact_term(text)
    for term in terms:
        raw = normalize_term(term)
        raw_compact = compact_term(raw)
        if len(raw) >= 4 and raw in normalized:
            return True
        if len(raw_compact) >= 6 and raw_compact in compact:
            return True
    return False


def _json_or_text(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return text


def _split_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _split_metadata_terms(value: Any) -> List[str]:
    if isinstance(value, list):
        raw = [str(item) for item in value]
    else:
        raw = re.split(r"[;,\n]+", str(value or ""))
    return [part.strip() for part in raw if part and part.strip()]


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def _strip_code_ticks(value: Any) -> str:
    text = normalize_text(value)
    if len(text) >= 2 and text[0] == "`" and text[-1] == "`":
        return text[1:-1].strip()
    return text


def _looks_like_smiles(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if " " in text:
        return False
    return any(char in text for char in "=#@[]()/\\") or bool(re.search(r"[cnops]\d", text, flags=re.I))


def _pmid_from_heading(text: str) -> Optional[str]:
    match = re.search(r"\bPMID\s+(\d+)\b", text, flags=re.I)
    return match.group(1) if match else None


def _pmid_from_text(text: str) -> Optional[str]:
    match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", text)
    return match.group(1) if match else None


def _year_from_text(text: str) -> Optional[str]:
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return match.group(0) if match else None
