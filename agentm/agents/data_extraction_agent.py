from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentm.tools.biomedical_clients import (
    BiomedicalQueryBuilder,
    BiomedicalQueryContext,
    ChEMBLClient,
    SemanticScholarClient,
    UniProtClient,
)
from agentm.utils.io import ensure_dir, write_json, write_text


@dataclass
class DataExtractionResult:
    query_context: BiomedicalQueryContext
    uniprot: Dict[str, Any] = field(default_factory=dict)
    chembl: Dict[str, Any] = field(default_factory=dict)
    semantic_scholar: List[Dict[str, Any]] = field(default_factory=list)
    literature_records: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["query_context"] = self.query_context.to_dict()
        return data


class DataExtractionAgent:
    """AgentD-style biomedical retrieval for target/disease workflows."""

    def __init__(
        self,
        output_dir: str | Path = "outputs/data_extraction",
        *,
        uniprot_client: Optional[UniProtClient] = None,
        chembl_client: Optional[ChEMBLClient] = None,
        semantic_scholar_client: Optional[SemanticScholarClient] = None,
        query_builder: Optional[BiomedicalQueryBuilder] = None,
    ):
        self.output_dir = ensure_dir(output_dir)
        self.query_builder = query_builder or BiomedicalQueryBuilder()
        self.uniprot_client = uniprot_client or UniProtClient(timeout=self._timeout())
        self.chembl_client = chembl_client or ChEMBLClient(timeout=self._timeout())
        self.semantic_scholar_client = semantic_scholar_client or SemanticScholarClient(
            timeout=self._timeout(),
            api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
        )

    def run(
        self,
        request: str,
        *,
        target: Optional[str] = None,
        disease: Optional[str] = None,
        max_papers: int = 5,
        max_chembl_molecules: int = 10,
        download_pdfs: bool = False,
    ) -> DataExtractionResult:
        context = self.query_builder.build(request, target=target, disease=disease)
        result = DataExtractionResult(query_context=context)
        result.uniprot = self._safe_uniprot(context, result)
        result.chembl = self._safe_chembl(context, max_chembl_molecules, result)
        result.semantic_scholar = self._safe_semantic_scholar(context, max_papers, result)
        result.literature_records = self.semantic_scholar_client.write_literature_records(
            result.semantic_scholar,
            self.output_dir / "literature",
            download_pdfs=download_pdfs,
        )
        artifacts = self._write_artifacts(result)
        result.artifacts.update(artifacts)
        return result

    def _safe_uniprot(self, context: BiomedicalQueryContext, result: DataExtractionResult) -> Dict[str, Any]:
        try:
            payload = self.uniprot_client.search_fasta(context.target, organism_id=context.organism_id)
            fasta = payload.get("fasta")
            if fasta:
                fasta_path = self.output_dir / "target.fasta"
                write_text(fasta_path, fasta + "\n")
                payload["fasta_path"] = str(fasta_path)
            return payload
        except Exception as exc:
            message = f"UniProt retrieval failed: {exc}"
            result.warnings.append(message)
            return {"status": "failed", "error": str(exc), "target": context.target}

    def _safe_chembl(self, context: BiomedicalQueryContext, max_molecules: int, result: DataExtractionResult) -> Dict[str, Any]:
        try:
            target_payload = self.chembl_client.search_target(context.target)
            targets = target_payload.get("targets") or []
            first_target = targets[0] if targets else {}
            target_chembl_id = first_target.get("target_chembl_id")
            molecules_payload = self.chembl_client.known_molecules_for_target(target_chembl_id, limit=max_molecules)
            molecules = []
            for item in molecules_payload.get("molecules", [])[:max_molecules]:
                molecule_payload = self.chembl_client.molecule_smiles(item.get("molecule_chembl_id"))
                molecules.append({**item, **molecule_payload})
            return {
                "status": "success",
                "target_search": target_payload,
                "selected_target_chembl_id": target_chembl_id,
                "molecules": molecules,
            }
        except Exception as exc:
            message = f"ChEMBL retrieval failed: {exc}"
            result.warnings.append(message)
            return {"status": "failed", "error": str(exc), "target": context.target, "molecules": []}

    def _safe_semantic_scholar(self, context: BiomedicalQueryContext, max_papers: int, result: DataExtractionResult) -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []
        for query in context.literature_queries:
            try:
                payloads.append(self.semantic_scholar_client.search_papers(query, limit=max_papers))
            except Exception as exc:
                result.warnings.append(f"Semantic Scholar retrieval failed for '{query}': {exc}")
                payloads.append({"status": "failed", "query": query, "error": str(exc), "papers": []})
        return payloads

    def _write_artifacts(self, result: DataExtractionResult) -> Dict[str, Any]:
        summary_path = write_json(self.output_dir / "data_extraction_summary.json", result.to_dict())
        smiles_rows = []
        for molecule in result.chembl.get("molecules", []) if isinstance(result.chembl, dict) else []:
            if molecule.get("canonical_smiles"):
                smiles_rows.append(
                    {
                        "molecule_chembl_id": molecule.get("molecule_chembl_id"),
                        "pref_name": molecule.get("pref_name"),
                        "canonical_smiles": molecule.get("canonical_smiles"),
                        "pchembl_value": molecule.get("pchembl_value"),
                    }
                )
        smiles_path = write_json(self.output_dir / "chembl_molecules.json", {"molecules": smiles_rows})
        return {
            "data_extraction_summary_path": str(summary_path),
            "chembl_molecules_path": str(smiles_path),
            "literature_dir": str(self.output_dir / "literature"),
            "fasta_path": result.uniprot.get("fasta_path") if isinstance(result.uniprot, dict) else None,
        }

    def _timeout(self) -> float:
        return float(os.getenv("AGENTM_BIOMEDICAL_TIMEOUT", "20"))
