from __future__ import annotations

from agentm.agents.drug_resolver_agent import DrugResolverAgent
from agentm.agents.input_agent import InputAgent
from agentm.schema import MoleculeRecord
from agentm.tools.smiles import standardize_smiles


class MoleculeProcessingAgent:
    def __init__(self, resolver_agent: DrugResolverAgent | None = None):
        self.resolver_agent = resolver_agent or DrugResolverAgent()

    def process_input(self, value: str, input_agent: InputAgent | None = None) -> MoleculeRecord:
        resolution = self.resolver_agent.resolve(value)
        record = MoleculeRecord(
            original_input=value,
            smiles=resolution.smiles,
            canonical_smiles=resolution.canonical_smiles,
            name=resolution.canonical_name,
            source=resolution.source,
            is_valid=resolution.is_valid,
            errors=list(resolution.errors),
            metadata={
                "resolver": "DrugResolverAgent",
                "resolution_trace": resolution.trace,
                "resolution_warnings": resolution.warnings,
                **resolution.metadata,
            },
        )
        if not resolution.is_valid:
            record.errors.append(f"Could not resolve input to SMILES: {resolution.source}")
        return record

    def process_smiles(self, smiles: str) -> MoleculeRecord:
        canonical, err = standardize_smiles(smiles)
        record = MoleculeRecord(original_input=smiles, smiles=smiles, canonical_smiles=canonical, source="smiles")
        if err:
            record.errors.append(err)
        else:
            record.is_valid = True
        return record
