from __future__ import annotations

from typing import Dict, Iterable, List

from rdkit import Chem, rdBase

from agentm.rules.compiler import OPS
from agentm.schema import Rule, RuleFeatureVector


class RuleExecutor:
    def execute(
        self,
        mol: Chem.Mol,
        descriptors: Dict[str, float],
        rules: Iterable[Rule],
    ) -> RuleFeatureVector:
        vector = RuleFeatureVector()
        for rule in rules:
            if not rule.is_executable:
                vector.failed_rules[rule.rule_id] = rule.error or "Rule marked non-executable"
                continue
            try:
                value = self._execute_one(mol, descriptors, rule)
                vector.values[rule.feature_name] = float(value)
                vector.executed_rules.append(rule.rule_id)
            except Exception as exc:
                vector.failed_rules[rule.rule_id] = str(exc)
        return vector

    def filter_executable_unique(self, rules: Iterable[Rule]) -> List[Rule]:
        seen = set()
        unique: List[Rule] = []
        for rule in rules:
            if not rule.is_executable:
                continue
            signature = rule.signature()
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(rule)
        return unique

    def _execute_one(self, mol: Chem.Mol, descriptors: Dict[str, float], rule: Rule) -> float:
        metadata = rule.metadata
        operation = metadata.get("operation")
        if operation == "descriptor_threshold":
            descriptor = metadata["descriptor"]
            op = metadata["operator"]
            threshold = float(metadata["threshold"])
            if op not in OPS:
                raise ValueError(f"Unsupported operator: {op}")
            return 1.0 if OPS[op](float(descriptors.get(descriptor, 0.0)), threshold) else 0.0
        if operation == "substructure":
            with rdBase.BlockLogs():
                pattern = Chem.MolFromSmarts(metadata["smarts"])
            if pattern is None:
                raise ValueError(f"Invalid SMARTS: {metadata['smarts']}")
            return 1.0 if mol.HasSubstructMatch(pattern) else 0.0
        raise ValueError(f"Unsupported rule operation: {operation}")
