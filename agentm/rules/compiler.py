from __future__ import annotations

import hashlib
import operator
import re
from typing import Any, Dict, Optional

from agentm.schema import Rule


OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
}


class RuleCompiler:
    """Compile natural-language chemistry rules into a safe RDKit-rule DSL.

    The generated `Rule.code` is a readable RDKit Python function, but execution
    uses the restricted metadata representation instead of arbitrary exec/eval.
    """

    DESCRIPTOR_ALIASES = {
        "molecular weight": "MolWt",
        "molwt": "MolWt",
        "mw": "MolWt",
        "logp": "MolLogP",
        "tpsa": "TPSA",
        "polar surface area": "TPSA",
        "hbd": "NumHDonors",
        "hydrogen bond donor": "NumHDonors",
        "hba": "NumHAcceptors",
        "hydrogen bond acceptor": "NumHAcceptors",
        "rotatable": "NumRotatableBonds",
        "ring": "RingCount",
        "aromatic ring": "NumAromaticRings",
        "qed": "QED",
    }

    SMARTS_PATTERNS = {
        "carboxylic acid": "C(=O)[OH]",
        "amide": "C(=O)N",
        "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
        "phenol": "c[OH]",
        "benzene": "c1ccccc1",
        "halogen": "[F,Cl,Br,I]",
        "nitro": "[NX3](=O)=O",
        "sulfonamide": "S(=O)(=O)N",
        "ester": "C(=O)O[#6]",
    }

    def compile_text(self, text: str, task_type: str, source: str = "literature") -> Rule:
        lower = text.lower()
        descriptor_rule = self._compile_descriptor_rule(lower)
        if descriptor_rule:
            metadata = descriptor_rule
            feature = self._feature_name(metadata)
            return self._build_rule(text, task_type, source, feature, metadata)

        smarts_rule = self._compile_smarts_rule(lower)
        if smarts_rule:
            feature = self._feature_name(smarts_rule)
            return self._build_rule(text, task_type, source, feature, smarts_rule)

        rule_id = self._rule_id(text)
        return Rule(
            rule_id=rule_id,
            task_type=task_type,
            source=source,
            text=text,
            feature_name=f"uncompiled_{rule_id}",
            is_executable=False,
            error="No safe descriptor or SMARTS pattern recognized.",
        )

    def from_spec(self, spec: Dict[str, Any], task_type: str, source: str = "data_inferred") -> Rule:
        metadata = dict(spec)
        feature = self._feature_name(metadata)
        text = metadata.get("text") or f"{metadata.get('descriptor')} {metadata.get('operator')} {metadata.get('threshold')}"
        return self._build_rule(text, task_type, source, feature, metadata)

    def _compile_descriptor_rule(self, lower: str) -> Optional[Dict[str, Any]]:
        descriptor = None
        for alias, desc_name in self.DESCRIPTOR_ALIASES.items():
            if alias in lower:
                descriptor = desc_name
                break
        if not descriptor:
            return None

        threshold_match = re.search(r"(>=|<=|>|<|above|over|below|under|less than|greater than)\s*([-+]?\d*\.?\d+)", lower)
        if not threshold_match:
            return None
        raw_op, threshold = threshold_match.groups()
        op_map = {
            "above": ">",
            "over": ">",
            "greater than": ">",
            "below": "<",
            "under": "<",
            "less than": "<",
        }
        op = op_map.get(raw_op, raw_op)
        return {
            "operation": "descriptor_threshold",
            "descriptor": descriptor,
            "operator": op,
            "threshold": float(threshold),
        }

    def _compile_smarts_rule(self, lower: str) -> Optional[Dict[str, Any]]:
        for label, smarts in self.SMARTS_PATTERNS.items():
            if label in lower:
                return {
                    "operation": "substructure",
                    "label": label,
                    "smarts": smarts,
                }
        return None

    def _build_rule(
        self,
        text: str,
        task_type: str,
        source: str,
        feature_name: str,
        metadata: Dict[str, Any],
    ) -> Rule:
        rule_id = self._rule_id(f"{task_type}:{source}:{feature_name}:{text}")
        return Rule(
            rule_id=rule_id,
            task_type=task_type,
            source=source,
            text=text,
            feature_name=feature_name,
            code=self._code(rule_id, metadata),
            metadata=metadata,
            is_executable=True,
        )

    def _code(self, rule_id: str, metadata: Dict[str, Any]) -> str:
        operation = metadata.get("operation")
        fn_name = f"rule_{rule_id}"
        if operation == "descriptor_threshold":
            descriptor = metadata["descriptor"]
            op = metadata["operator"]
            threshold = metadata["threshold"]
            return (
                f"def {fn_name}(mol, descriptors):\n"
                f"    value = float(descriptors.get('{descriptor}', 0.0))\n"
                f"    return 1.0 if value {op} {threshold!r} else 0.0\n"
            )
        if operation == "substructure":
            smarts = metadata["smarts"]
            return (
                f"def {fn_name}(mol, descriptors):\n"
                f"    from rdkit import Chem\n"
                f"    pattern = Chem.MolFromSmarts({smarts!r})\n"
                f"    return 1.0 if pattern is not None and mol.HasSubstructMatch(pattern) else 0.0\n"
            )
        return f"def {fn_name}(mol, descriptors):\n    return 0.0\n"

    def _feature_name(self, metadata: Dict[str, Any]) -> str:
        operation = metadata.get("operation")
        if operation == "descriptor_threshold":
            descriptor = metadata.get("descriptor", "descriptor")
            op = metadata.get("operator", "op").replace(">", "gt").replace("<", "lt").replace("=", "eq")
            threshold = str(metadata.get("threshold", "x")).replace(".", "p").replace("-", "neg")
            return f"rule_{descriptor}_{op}_{threshold}"
        if operation == "substructure":
            label = metadata.get("label", "substructure").replace(" ", "_")
            return f"rule_has_{label}"
        return f"rule_{self._rule_id(str(metadata))}"

    def _rule_id(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

