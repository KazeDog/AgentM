from __future__ import annotations

from typing import Iterable, List

import pandas as pd

from agentm.rules.compiler import RuleCompiler
from agentm.schema import Rule
from agentm.tools.descriptors import DESCRIPTOR_NAMES, compute_rdkit_descriptors
from agentm.tools.smiles import mol_from_smiles


class RuleLibrary:
    """Rule generation inspired by LLM4SD.

    This first implementation keeps deterministic, auditable rules so the
    workflow runs without a live LLM. Real LLM providers can replace or augment
    these text rules before compilation.
    """

    LITERATURE_RULES = {
        "BBBP": [
            "Molecules with TPSA below 90 are more likely to cross the blood-brain barrier.",
            "Molecules with molecular weight below 450 are more likely to cross the blood-brain barrier.",
            "Molecules with hydrogen bond donor below 3 are more likely to be brain permeable.",
        ],
        "BACE": [
            "Higher MolLogP above 2 can support binding to hydrophobic BACE pockets.",
            "Molecules with molecular weight below 600 are preferred for drug-like BACE inhibitors.",
        ],
        "HIV": [
            "Halogen containing molecules can change antiviral potency and permeability.",
            "Aromatic ring above 1 can indicate hydrophobic pharmacophore content.",
        ],
        "ESOL": [
            "Molecules with MolLogP below 3 tend to have better aqueous solubility.",
            "Molecules with molecular weight below 350 tend to have better aqueous solubility.",
        ],
        "FreeSolv": [
            "Molecules with hydrogen bond acceptor above 2 can have stronger hydration effects.",
            "Molecules with MolLogP below 2 tend to have favorable hydration free energy.",
        ],
        "Lipophilicity": [
            "Molecules with MolLogP above 3 are more lipophilic.",
            "Halogen containing molecules can increase lipophilicity.",
        ],
        "admet_prediction": [
            "Molecules with molecular weight above 500 may have ADMET risk.",
            "Molecules with MolLogP above 5 may have ADMET risk.",
            "Molecules with TPSA above 140 may have poor permeability.",
            "Nitro groups can indicate toxicity alerts.",
            "Carboxylic acid groups can affect ionization and clearance.",
        ],
        "ddi_prediction": [
            "Molecules with MolLogP above 4 may have CYP-related interaction risk.",
            "Aromatic ring above 2 can indicate CYP binding risk.",
            "Amine groups can indicate ionizable centers relevant to transporter or CYP interactions.",
        ],
    }

    def __init__(self, compiler: RuleCompiler | None = None):
        self.compiler = compiler or RuleCompiler()

    def literature_rules(self, dataset_name: str, task_type: str) -> List[Rule]:
        keys = [dataset_name, task_type]
        texts: List[str] = []
        for key in keys:
            if key in self.LITERATURE_RULES:
                texts.extend(self.LITERATURE_RULES[key])
        if not texts:
            texts.extend(
                [
                    "Molecules with molecular weight above 500 may have drug-like property risk.",
                    "Molecules with MolLogP above 5 may have lipophilicity-driven risk.",
                    "Molecules with TPSA above 140 may have permeability risk.",
                ]
            )
        return [self.compiler.compile_text(text, task_type=task_type, source="literature") for text in texts]

    def data_inferred_rules(
        self,
        samples: pd.DataFrame,
        smiles_column: str = "smiles",
        label_column: str = "label",
        task_type: str = "property_prediction",
        max_rules: int = 8,
    ) -> List[Rule]:
        if samples is None or samples.empty or smiles_column not in samples or label_column not in samples:
            return []

        rows = []
        for _, row in samples.iterrows():
            mol = mol_from_smiles(str(row[smiles_column]))
            if mol is None:
                continue
            desc = compute_rdkit_descriptors(mol)
            desc[label_column] = row[label_column]
            rows.append(desc)
        if not rows:
            return []

        df = pd.DataFrame(rows)
        labels = pd.to_numeric(df[label_column], errors="coerce")
        if labels.dropna().nunique() < 2:
            return []
        positive = df[labels >= labels.median()]
        negative = df[labels < labels.median()]
        if positive.empty or negative.empty:
            return []

        candidates = []
        for descriptor in DESCRIPTOR_NAMES:
            pos_mean = float(positive[descriptor].mean())
            neg_mean = float(negative[descriptor].mean())
            delta = pos_mean - neg_mean
            if abs(delta) < 1e-6:
                continue
            threshold = float(df[descriptor].median())
            op = ">" if delta > 0 else "<"
            candidates.append((abs(delta), descriptor, op, threshold, pos_mean, neg_mean))
        candidates.sort(reverse=True)

        rules = []
        for _, descriptor, op, threshold, pos_mean, neg_mean in candidates[:max_rules]:
            spec = {
                "operation": "descriptor_threshold",
                "descriptor": descriptor,
                "operator": op,
                "threshold": threshold,
                "text": (
                    f"Data inferred: molecules with {descriptor} {op} {threshold:.3f} "
                    f"were enriched in higher labels (positive mean {pos_mean:.3f}, negative mean {neg_mean:.3f})."
                ),
            }
            rules.append(self.compiler.from_spec(spec, task_type=task_type, source="data_inferred"))
        return rules

    def merge_rules(self, *groups: Iterable[Rule]) -> List[Rule]:
        seen = set()
        merged: List[Rule] = []
        for group in groups:
            for rule in group:
                signature = rule.signature()
                if signature in seen:
                    continue
                seen.add(signature)
                merged.append(rule)
        return merged

