from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from agentm.llm.provider import LLMClient, build_llm_client
from agentm.rules.compiler import RuleCompiler
from agentm.rules.library import RuleLibrary
from agentm.schema import Rule
from agentm.tools.descriptors import DESCRIPTOR_NAMES
from agentm.utils.io import ensure_dir, write_json, write_text


class RuleDiscoveryAgent:
    def __init__(
        self,
        output_dir: str | Path = "outputs/rules",
        llm_client: Optional[LLMClient] = None,
        llm_profile: Optional[str] = None,
        use_llm: Optional[bool] = None,
    ):
        self.output_dir = ensure_dir(output_dir)
        self.library = RuleLibrary()
        self.compiler = RuleCompiler()
        self.llm_client = llm_client or build_llm_client(profile_name=llm_profile)
        self.use_llm = self.llm_client.available if use_llm is None else use_llm

    def discover(
        self,
        task_type: str,
        dataset_name: str,
        training_samples: Optional[pd.DataFrame] = None,
    ) -> List[Rule]:
        llm_literature: List[Rule] = []
        llm_inferred: List[Rule] = []
        llm_errors: List[str] = []

        if self.use_llm and self.llm_client.available:
            llm_literature, err = self._llm_literature_rules(task_type, dataset_name)
            llm_errors.extend(err)
            if training_samples is not None:
                llm_inferred, err = self._llm_data_rules(task_type, dataset_name, training_samples)
                llm_errors.extend(err)

        fallback_literature = self.library.literature_rules(dataset_name, task_type)
        fallback_inferred = []
        if training_samples is not None:
            fallback_inferred = self.library.data_inferred_rules(training_samples, task_type=task_type)

        # Keep fallback rules as a safety net; merge removes exact duplicate signatures.
        rules = self.library.merge_rules(llm_literature, llm_inferred, fallback_literature, fallback_inferred)
        payload = {
            "task_type": task_type,
            "dataset_name": dataset_name,
            "llm_profile": self.llm_client.config.profile,
            "llm_provider": self.llm_client.config.provider,
            "llm_model": self.llm_client.config.model,
            "llm_used": bool(llm_literature or llm_inferred),
            "llm_errors": llm_errors,
            "rules": [rule.to_dict() for rule in rules],
            "non_executable": [rule.to_dict() for rule in rules if not rule.is_executable],
        }
        write_json(self.output_dir / f"{task_type}_{dataset_name}_rules.json", payload)
        return rules

    def _llm_literature_rules(self, task_type: str, dataset_name: str) -> tuple[List[Rule], List[str]]:
        prompt = self._literature_prompt(task_type, dataset_name)
        return self._generate_and_compile(
            prompt=prompt,
            task_type=task_type,
            dataset_name=dataset_name,
            source="llm_literature",
            artifact_prefix="llm_literature",
        )

    def _llm_data_rules(
        self,
        task_type: str,
        dataset_name: str,
        training_samples: pd.DataFrame,
    ) -> tuple[List[Rule], List[str]]:
        sample_json = self._sample_training_rows(training_samples)
        prompt = self._data_prompt(task_type, dataset_name, sample_json)
        return self._generate_and_compile(
            prompt=prompt,
            task_type=task_type,
            dataset_name=dataset_name,
            source="llm_data_inferred",
            artifact_prefix="llm_data_inferred",
        )

    def _generate_and_compile(
        self,
        *,
        prompt: str,
        task_type: str,
        dataset_name: str,
        source: str,
        artifact_prefix: str,
    ) -> tuple[List[Rule], List[str]]:
        errors: List[str] = []
        safe_dataset = "".join(ch if ch.isalnum() else "_" for ch in dataset_name)
        base = self.output_dir / f"{artifact_prefix}_{task_type}_{safe_dataset}"
        error_path = base.with_suffix(".error.json")
        write_text(base.with_suffix(".prompt.txt"), prompt)
        try:
            response = self.llm_client.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "You generate conservative, auditable molecular-property rules. "
                            "Return valid JSON only. Do not include markdown fences."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            write_text(base.with_suffix(".response.json"), response)
            parsed = self._parse_json_response(response)
            rules = self._compile_specs(parsed.get("rules", []), task_type=task_type, source=source)
            write_json(base.with_suffix(".compiled.json"), {"rules": [rule.to_dict() for rule in rules]})
            if error_path.exists():
                error_path.unlink()
            return rules, errors
        except Exception as exc:
            errors.append(f"{source}: {exc}")
            write_json(error_path, {"error": str(exc)})
            return [], errors

    def _compile_specs(self, specs: Iterable[Dict[str, Any]], task_type: str, source: str) -> List[Rule]:
        rules: List[Rule] = []
        for spec in specs:
            normalized = self._normalize_spec(spec)
            if normalized:
                rules.append(self.compiler.from_spec(normalized, task_type=task_type, source=source))
            else:
                text = str(spec.get("text") or spec)
                rules.append(self.compiler.compile_text(text, task_type=task_type, source=source))
        return rules

    def _normalize_spec(self, spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        rule_type = str(spec.get("rule_type") or spec.get("operation") or "").strip()
        if rule_type == "descriptor_threshold":
            descriptor = str(spec.get("descriptor") or "").strip()
            operator = str(spec.get("operator") or "").strip()
            if descriptor not in DESCRIPTOR_NAMES or operator not in {">", ">=", "<", "<=", "=="}:
                return None
            try:
                threshold = float(spec.get("threshold"))
            except Exception:
                return None
            return {
                "operation": "descriptor_threshold",
                "descriptor": descriptor,
                "operator": operator,
                "threshold": threshold,
                "text": str(spec.get("text") or f"{descriptor} {operator} {threshold}"),
                "rationale": spec.get("rationale"),
            }
        if rule_type == "substructure":
            label = str(spec.get("label") or spec.get("functional_group") or "").strip()
            smarts = str(spec.get("smarts") or "").strip()
            if not label or not smarts:
                return None
            return {
                "operation": "substructure",
                "label": label,
                "smarts": smarts,
                "text": str(spec.get("text") or f"Molecules containing {label} match this rule."),
                "rationale": spec.get("rationale"),
            }
        return None

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        text = response.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        data = json.loads(text)
        if isinstance(data, list):
            data = {"rules": data}
        if not isinstance(data, dict) or "rules" not in data:
            raise ValueError("LLM response must be a JSON object with a 'rules' array.")
        if not isinstance(data["rules"], list):
            raise ValueError("'rules' must be a list.")
        return data

    def _literature_prompt(self, task_type: str, dataset_name: str) -> str:
        return f"""
Generate LLM4SD-style literature rules for molecular property prediction.

Task type: {task_type}
Dataset/task name: {dataset_name}

Use only conservative medicinal-chemistry knowledge that can be converted into RDKit descriptors or SMARTS substructure checks.
Allowed descriptor names:
{", ".join(DESCRIPTOR_NAMES)}

Allowed rule JSON formats:
1. descriptor_threshold:
{{"text": "...", "rule_type": "descriptor_threshold", "descriptor": "TPSA", "operator": "<", "threshold": 90, "rationale": "..."}}
2. substructure:
{{"text": "...", "rule_type": "substructure", "label": "halogen", "smarts": "[F,Cl,Br,I]", "rationale": "..."}}

Return JSON only:
{{"rules": [ ... ]}}
Generate 3 to 6 rules.
""".strip()

    def _data_prompt(self, task_type: str, dataset_name: str, sample_json: str) -> str:
        return f"""
Generate LLM4SD-style data-inferred rules from these training examples.

Task type: {task_type}
Dataset/task name: {dataset_name}
Rows are JSON objects containing SMILES, label, and RDKit descriptor values.
Training sample summary:
{sample_json}

Allowed descriptor names:
{", ".join(DESCRIPTOR_NAMES)}

Rules must be computable and safe. Prefer descriptor_threshold rules. Use substructure rules only with valid SMARTS.
Return JSON only:
{{"rules": [{{"text": "...", "rule_type": "descriptor_threshold", "descriptor": "MolLogP", "operator": ">", "threshold": 2.0, "rationale": "..."}}]}}
Generate up to 5 rules.
""".strip()

    def _sample_training_rows(self, training_samples: pd.DataFrame, max_rows: int = 24) -> str:
        rows = []
        df = training_samples.head(max_rows).copy()
        smiles_col = "smiles" if "smiles" in df.columns else "SMILES" if "SMILES" in df.columns else None
        label_col = "label" if "label" in df.columns else None
        if smiles_col is None:
            return "[]"
        from agentm.tools.descriptors import compute_rdkit_descriptors
        from agentm.tools.smiles import mol_from_smiles

        for _, row in df.iterrows():
            mol = mol_from_smiles(str(row[smiles_col]))
            if mol is None:
                continue
            item: Dict[str, Any] = {"smiles": str(row[smiles_col])}
            if label_col:
                item["label"] = row[label_col]
            item.update(compute_rdkit_descriptors(mol))
            rows.append(item)
        return json.dumps(rows, indent=2, default=str)
