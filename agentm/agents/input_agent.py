from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from agentm.tools.name_resolver import LOCAL_NAME_TO_SMILES, resolve_name_to_smiles
from agentm.tools.smiles import likely_smiles, standardize_smiles


@dataclass
class ParsedInput:
    task_type: str
    dataset_name: str
    molecules: List[str]
    raw_request: str


class InputAgent:
    NON_MOLECULE_LATIN_TOKENS = {
        "admet",
        "bace",
        "bbbp",
        "clintox",
        "cyp1a2",
        "cyp2c9",
        "cyp2c19",
        "cyp2d6",
        "cyp3a4",
        "dataset",
        "ddi",
        "dili",
        "esol",
        "freesolv",
        "herg",
        "hiv",
        "interaction",
        "lipophilicity",
        "property",
        "prediction",
        "predict",
        "smiles",
        "task",
        "tox21",
    }
    DEFAULT_DATASET = {
        "property_prediction": "BBBP",
        "admet_prediction": "ClinTox",
        "ddi_prediction": "DDI",
    }
    DATASET_NAMES = {
        "BBBP",
        "BACE",
        "HIV",
        "ESOL",
        "FreeSolv",
        "Lipophilicity",
        "ClinTox",
        "Tox21",
        "SIDER",
        "CYP1A2",
        "CYP2C9",
        "CYP2C19",
        "CYP2D6",
        "CYP3A4",
        "hERG",
        "Ames",
        "DILI",
        "DDI",
        "ADMET",
    }

    def parse(
        self,
        request: str,
        task_type: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> ParsedInput:
        inferred_task = task_type or self._infer_task(request)
        inferred_dataset = dataset_name or self._infer_dataset(request, inferred_task)
        molecules = self.extract_molecule_mentions(request)
        return ParsedInput(
            task_type=inferred_task,
            dataset_name=inferred_dataset,
            molecules=molecules,
            raw_request=request,
        )

    def extract_molecule_mentions(self, request: str) -> List[str]:
        text = (request or "").strip()
        if not text:
            return []

        quoted = re.findall(r"['\"]([^'\"]+)['\"]", text)
        if quoted:
            return quoted[:2]

        known_names = self._known_drug_names_in_text(text)
        if len(known_names) >= 2:
            return known_names[:2]

        explicit_smiles = self._explicit_smiles_mentions(text)
        if known_names and explicit_smiles:
            return self._dedupe_mentions(known_names + explicit_smiles)[:2]
        if explicit_smiles:
            return explicit_smiles[:2]

        token_smiles = self._smiles_tokens_in_text(text)
        if known_names and token_smiles:
            return self._dedupe_mentions(known_names + token_smiles)[:2]
        if token_smiles:
            return token_smiles[:2]

        if known_names:
            return known_names[:2]

        latin_names = self._latin_name_mentions(text)
        if latin_names:
            return latin_names[:2]

        # Common DDI delimiters.
        normalized = re.sub(r"\b(?:and|with|vs|versus|plus)\b", ",", text, flags=re.IGNORECASE)
        normalized = re.sub(r"(一起用|联合用药|联合使用|共同使用|合用|联用|同服|配伍)", ",", normalized)
        normalized = re.sub(r"([A-Za-z0-9\]\)])\s*(?:以及|和|与|跟|同|及|加)\s*([A-Za-z0-9\[\(])", r"\1,\2", normalized)
        normalized = re.sub(r"[，、；;+/]", ",", normalized)
        parts = [p.strip(" .;:：。！？\n\t") for p in normalized.split(",") if p.strip(" .;:：。！？\n\t")]
        if len(parts) >= 2:
            cleaned_parts = [self._strip_task_words(part) for part in parts]
            cleaned_parts = [part for part in cleaned_parts if part]
            if cleaned_parts:
                return cleaned_parts[:2]

        tokens = text.split()
        if likely_smiles(text):
            return [text]
        # Last-resort name extraction: remove task verbs and keep the remaining phrase.
        cleaned = self._strip_task_words(text)
        if self._is_non_molecule_token(cleaned):
            return []
        return [cleaned] if cleaned else tokens[:1]

    def resolve_to_smiles(self, value: str) -> tuple[Optional[str], Optional[str], str]:
        cleaned_value = (value or "").strip()
        if cleaned_value.lower() in LOCAL_NAME_TO_SMILES:
            smiles, source = resolve_name_to_smiles(cleaned_value)
            return smiles, cleaned_value, source
        if likely_smiles(cleaned_value):
            canonical, err = standardize_smiles(cleaned_value)
            return canonical, None, "smiles" if not err else f"invalid_smiles:{err}"
        smiles, source = resolve_name_to_smiles(cleaned_value)
        return smiles, cleaned_value, source

    def _infer_task(self, request: str) -> str:
        lower = request.lower()
        if any(
            k in lower
            for k in [
                "ddi",
                "interaction",
                "drug pair",
                "coadmin",
                "co-admin",
                "coadministration",
                "co-administer",
                "相互作用",
                "一起用",
                "合用",
                "联用",
                "联合用药",
                "同服",
                "配伍",
            ]
        ):
            return "ddi_prediction"
        if any(
            k in lower
            for k in [
                "admet",
                "tox",
                "toxicity",
                "side effect",
                "cyp",
                "herg",
                "ames",
                "dili",
                "clintox",
                "sider",
                "毒",
                "毒性",
                "副作用",
                "不良反应",
                "代谢",
                "肝损伤",
                "心脏毒性",
                "致突变",
            ]
        ):
            return "admet_prediction"
        return "property_prediction"

    def _infer_dataset(self, request: str, task_type: str) -> str:
        candidates = [
            "CYP2C19 inhibition",
            "CYP1A2 inhibition",
            "CYP2C9 inhibition",
            "CYP2D6 inhibition",
            "CYP3A4 inhibition",
            "Lipophilicity",
            "FreeSolv",
            "ClinTox",
            "Tox21",
            "SIDER",
            "BBBP",
            "BACE",
            "HIV",
            "ESOL",
            "hERG",
            "Ames",
            "DILI",
        ]
        lower = request.lower()
        for candidate in candidates:
            if candidate.lower() in lower:
                return candidate
        keyword_map = [
            (["blood brain barrier", "blood-brain barrier", "bbb", "brain penetration", "血脑屏障", "脑屏障", "入脑", "透过脑"], "BBBP"),
            (["solubility", "aqueous solubility", "水溶性", "溶解度"], "ESOL"),
            (["hydration free energy", "free solvation", "freesolv", "溶剂化自由能"], "FreeSolv"),
            (["lipophilicity", "logd", "logp", "脂溶性", "亲脂性"], "Lipophilicity"),
            (["beta secretase", "bace", "β-secretase", "阿尔茨海默"], "BACE"),
            (["hiv", "抗hiv", "抗病毒"], "HIV"),
            (["clinical toxicity", "clinical tox", "clintox", "临床毒性"], "ClinTox"),
            (["tox21", "toxicity panel", "毒性面板"], "Tox21"),
            (["side effect", "side effects", "sider", "副作用", "不良反应"], "SIDER"),
            (["cyp1a2"], "CYP1A2 inhibition"),
            (["cyp2c9"], "CYP2C9 inhibition"),
            (["cyp2c19"], "CYP2C19 inhibition"),
            (["cyp2d6"], "CYP2D6 inhibition"),
            (["cyp3a4"], "CYP3A4 inhibition"),
            (["herg", "心脏毒性", "qt"], "hERG"),
            (["ames", "mutagenicity", "致突变"], "Ames"),
            (["dili", "drug induced liver injury", "肝损伤", "肝毒性"], "DILI"),
        ]
        for keywords, dataset in keyword_map:
            if any(keyword in lower for keyword in keywords):
                return dataset
        return self.DEFAULT_DATASET[task_type]

    def _strip_task_words(self, text: str) -> str:
        cleaned = re.sub(
            r"\b(predict|prediction|property|admet|ddi|interaction|for|please|run|task|dataset)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        for word in ["请", "预测", "判断", "评估", "任务", "数据集", "性质", "相互作用", "解释", "结构", "依据", "是否", "能否", "通过", "血脑屏障", "并", "给出"]:
            cleaned = cleaned.replace(word, " ")
        for phrase in [
            "会不会",
            "有没有",
            "有无",
            "是否存在",
            "存在",
            "风险",
            "一起用",
            "联合用药",
            "联合使用",
            "共同使用",
            "合用",
            "联用",
            "同服",
            "配伍",
            "用",
            "有",
        ]:
            cleaned = cleaned.replace(phrase, " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;:：，,。！？")
        return cleaned

    def _known_drug_names_in_text(self, text: str) -> List[str]:
        lower = text.lower()
        matches = []
        for name in sorted(LOCAL_NAME_TO_SMILES, key=len, reverse=True):
            pattern = rf"(?<![A-Za-z0-9_-]){re.escape(name)}(?![A-Za-z0-9_-])"
            match = re.search(pattern, lower)
            if match:
                matches.append((match.start(), name))
        matches.sort(key=lambda item: item[0])
        deduped: List[str] = []
        for _, name in matches:
            if name not in deduped:
                deduped.append(name)
        return deduped

    def _explicit_smiles_mentions(self, text: str) -> List[str]:
        mentions = []
        for match in re.finditer(r"(?:smiles|SMILES)\s*[:：]\s*([^\s,，;；。]+)", text):
            candidate = match.group(1).strip(" .;:：，,。！？")
            if likely_smiles(candidate):
                mentions.append(candidate)
        return mentions

    def _latin_name_mentions(self, text: str) -> List[str]:
        mentions: List[str] = []
        for match in re.finditer(r"[A-Za-z][A-Za-z0-9_-]{2,}", text):
            candidate = match.group(0).strip(" .;:：，,。！？\"'“”‘’")
            lower = candidate.lower()
            if lower in self.NON_MOLECULE_LATIN_TOKENS:
                continue
            if lower in {name.lower() for name in self.DATASET_NAMES}:
                continue
            if likely_smiles(candidate):
                continue
            if candidate not in mentions:
                mentions.append(candidate)
        return mentions

    def _smiles_tokens_in_text(self, text: str) -> List[str]:
        tokens = re.split(r"[\s,，;；。！？、]+", text)
        mentions: List[str] = []
        for token in tokens:
            candidate = token.strip(" .;:：，,。！？\"'“”‘’")
            if candidate.lower() in {name.lower() for name in self.DATASET_NAMES}:
                continue
            if candidate and likely_smiles(candidate):
                mentions.append(candidate)
        return mentions

    def _dedupe_mentions(self, mentions: List[str]) -> List[str]:
        deduped: List[str] = []
        for mention in mentions:
            if mention not in deduped:
                deduped.append(mention)
        return deduped

    def _is_non_molecule_token(self, value: str) -> bool:
        text = (value or "").strip()
        if not text:
            return True
        lower = text.lower()
        if lower in self.NON_MOLECULE_LATIN_TOKENS:
            return True
        if lower in {name.lower() for name in self.DATASET_NAMES}:
            return True
        return False
