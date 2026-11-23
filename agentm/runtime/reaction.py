from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from agentm.runtime.compat import to_jsonable, write_json
from agentm.runtime.tools.registry import ToolResult


DEFAULT_REACTION_DATA_ROOT = Path(os.environ.get(
    "AGENTM_REACTION_DATA_ROOT",
    str(Path(__file__).resolve().parents[2] / "data" / "reaction_sources"),
))
DEFAULT_ORDERLY_TRAIN = DEFAULT_REACTION_DATA_ROOT / "ORDerly" / "orderly_condition_with_rare_train_dedup_uspto_condition_valtest.parquet"
DEFAULT_ORDERLY_TEST = DEFAULT_REACTION_DATA_ROOT / "ORDerly" / "orderly_condition_with_rare_test_dedup_uspto_condition_valtest.parquet"

REACTION_TASK_TYPES = {
    "reaction_profile",
    "reaction_classification",
    "reaction_yield_prediction",
    "reaction_condition_retrieval",
}


@dataclass
class ReactionResourceResolver:
    data_root: Path = DEFAULT_REACTION_DATA_ROOT

    def resolve(self) -> Dict[str, Any]:
        root = Path(self.data_root)
        resources = {
            "data_root": str(root),
            "read_only_source_root": True,
            "mit": self._entry(root / "MIT", required_files=["train.csv", "valid.csv", "test.csv"]),
            "orderly": {
                "path": str(root / "ORDerly"),
                "exists": (root / "ORDerly").exists(),
                "preferred_train": self._file(root / "ORDerly" / DEFAULT_ORDERLY_TRAIN.name),
                "preferred_test": self._file(root / "ORDerly" / DEFAULT_ORDERLY_TEST.name),
                "fallback_parquet_files": [str(path) for path in sorted((root / "ORDerly").glob("*.parquet"))[:12]],
            },
            "schneider": self._entry(root / "schneider" / "processed_reaction_class", required_files=["train_split.csv", "val_split.csv", "test_split.csv"]),
            "buchwald_hartwig_kg": {
                "path": str(root / "processed_yield" / "buchwald_hartwig_kg"),
                "exists": (root / "processed_yield" / "buchwald_hartwig_kg").exists(),
                "splits": {
                    split.name: self._entry(split, required_files=["train_split.csv", "val_split.csv", "test_split.csv"])
                    for split in sorted((root / "processed_yield" / "buchwald_hartwig_kg").glob("test*"))
                    if split.is_dir()
                },
            },
        }
        resources["all_required_present"] = all(
            [
                resources["mit"]["exists"],
                resources["schneider"]["exists"],
                resources["buchwald_hartwig_kg"]["exists"],
                resources["orderly"]["exists"],
            ]
        )
        return resources

    def _entry(self, path: Path, *, required_files: Iterable[str]) -> Dict[str, Any]:
        files = {name: self._file(path / name) for name in required_files}
        return {
            "path": str(path),
            "exists": path.exists(),
            "required_files": files,
            "all_required_present": path.exists() and all(item["exists"] for item in files.values()),
        }

    def _file(self, path: Path) -> Dict[str, Any]:
        return {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}


class ReactionRuntime:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        data_root: str | Path = DEFAULT_REACTION_DATA_ROOT,
        model_dir: str | Path | None = None,
        index_dir: str | Path | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.data_root = Path(data_root)
        self.model_dir = Path(model_dir) if model_dir else self.output_dir / "reaction_models"
        self.index_dir = Path(index_dir) if index_dir else self.output_dir / "reaction_indices"
        self.resolver = ReactionResourceResolver(self.data_root)

    def resolve_resources(self) -> Dict[str, Any]:
        return self.resolver.resolve()

    def parse_reaction(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        text = str(args.get("reaction_smiles") or context.get("reaction_request") or context.get("request") or "")
        reaction = parse_reaction_text(text)
        if not reaction.get("reaction_smiles"):
            context.setdefault("warnings", []).append("reaction_parse_failed:no_reaction_smiles_found")
            return ToolResult("parse_reaction", "failed", {"reaction": reaction}, error="no_reaction_smiles_found")
        context["reaction"] = reaction
        return ToolResult("parse_reaction", "completed", {"reaction": reaction})

    def retrieve_mit_template_evidence(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        reaction = _reaction_or_parse(args, context)
        rows = _read_csv_rows(self.data_root / "MIT" / "train.csv", max_rows=int(args.get("max_rows") or 12000))
        evidence = []
        for idx, row in enumerate(_top_rows_by_similarity(rows, reaction, reactant_key="reactant_smiles", product_key="prod_smiles", k=int(args.get("top_k") or 5))):
            claim = f"MIT similar reaction template {row.get('template_r') or row.get('reaction_type') or 'unknown'} maps related reactants to product-like outcomes."
            item = {
                "citation_id": f"mit_template_{idx + 1}",
                "source": "mit_reaction_template",
                "claim": claim,
                "reactant_smiles": row.get("reactant_smiles"),
                "product_smiles": row.get("prod_smiles") or row.get("product_smiles"),
                "template_r": row.get("template_r"),
                "template_p": row.get("template_p"),
                "score": row.get("_similarity"),
                "reaction_specific": True,
            }
            evidence.append(item)
        _append_reaction_evidence(context, evidence)
        context.setdefault("artifacts", {})["reaction_template_evidence"] = evidence
        return ToolResult("retrieve_mit_template_evidence", "completed", {"n_evidence_items": len(evidence), "evidence_items": evidence})

    def retrieve_orderly_condition_evidence(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        reaction = _reaction_or_parse(args, context)
        rows, source_path = self._read_orderly_rows(max_rows=int(args.get("max_rows") or 12000))
        evidence = []
        for idx, row in enumerate(_top_rows_by_similarity(rows, reaction, reactant_key="rxn_str", product_key="product_000", k=int(args.get("top_k") or 5))):
            conditions = {
                "agents": _values(row, "agent_"),
                "solvents": _values(row, "solvent_"),
                "temperature": row.get("temperature"),
                "rxn_time": row.get("rxn_time"),
                "yield": row.get("yield_000"),
                "procedure": _shorten(row.get("procedure_details"), 360),
            }
            parts = []
            if conditions["solvents"]:
                parts.append("solvent(s): " + ", ".join(conditions["solvents"][:3]))
            if conditions["agents"]:
                parts.append("agent(s): " + ", ".join(conditions["agents"][:4]))
            if conditions["temperature"] not in {None, "", "nan"}:
                parts.append(f"temperature: {conditions['temperature']}")
            if conditions["yield"] not in {None, "", "nan"}:
                parts.append(f"reported yield: {conditions['yield']}")
            claim = "Similar ORDerly reactions provide condition context" + (": " + "; ".join(parts) if parts else ".")
            item = {
                "citation_id": f"orderly_condition_{idx + 1}",
                "source": "orderly_condition",
                "source_path": source_path,
                "claim": claim,
                "reaction_smiles": row.get("rxn_str"),
                "conditions": conditions,
                "score": row.get("_similarity"),
                "reaction_specific": True,
            }
            evidence.append(item)
        _append_reaction_evidence(context, evidence)
        context.setdefault("artifacts", {})["reaction_condition_evidence"] = evidence
        return ToolResult("retrieve_orderly_condition_evidence", "completed", {"n_evidence_items": len(evidence), "evidence_items": evidence})

    def classify_reaction_type(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        reaction = _reaction_or_parse(args, context)
        model = self._load_or_build_schneider_model(max_rows=int(args.get("max_rows") or 8000))
        pred = _classify_with_model(reaction, model)
        prediction = {
            "task_type": "reaction_classification",
            "dataset_name": "schneider",
            "label": pred["label"],
            "score": pred["confidence"],
            "probability": pred["confidence"],
            "model_type": model.get("model_type", "schneider_token_knn_classifier"),
            "details": pred,
        }
        context.setdefault("reaction_predictions", []).append(prediction)
        context["predictions"] = [prediction]
        return ToolResult("classify_reaction_type", "completed", {"prediction": prediction, "model_path": model.get("model_path")})

    def predict_buchwald_hartwig_yield(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        reaction = _reaction_or_parse(args, context)
        split = _bh_split_from_args(args, context)
        model = self._load_or_build_bh_model(split=split, max_rows=int(args.get("max_rows") or 4000))
        pred = _regress_with_model(reaction, model)
        label = "high_yield" if pred["predicted_yield"] >= 70 else "moderate_yield" if pred["predicted_yield"] >= 30 else "low_yield"
        prediction = {
            "task_type": "reaction_yield_prediction",
            "dataset_name": f"buchwald_hartwig_kg:{split}",
            "label": label,
            "score": pred["predicted_yield"],
            "probability": None,
            "model_type": model.get("model_type", "buchwald_hartwig_token_knn_regressor"),
            "details": pred,
        }
        context.setdefault("reaction_predictions", []).append(prediction)
        context["predictions"] = [prediction]
        return ToolResult("predict_buchwald_hartwig_yield", "completed", {"prediction": prediction, "model_path": model.get("model_path")})

    def build_reaction_profile(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        reaction = context.get("reaction") or parse_reaction_text(str(context.get("request") or ""))
        template_evidence = context.get("artifacts", {}).get("reaction_template_evidence", [])
        condition_evidence = context.get("artifacts", {}).get("reaction_condition_evidence", [])
        predictions = context.get("reaction_predictions") or context.get("predictions") or []
        profile = {
            "reaction": reaction,
            "predictions": predictions,
            "template_evidence_count": len(template_evidence),
            "condition_evidence_count": len(condition_evidence),
            "summary": reaction_profile_summary(context, reaction, predictions, template_evidence, condition_evidence),
        }
        context["reaction_profile"] = profile
        context.setdefault("reasoning", {})["structured_explanation"] = profile["summary"]
        context.setdefault("artifacts", {})["reaction_profile"] = profile
        return ToolResult("build_reaction_profile", "completed", {"reaction_profile": profile})

    def build_indices(self, *, dataset: str, max_records: Optional[int], output_dir: str | Path | None = None, shards: int = 1) -> Dict[str, Any]:
        target = Path(output_dir) if output_dir else self.index_dir
        target.mkdir(parents=True, exist_ok=True)
        if dataset == "all":
            manifests = [
                self.build_indices(dataset="mit", max_records=max_records, output_dir=target, shards=shards),
                self.build_indices(dataset="orderly", max_records=max_records, output_dir=target, shards=shards),
            ]
            manifest = {
                "dataset": "all",
                "n_docs": sum(item.get("n_docs", 0) for item in manifests),
                "children": manifests,
                "source_root": str(self.data_root),
                "read_only_source_root": True,
            }
            write_json(target / "all_reaction_index_manifest.json", manifest)
            return manifest
        if dataset == "mit":
            rows = _read_csv_rows(self.data_root / "MIT" / "train.csv", max_rows=max_records)
            docs = [
                {
                    "id": f"mit_{idx + 1}",
                    "source": "mit",
                    "reaction_hash": _hash_text(" ".join(str(row.get(key) or "") for key in ["reactant_smiles", "prod_smiles", "template_r", "template_p"])),
                    "reactant_smiles": row.get("reactant_smiles"),
                    "product_smiles": row.get("prod_smiles"),
                    "template_r": row.get("template_r"),
                    "template_p": row.get("template_p"),
                    "token_metadata": sorted(_reaction_tokens({"reactants": row.get("reactant_smiles"), "products": row.get("prod_smiles")}))[:64],
                    "text": " ".join(str(row.get(key) or "") for key in ["reactant_smiles", "prod_smiles", "template_r", "template_p"]),
                }
                for idx, row in enumerate(rows)
            ]
        elif dataset == "orderly":
            rows, source_path = self._read_orderly_rows(max_rows=max_records)
            docs = [
                {
                    "id": f"orderly_{idx + 1}",
                    "source": "orderly",
                    "source_path": source_path,
                    "reaction_hash": _hash_text(str(row.get("rxn_str") or "")),
                    "reaction_smiles": row.get("rxn_str"),
                    "product": row.get("product_000"),
                    "agents": _values(row, "agent_"),
                    "solvents": _values(row, "solvent_"),
                    "temperature": row.get("temperature"),
                    "rxn_time": row.get("rxn_time"),
                    "yield": row.get("yield_000"),
                    "token_metadata": sorted(_reaction_tokens({"reaction_smiles": row.get("rxn_str"), "products": row.get("product_000")}))[:64],
                    "text": " ".join(str(row.get(key) or "") for key in row),
                }
                for idx, row in enumerate(rows)
            ]
        else:
            raise ValueError(f"unsupported reaction index dataset: {dataset}")
        index_paths = _write_jsonl_shards(target, dataset, docs, max(1, int(shards or 1)))
        manifest = {
            "dataset": dataset,
            "n_docs": len(docs),
            "index_path": index_paths[0] if len(index_paths) == 1 else None,
            "index_paths": index_paths,
            "n_shards": len(index_paths),
            "source_root": str(self.data_root),
            "read_only_source_root": True,
        }
        write_json(target / f"{dataset}_reaction_index_manifest.json", manifest)
        return manifest

    def train_model(
        self,
        *,
        task: str,
        split: str = "test1",
        max_rows: Optional[int] = None,
        output_dir: str | Path | None = None,
        model_family: str = "tfidf_linear",
    ) -> Dict[str, Any]:
        target = Path(output_dir) if output_dir else self.model_dir
        if task == "schneider_classification":
            if model_family == "token_knn":
                return self._build_schneider_model(max_rows=max_rows or 8000, output_dir=target / "schneider_classification")
            if model_family == "tfidf_linear":
                return self._build_schneider_tfidf_model(max_rows=max_rows, output_dir=target / "schneider_classification")
        if task == "buchwald_hartwig_yield":
            if model_family == "token_knn":
                return self._build_bh_model(split=split, max_rows=max_rows or 4000, output_dir=target / "buchwald_hartwig_kg" / split)
            if model_family == "tfidf_linear":
                return self._build_bh_tfidf_model(split=split, max_rows=max_rows, output_dir=target / "buchwald_hartwig_kg" / split)
        raise ValueError(f"unsupported reaction model task: {task}")

    def _read_orderly_rows(self, *, max_rows: Optional[int]) -> tuple[List[Dict[str, Any]], str]:
        candidates = [DEFAULT_ORDERLY_TRAIN, DEFAULT_ORDERLY_TEST]
        candidates.extend(sorted((self.data_root / "ORDerly").glob("orderly_condition*.parquet")))
        candidates.extend(sorted((self.data_root / "ORDerly").glob("orderly_forward*.parquet")))
        for path in candidates:
            if not path.exists():
                continue
            try:
                import pandas as pd

                frame = pd.read_parquet(path)
                if max_rows:
                    frame = frame.head(max_rows)
                frame = frame.where(frame.notna(), None)
                return frame.to_dict(orient="records"), str(path)
            except Exception:
                continue
        return [], ""

    def _load_or_build_schneider_model(self, *, max_rows: int) -> Dict[str, Any]:
        full_path = self.model_dir / "schneider_classification" / "reaction_tfidf_linear_classifier.pkl"
        if full_path.exists():
            return _load_pickle_model(full_path, "reaction_tfidf_linear_classifier")
        path = self.model_dir / "schneider_classification" / "reaction_classifier.json"
        if path.exists():
            model = json.loads(path.read_text(encoding="utf-8"))
            model["model_path"] = str(path)
            return model
        return self._build_schneider_model(max_rows=max_rows, output_dir=path.parent)

    def _build_schneider_tfidf_model(self, *, max_rows: Optional[int], output_dir: Path) -> Dict[str, Any]:
        rows = _read_csv_rows(self.data_root / "schneider" / "processed_reaction_class" / "train_split.csv", max_rows=max_rows)
        texts: List[str] = []
        labels: List[str] = []
        for row in rows:
            label = str(row.get("labels") or row.get("reaction_type") or "")
            if not label:
                continue
            texts.append(_reaction_text_from_row(row))
            labels.append(label)
        if not texts:
            raise ValueError("no Schneider training rows available")
        from sklearn.dummy import DummyClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC

        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, max_features=250000)
        matrix = vectorizer.fit_transform(texts)
        classifier = DummyClassifier(strategy="most_frequent") if len(set(labels)) < 2 else LinearSVC(class_weight="balanced")
        classifier.fit(matrix, labels)
        artifact = {
            "model_type": "reaction_tfidf_linear_classifier",
            "dataset": "schneider",
            "vectorizer": vectorizer,
            "model": classifier,
            "labels": sorted(set(labels)),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "reaction_tfidf_linear_classifier.pkl"
        with path.open("wb") as handle:
            pickle.dump(artifact, handle)
        metadata = {
            "model_type": "reaction_tfidf_linear_classifier",
            "dataset": "schneider",
            "model_family": "tfidf_linear",
            "n_examples": len(texts),
            "label_counts": dict(Counter(labels)),
            "model_path": str(path),
            "metadata_path": str(output_dir / "reaction_tfidf_linear_classifier.meta.json"),
        }
        write_json(output_dir / "reaction_tfidf_linear_classifier.meta.json", metadata)
        return metadata

    def _build_schneider_model(self, *, max_rows: int, output_dir: Path) -> Dict[str, Any]:
        rows = _read_csv_rows(self.data_root / "schneider" / "processed_reaction_class" / "train_split.csv", max_rows=max_rows)
        examples = []
        labels = Counter()
        for row in rows:
            label = str(row.get("labels") or row.get("reaction_type") or "")
            if not label:
                continue
            labels[label] += 1
            examples.append(
                {
                    "reactant_smiles": row.get("reactant_smiles"),
                    "product_smiles": row.get("product_smiles"),
                    "label": label,
                    "tokens": sorted(_reaction_tokens({"reactants": row.get("reactant_smiles"), "products": row.get("product_smiles")})),
                }
            )
        model = {
            "model_type": "schneider_token_knn_classifier",
            "dataset": "schneider",
            "n_examples": len(examples),
            "majority_label": labels.most_common(1)[0][0] if labels else "",
            "label_counts": dict(labels),
            "examples": examples[: max_rows],
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "reaction_classifier.json"
        write_json(path, model)
        model["model_path"] = str(path)
        return model

    def _load_or_build_bh_model(self, *, split: str, max_rows: int) -> Dict[str, Any]:
        full_path = self.model_dir / "buchwald_hartwig_kg" / split / "yield_tfidf_ridge_regressor.pkl"
        if full_path.exists():
            return _load_pickle_model(full_path, "reaction_tfidf_ridge_regressor")
        path = self.model_dir / "buchwald_hartwig_kg" / split / "yield_regressor.json"
        if path.exists():
            model = json.loads(path.read_text(encoding="utf-8"))
            model["model_path"] = str(path)
            return model
        return self._build_bh_model(split=split, max_rows=max_rows, output_dir=path.parent)

    def _build_bh_tfidf_model(self, *, split: str, max_rows: Optional[int], output_dir: Path) -> Dict[str, Any]:
        rows = _read_csv_rows(self.data_root / "processed_yield" / "buchwald_hartwig_kg" / split / "train_split.csv", max_rows=max_rows)
        texts: List[str] = []
        values: List[float] = []
        for row in rows:
            value = _safe_float(row.get("yield"))
            if value is None:
                continue
            texts.append(_reaction_text_from_row(row))
            values.append(value)
        if not texts:
            raise ValueError(f"no Buchwald-Hartwig training rows available for {split}")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge

        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, max_features=250000)
        matrix = vectorizer.fit_transform(texts)
        regressor = Ridge(alpha=1.0)
        regressor.fit(matrix, values)
        artifact = {
            "model_type": "reaction_tfidf_ridge_regressor",
            "dataset": "buchwald_hartwig_kg",
            "split": split,
            "vectorizer": vectorizer,
            "model": regressor,
            "mean_yield": sum(values) / len(values),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "yield_tfidf_ridge_regressor.pkl"
        with path.open("wb") as handle:
            pickle.dump(artifact, handle)
        metadata = {
            "model_type": "reaction_tfidf_ridge_regressor",
            "dataset": "buchwald_hartwig_kg",
            "split": split,
            "model_family": "tfidf_linear",
            "n_examples": len(texts),
            "mean_yield": sum(values) / len(values),
            "model_path": str(path),
            "metadata_path": str(output_dir / "yield_tfidf_ridge_regressor.meta.json"),
        }
        write_json(output_dir / "yield_tfidf_ridge_regressor.meta.json", metadata)
        return metadata

    def _build_bh_model(self, *, split: str, max_rows: int, output_dir: Path) -> Dict[str, Any]:
        rows = _read_csv_rows(self.data_root / "processed_yield" / "buchwald_hartwig_kg" / split / "train_split.csv", max_rows=max_rows)
        examples = []
        yields = []
        for row in rows:
            try:
                value = float(row.get("yield"))
            except (TypeError, ValueError):
                continue
            yields.append(value)
            examples.append(
                {
                    "reaction_smiles": row.get("reaction_smiles"),
                    "reactant_smiles": row.get("reactant_smiles"),
                    "product_smiles": row.get("product_smiles"),
                    "aux_smiles": row.get("aux_smiles"),
                    "yield": value,
                    "tokens": sorted(_reaction_tokens({"reaction_smiles": row.get("reaction_smiles"), "aux_smiles": row.get("aux_smiles")})),
                }
            )
        mean_yield = sum(yields) / len(yields) if yields else 0.0
        model = {
            "model_type": "buchwald_hartwig_token_knn_regressor",
            "dataset": "buchwald_hartwig_kg",
            "split": split,
            "n_examples": len(examples),
            "mean_yield": mean_yield,
            "examples": examples[: max_rows],
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "yield_regressor.json"
        write_json(path, model)
        model["model_path"] = str(path)
        return model


def is_reaction_task(task_type: Any) -> bool:
    return str(task_type or "") in REACTION_TASK_TYPES


def is_reaction_request(text: str, explicit_task: Optional[str] = None) -> bool:
    if is_reaction_task(explicit_task):
        return True
    lower = str(text or "").lower()
    markers = [">>", "reaction", "synthesis", "condition", "yield", "产率", "反应类型", "合成条件", "反应", "催化剂", "溶剂"]
    return any(marker in lower for marker in markers)


def infer_reaction_task_type(text: str, explicit_task: Optional[str] = None) -> str:
    if is_reaction_task(explicit_task):
        return str(explicit_task)
    lower = str(text or "").lower()
    if "yield" in lower or "产率" in lower:
        return "reaction_yield_prediction"
    if "class" in lower or "type" in lower or "反应类型" in lower or "分类" in lower:
        return "reaction_classification"
    if "condition" in lower or "条件" in lower or "溶剂" in lower or "催化剂" in lower:
        return "reaction_condition_retrieval"
    return "reaction_profile"


def parse_reaction_text(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    match = re.search(r"([A-Za-z0-9@+\-\[\]\(\)=#$\\/%.:]+(?:\.[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.:]+)*>>[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.:]+(?:\.[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.:]+)*)", raw)
    reaction_smiles = match.group(1) if match else (raw if ">>" in raw and " " not in raw else "")
    reactants = ""
    products = ""
    if ">>" in reaction_smiles:
        reactants, products = reaction_smiles.split(">>", 1)
    return {
        "reaction_smiles": reaction_smiles,
        "reactants": reactants,
        "products": products,
        "reactant_list": [part for part in reactants.split(".") if part],
        "product_list": [part for part in products.split(".") if part],
        "is_valid_reaction": bool(reactants and products),
        "source_text": raw,
    }


def reaction_profile_summary(
    context: Mapping[str, Any],
    reaction: Mapping[str, Any],
    predictions: List[Dict[str, Any]],
    template_evidence: List[Dict[str, Any]],
    condition_evidence: List[Dict[str, Any]],
) -> str:
    chinese = any("\u4e00" <= char <= "\u9fff" for char in str(context.get("request") or ""))
    if chinese:
        parts = ["结论：AgentM 已将输入解析为化学反应任务。"]
        if predictions:
            pred = predictions[-1]
            if pred.get("task_type") == "reaction_classification":
                parts.append(f"Schneider 反应分类模型给出的类别为 {pred.get('label')}，置信度为定性参考。")
            elif pred.get("task_type") == "reaction_yield_prediction":
                parts.append(f"Buchwald-Hartwig KG 产率模型倾向判断该反应处于 {pred.get('label')} 区间。")
        if condition_evidence:
            parts.append(f"ORDerly 中检索到 {len(condition_evidence)} 条相似反应条件，可作为溶剂、试剂、温度、时间和产率背景。")
        if template_evidence:
            parts.append(f"MIT 反应库中检索到 {len(template_evidence)} 条模板/产物相似证据。")
        parts.append("这些条件是相似反应中的观察背景，不等同于经过优化的推荐条件。")
        return " ".join(parts)
    parts = ["Conclusion: AgentM parsed the input as a chemistry reaction task."]
    if predictions:
        pred = predictions[-1]
        if pred.get("task_type") == "reaction_classification":
            parts.append(f"The Schneider reaction-classification model assigns class {pred.get('label')} with qualitative confidence.")
        elif pred.get("task_type") == "reaction_yield_prediction":
            parts.append(f"The Buchwald-Hartwig KG yield model places the reaction in the {pred.get('label')} range.")
    if condition_evidence:
        parts.append(f"ORDerly supplied {len(condition_evidence)} similar-condition records covering solvent, agent, temperature, time, or yield context.")
    if template_evidence:
        parts.append(f"MIT supplied {len(template_evidence)} template/product-similarity records.")
    parts.append("Retrieved conditions are observed context from similar reactions, not optimized reaction-condition recommendations.")
    return " ".join(parts)


def _reaction_or_parse(args: Mapping[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    reaction = context.get("reaction")
    if isinstance(reaction, dict) and reaction.get("reaction_smiles"):
        return reaction
    reaction = parse_reaction_text(str(args.get("reaction_smiles") or context.get("request") or ""))
    context["reaction"] = reaction
    return reaction


def _bh_split_from_args(args: Mapping[str, Any], context: Mapping[str, Any]) -> str:
    explicit = str(args.get("split") or "").strip()
    if explicit:
        return explicit
    for value in [context.get("dataset_name"), context.get("task_dataset"), context.get("dataset")]:
        text = str(value or "")
        match = re.search(r"(test[1-4])", text)
        if match:
            return match.group(1)
    return "test1"


def _read_csv_rows(path: Path, *, max_rows: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def _top_rows_by_similarity(rows: List[Dict[str, Any]], reaction: Mapping[str, Any], *, reactant_key: str, product_key: str, k: int) -> List[Dict[str, Any]]:
    query_tokens = _reaction_tokens(reaction)
    scored = []
    for row in rows:
        row_tokens = _reaction_tokens({"reactants": row.get(reactant_key), "products": row.get(product_key), "reaction_smiles": row.get(reactant_key)})
        score = _jaccard(query_tokens, row_tokens)
        if score <= 0 and query_tokens:
            continue
        copy = dict(row)
        copy["_similarity"] = round(score, 6)
        scored.append(copy)
    scored.sort(key=lambda item: item.get("_similarity", 0.0), reverse=True)
    return scored[:k]


def _knn_classify(reaction: Mapping[str, Any], model: Mapping[str, Any]) -> Dict[str, Any]:
    query = _reaction_tokens(reaction)
    best = []
    for example in model.get("examples", []) or []:
        score = _jaccard(query, set(example.get("tokens", []) or []))
        best.append((score, str(example.get("label") or "")))
    best.sort(reverse=True)
    top = [item for item in best[:5] if item[1]]
    if not top:
        return {"label": model.get("majority_label", ""), "confidence": 0.0, "neighbors": []}
    votes = Counter(label for _, label in top)
    label, count = votes.most_common(1)[0]
    confidence = max(top[0][0], count / max(len(top), 1))
    return {"label": label, "confidence": round(float(confidence), 4), "neighbors": [{"score": score, "label": label} for score, label in top]}


def _knn_regress(reaction: Mapping[str, Any], model: Mapping[str, Any]) -> Dict[str, Any]:
    query = _reaction_tokens(reaction)
    best = []
    for example in model.get("examples", []) or []:
        score = _jaccard(query, set(example.get("tokens", []) or []))
        if score > 0:
            best.append((score, float(example.get("yield", 0.0))))
    best.sort(reverse=True)
    top = best[:5]
    if not top:
        value = float(model.get("mean_yield", 0.0))
        return {"predicted_yield": round(value, 3), "confidence": 0.0, "neighbors": []}
    denom = sum(score for score, _ in top) or len(top)
    value = sum(score * yield_value for score, yield_value in top) / denom
    confidence = min(1.0, max(score for score, _ in top))
    return {"predicted_yield": round(value, 3), "confidence": round(confidence, 4), "neighbors": [{"score": score, "yield": y} for score, y in top]}


def _classify_with_model(reaction: Mapping[str, Any], model: Mapping[str, Any]) -> Dict[str, Any]:
    if model.get("artifact_type") == "sklearn_pickle":
        artifact = model.get("artifact") or {}
        vectorizer = artifact.get("vectorizer")
        classifier = artifact.get("model")
        if vectorizer is None or classifier is None:
            return {"label": "", "confidence": 0.0, "model_error": "missing_sklearn_artifact"}
        text = _reaction_text_from_reaction(reaction)
        matrix = vectorizer.transform([text])
        label = str(classifier.predict(matrix)[0])
        confidence = 0.0
        if hasattr(classifier, "decision_function"):
            raw = classifier.decision_function(matrix)
            values = raw[0] if hasattr(raw, "__len__") and getattr(raw, "ndim", 1) > 1 else raw
            try:
                if hasattr(values, "__iter__"):
                    margin = max(float(v) for v in values)
                else:
                    margin = abs(float(values))
                confidence = 1.0 / (1.0 + math.exp(-margin))
            except Exception:
                confidence = 0.0
        # The prediction label is serialized as str, so serialize the fitted
        # estimator vocabulary in the same type. Never infer it from gold.
        classes = getattr(classifier, "classes_", None)
        return {"label": label, "confidence": round(float(confidence), 4), "model_family": "tfidf_linear",
                "classes": [str(value) for value in classes] if classes is not None else [],
                "class_vocabulary_source": "fitted_estimator.classes_"}
    prediction = _knn_classify(reaction, model)
    prediction['classes'] = sorted({str(row['label']) for row in model.get('examples', [])
                                    if row.get('label') is not None})
    prediction['class_vocabulary_source'] = 'stored_training_examples'
    return prediction


def _regress_with_model(reaction: Mapping[str, Any], model: Mapping[str, Any]) -> Dict[str, Any]:
    if model.get("artifact_type") == "sklearn_pickle":
        artifact = model.get("artifact") or {}
        vectorizer = artifact.get("vectorizer")
        regressor = artifact.get("model")
        if vectorizer is None or regressor is None:
            fallback = float(artifact.get("mean_yield") or 0.0)
            return {"predicted_yield": round(fallback, 3), "confidence": 0.0, "model_error": "missing_sklearn_artifact"}
        matrix = vectorizer.transform([_reaction_text_from_reaction(reaction)])
        value = float(regressor.predict(matrix)[0])
        value = max(0.0, min(100.0, value))
        return {"predicted_yield": round(value, 3), "confidence": 1.0, "model_family": "tfidf_linear"}
    return _knn_regress(reaction, model)


def _load_pickle_model(path: Path, model_type: str) -> Dict[str, Any]:
    with path.open("rb") as handle:
        artifact = pickle.load(handle)
    return {
        "artifact_type": "sklearn_pickle",
        "model_type": artifact.get("model_type") or model_type,
        "dataset": artifact.get("dataset"),
        "split": artifact.get("split"),
        "model_path": str(path),
        "artifact": artifact,
    }


def _reaction_text_from_row(row: Mapping[str, Any]) -> str:
    reaction_smiles = reaction_smiles_from_row(row)
    parts = [
        reaction_smiles,
        str(row.get("reactant_smiles") or ""),
        str(row.get("product_smiles") or row.get("prod_smiles") or ""),
        str(row.get("aux_smiles") or ""),
        str(row.get("template_smiles") or ""),
        str(row.get("template_r") or ""),
        str(row.get("template_p") or ""),
    ]
    return " ".join(part for part in parts if part and part.lower() != "nan")


def _reaction_text_from_reaction(reaction: Mapping[str, Any]) -> str:
    return " ".join(
        str(reaction.get(key) or "")
        for key in ["reaction_smiles", "reactants", "products", "reactant_smiles", "product_smiles", "aux_smiles"]
    )


def reaction_smiles_from_row(row: Mapping[str, Any]) -> str:
    if row.get("reaction_smiles"):
        return str(row.get("reaction_smiles"))
    reactants = str(row.get("reactant_smiles") or row.get("reactants") or "")
    products = str(row.get("product_smiles") or row.get("prod_smiles") or row.get("products") or "")
    return f"{reactants}>>{products}" if reactants and products else ""


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value in {None, "", "nan"}:
            return None
        parsed = float(value)
        if math.isnan(parsed):
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def _hash_text(text: str) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8", errors="ignore")).hexdigest()


def _write_jsonl_shards(target: Path, dataset: str, docs: List[Dict[str, Any]], shards: int) -> List[str]:
    if shards <= 1:
        path = target / f"{dataset}_reaction_index.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for doc in docs:
                handle.write(json.dumps(to_jsonable(doc), ensure_ascii=False) + "\n")
        return [str(path)]
    paths: List[str] = []
    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(shards)]
    for idx, doc in enumerate(docs):
        buckets[idx % shards].append(doc)
    for shard_idx, bucket in enumerate(buckets):
        path = target / f"{dataset}_reaction_index_shard_{shard_idx:03d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for doc in bucket:
                handle.write(json.dumps(to_jsonable(doc), ensure_ascii=False) + "\n")
        paths.append(str(path))
    return paths


def classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    if not y_true:
        return {"n": 0, "accuracy": None, "macro_f1": None, "weighted_f1": None}
    labels = sorted(set(y_true) | set(y_pred))
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    per_label: Dict[str, Dict[str, float]] = {}
    macro_f1_values = []
    weighted_sum = 0.0
    for label in labels:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
        support = sum(1 for true in y_true if true == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        macro_f1_values.append(f1)
        weighted_sum += f1 * support
    return {
        "n": len(y_true),
        "accuracy": correct / len(y_true),
        "macro_f1": sum(macro_f1_values) / len(macro_f1_values) if macro_f1_values else 0.0,
        "weighted_f1": weighted_sum / len(y_true),
        "per_label": per_label,
    }


def regression_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, Any]:
    if not y_true:
        return {"n": 0, "mae": None, "rmse": None, "r2": None, "pearson": None, "spearman": None}
    errors = [pred - true for true, pred in zip(y_true, y_pred)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    mean_true = sum(y_true) / len(y_true)
    ss_res = sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred))
    ss_tot = sum((true - mean_true) ** 2 for true in y_true)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return {
        "n": len(y_true),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson": _pearson(y_true, y_pred),
        "spearman": _pearson(_ranks(y_true), _ranks(y_pred)),
    }


def _pearson(a: List[float], b: List[float]) -> float:
    if not a or len(a) != len(b):
        return 0.0
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denom_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    denom_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
    return numerator / (denom_a * denom_b) if denom_a and denom_b else 0.0


def _ranks(values: List[float]) -> List[float]:
    order = sorted((value, idx) for idx, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][0] == order[i][0]:
            j += 1
        rank = (i + j + 2) / 2.0
        for _, idx in order[i : j + 1]:
            ranks[idx] = rank
        i = j + 1
    return ranks


def _reaction_tokens(reaction: Mapping[str, Any]) -> set[str]:
    text = " ".join(str(reaction.get(key) or "") for key in ["reaction_smiles", "reactants", "products", "reactant_smiles", "product_smiles", "aux_smiles"])
    tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]{1,}", text))
    atoms = set(re.findall(r"Br|Cl|[BCNOFPSIbcno]", text))
    return tokens | atoms


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _append_reaction_evidence(context: Dict[str, Any], evidence: List[Dict[str, Any]]) -> None:
    context.setdefault("evidence", []).extend(evidence)
    context.setdefault("citation_map", []).extend(
        {
            "claim_id": item.get("citation_id"),
            "claim": item.get("claim"),
            "source": item.get("source"),
            "source_path": item.get("source_path"),
            "citation_exists": True,
            "reaction_specific": True,
        }
        for item in evidence
    )
    context.setdefault("retrieval_trace", []).append(
        {
            "provider": "RetrieveAgent",
            "backend": "local_reaction_data",
            "evidence_items": evidence,
            "warnings": [],
        }
    )


def _values(row: Mapping[str, Any], prefix: str) -> List[str]:
    values = []
    for key, value in row.items():
        if not str(key).startswith(prefix):
            continue
        if value in {None, "", "nan"}:
            continue
        text = str(value)
        if text and text.lower() != "nan":
            values.append(text)
    return values


def _shorten(value: Any, limit: int) -> str:
    text = str(value or "")
    return text[:limit] + ("..." if len(text) > limit else "")
