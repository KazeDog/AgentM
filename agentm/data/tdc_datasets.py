from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd

from agentm.tools.smiles import standardize_smiles
from agentm.utils.io import ensure_dir, write_json


@dataclass(frozen=True)
class TDCTaskSpec:
    name: str
    collection: str
    mode: str
    label_description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


TDC_TASKS: Dict[str, TDCTaskSpec] = {
    # ADME classification
    "BBB_Martins": TDCTaskSpec("BBB_Martins", "ADME", "classification", "blood-brain barrier penetration"),
    "HIA_Hou": TDCTaskSpec("HIA_Hou", "ADME", "classification", "human intestinal absorption"),
    "Pgp_Broccatelli": TDCTaskSpec("Pgp_Broccatelli", "ADME", "classification", "P-glycoprotein inhibition"),
    "Bioavailability_Ma": TDCTaskSpec("Bioavailability_Ma", "ADME", "classification", "oral bioavailability"),
    "CYP1A2_Veith": TDCTaskSpec("CYP1A2_Veith", "ADME", "classification", "CYP1A2 inhibition"),
    "CYP2C9_Veith": TDCTaskSpec("CYP2C9_Veith", "ADME", "classification", "CYP2C9 inhibition"),
    "CYP2C19_Veith": TDCTaskSpec("CYP2C19_Veith", "ADME", "classification", "CYP2C19 inhibition"),
    "CYP2D6_Veith": TDCTaskSpec("CYP2D6_Veith", "ADME", "classification", "CYP2D6 inhibition"),
    "CYP3A4_Veith": TDCTaskSpec("CYP3A4_Veith", "ADME", "classification", "CYP3A4 inhibition"),
    "CYP2C9_Substrate_CarbonMangels": TDCTaskSpec(
        "CYP2C9_Substrate_CarbonMangels",
        "ADME",
        "classification",
        "CYP2C9 substrate activity",
    ),
    "CYP2D6_Substrate_CarbonMangels": TDCTaskSpec(
        "CYP2D6_Substrate_CarbonMangels",
        "ADME",
        "classification",
        "CYP2D6 substrate activity",
    ),
    "CYP3A4_Substrate_CarbonMangels": TDCTaskSpec(
        "CYP3A4_Substrate_CarbonMangels",
        "ADME",
        "classification",
        "CYP3A4 substrate activity",
    ),
    # ADME regression
    "Caco2_Wang": TDCTaskSpec("Caco2_Wang", "ADME", "regression", "Caco-2 permeability"),
    "Lipophilicity_AstraZeneca": TDCTaskSpec("Lipophilicity_AstraZeneca", "ADME", "regression", "lipophilicity"),
    "Solubility_AqSolDB": TDCTaskSpec("Solubility_AqSolDB", "ADME", "regression", "aqueous solubility"),
    "PPBR_AZ": TDCTaskSpec("PPBR_AZ", "ADME", "regression", "plasma protein binding rate"),
    "VDss_Lombardo": TDCTaskSpec("VDss_Lombardo", "ADME", "regression", "steady-state volume of distribution"),
    "HydrationFreeEnergy_FreeSolv": TDCTaskSpec("HydrationFreeEnergy_FreeSolv", "ADME", "regression", "hydration free energy"),
    "Half_Life_Obach": TDCTaskSpec("Half_Life_Obach", "ADME", "regression", "drug half-life"),
    "Clearance_Hepatocyte_AZ": TDCTaskSpec("Clearance_Hepatocyte_AZ", "ADME", "regression", "hepatocyte clearance"),
    "Clearance_Microsome_AZ": TDCTaskSpec("Clearance_Microsome_AZ", "ADME", "regression", "microsome clearance"),
    # Toxicity tasks commonly used as ADMET endpoints
    "hERG": TDCTaskSpec("hERG", "Tox", "classification", "hERG blockade"),
    "AMES": TDCTaskSpec("AMES", "Tox", "classification", "Ames mutagenicity"),
    "DILI": TDCTaskSpec("DILI", "Tox", "classification", "drug-induced liver injury"),
    "ClinTox": TDCTaskSpec("ClinTox", "Tox", "classification", "clinical toxicity"),
    "LD50_Zhu": TDCTaskSpec("LD50_Zhu", "Tox", "regression", "acute toxicity LD50"),
}

RECOMMENDED_TDC_ADMET_TASKS = [
    "BBB_Martins",
    "HIA_Hou",
    "Bioavailability_Ma",
    "CYP3A4_Veith",
    "CYP2D6_Veith",
    "Lipophilicity_AstraZeneca",
    "Solubility_AqSolDB",
    "hERG",
    "AMES",
    "DILI",
]


def task_spec(name: str) -> TDCTaskSpec:
    if name not in TDC_TASKS:
        known = ", ".join(sorted(TDC_TASKS))
        raise KeyError(f"Unknown TDC task '{name}'. Known tasks: {known}")
    return TDC_TASKS[name]


def normalize_tdc_dataframe(
    df: pd.DataFrame,
    spec: TDCTaskSpec,
    *,
    standardize: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    smiles_col = _first_present(df, ["Drug", "SMILES", "smiles", "drug", "Smiles"])
    label_col = _first_present(df, ["Y", "y", "Label", "label", "target"])
    id_col = _first_present(df, ["Drug_ID", "drug_id", "DrugID", "Name", "name", "mol_id"], required=False)
    rows: List[Dict[str, Any]] = []
    invalid_rows: List[Dict[str, Any]] = []
    for row_idx, row in df.iterrows():
        raw_smiles = str(row.get(smiles_col, "")).strip()
        label = row.get(label_col)
        if not raw_smiles or pd.isna(label):
            invalid_rows.append({"row_index": row_idx, "smiles": raw_smiles, "error": "missing_smiles_or_label"})
            continue
        smiles = raw_smiles
        if standardize:
            canonical, err = standardize_smiles(raw_smiles)
            if err or not canonical:
                invalid_rows.append({"row_index": row_idx, "smiles": raw_smiles, "error": err or "invalid_smiles"})
                continue
            smiles = canonical
        item = {
            "smiles": smiles,
            "label": label,
            "tdc_task": spec.name,
            "tdc_collection": spec.collection,
            "task_mode": spec.mode,
        }
        if id_col:
            item["drug_id"] = row.get(id_col)
        rows.append(item)
    clean = pd.DataFrame(rows)
    if not clean.empty:
        try:
            clean["label"] = pd.to_numeric(clean["label"])
        except Exception:
            pass
    invalid = pd.DataFrame(invalid_rows)
    return clean, invalid


def write_tdc_training_config(
    path: str | Path,
    *,
    spec: TDCTaskSpec,
    data_path: str | Path,
    split_type: str,
    output_dir: str | Path,
    feature_set: str = "all_features",
    model_type: str = "random_forest",
    random_seed: int = 13,
    imagemol_checkpoint_path: str = "checkpoints/imagemol/ImageMol.pth.tar",
    smiles_encoder_model_name: str = "checkpoints/chemberta2/DeepChem-ChemBERTa-77M-MTR",
    use_smiles_encoder: bool = True,
    use_mllm_visual_semantic: bool = False,
    llm_profile: str = "qwen_flash",
    mllm_profile: str = "kimi_k2_6_bailian",
) -> Path:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write TDC YAML configs.") from exc
    config = {
        "task_type": "admet_prediction",
        "dataset_name": spec.name,
        "data_path": str(data_path),
        "split_type": split_type,
        "feature_set": feature_set,
        "model_type": model_type,
        "use_llm_rules": True,
        "use_imagemol": True,
        "use_smiles_encoder": bool(use_smiles_encoder),
        "use_visual_semantic": True,
        "use_evidence_retrieval": True,
        "imagemol_checkpoint_path": imagemol_checkpoint_path,
        "smiles_encoder_model_name": smiles_encoder_model_name,
        "smiles_encoder_device": "auto",
        "smiles_pooling": "cls",
        "smiles_local_files_only": True,
        "llm_profile": llm_profile,
        "use_mllm_visual_semantic": bool(use_mllm_visual_semantic),
        "mllm_profile": mllm_profile,
        "output_dir": str(output_dir),
        "random_seed": int(random_seed),
        "smiles_column": "smiles",
        "label_column": "label",
        "tdc_collection": spec.collection,
        "tdc_task_mode": spec.mode,
        "tdc_label_description": spec.label_description,
    }
    if spec.mode == "classification":
        config.update(
            {
                "positive_label": 1,
                "negative_label": 0,
                "positive_label_meaning": spec.label_description or "positive class",
                "negative_label_meaning": f"not {spec.label_description}" if spec.label_description else "negative class",
                "decision_threshold": 0.5,
            }
        )
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return target


def write_dataset_metadata(path: str | Path, *, spec: TDCTaskSpec, counts: Mapping[str, int], files: Mapping[str, str]) -> Path:
    return write_json(
        path,
        {
            "task": spec.to_dict(),
            "counts": dict(counts),
            "files": dict(files),
        },
    )


def expand_task_names(values: Iterable[str] | None) -> List[str]:
    if not values:
        return list(RECOMMENDED_TDC_ADMET_TASKS)
    names: List[str] = []
    for value in values:
        for item in str(value).split(","):
            name = item.strip()
            if not name:
                continue
            if name == "recommended":
                names.extend(RECOMMENDED_TDC_ADMET_TASKS)
            elif name == "all":
                names.extend(TDC_TASKS)
            else:
                names.append(name)
    deduped: List[str] = []
    seen = set()
    for name in names:
        if name not in seen:
            task_spec(name)
            deduped.append(name)
            seen.add(name)
    return deduped


def _first_present(df: pd.DataFrame, names: List[str], *, required: bool = True) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    if required:
        raise ValueError(f"Could not find any of columns {names}; available columns: {list(df.columns)}")
    return None
