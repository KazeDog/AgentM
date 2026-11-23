from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from rdkit import Chem

from agentm.features.imagemol import ImageMolEncoder, load_imagemol_encoder
from agentm.features.smiles import DEFAULT_CHEMBERTA2_MODEL, SmilesEncoder, load_smiles_encoder
from agentm.features.visual_semantic import VisualSemanticExtractor
from agentm.rules.executor import RuleExecutor
from agentm.schema import (
    DDIPairFeatureBundle,
    FeatureSchema,
    ImageFeatureBundle,
    MoleculeRecord,
    Rule,
    RuleFeatureVector,
    SingleMoleculeFeatureBundle,
    SmilesFeatureBundle,
    VisualSemanticBundle,
)
from agentm.tools.descriptors import (
    DESCRIPTOR_NAMES,
    compute_rdkit_descriptors,
    descriptor_vector,
    maccs_fingerprint,
    morgan_fingerprint,
    tanimoto_from_bits,
)
from agentm.tools.imaging import generate_molecule_image
from agentm.tools.smiles import mol_from_smiles
from agentm.utils.io import ensure_dir, write_json


FEATURE_SETS = {
    "rdkit_only",
    "fingerprint_only",
    "llm_rules_only",
    "smiles_only",
    "imagemol_only",
    "visual_semantic_only",
    "rdkit_plus_llm_rules",
    "rdkit_plus_smiles",
    "llm_rules_plus_smiles",
    "llm_rules_plus_imagemol",
    "smiles_plus_imagemol",
    "smiles_plus_imagemol_plus_visual_semantic",
    "llm_rules_plus_imagemol_plus_visual_semantic",
    "llm_rules_plus_smiles_plus_imagemol_plus_visual_semantic",
    "all_features",
}

SMILES_FEATURE_SETS = {
    "smiles_only",
    "rdkit_plus_smiles",
    "llm_rules_plus_smiles",
    "smiles_plus_imagemol",
    "smiles_plus_imagemol_plus_visual_semantic",
    "llm_rules_plus_smiles_plus_imagemol_plus_visual_semantic",
}

RULE_FEATURE_SETS = {
    "llm_rules_only",
    "rdkit_plus_llm_rules",
    "llm_rules_plus_smiles",
    "llm_rules_plus_imagemol",
    "llm_rules_plus_imagemol_plus_visual_semantic",
    "llm_rules_plus_smiles_plus_imagemol_plus_visual_semantic",
    "all_features",
}

IMAGEMOL_FEATURE_SETS = {
    "imagemol_only",
    "llm_rules_plus_imagemol",
    "smiles_plus_imagemol",
    "smiles_plus_imagemol_plus_visual_semantic",
    "llm_rules_plus_imagemol_plus_visual_semantic",
    "llm_rules_plus_smiles_plus_imagemol_plus_visual_semantic",
    "all_features",
}

VISUAL_FEATURE_SETS = {
    "visual_semantic_only",
    "smiles_plus_imagemol_plus_visual_semantic",
    "llm_rules_plus_imagemol_plus_visual_semantic",
    "llm_rules_plus_smiles_plus_imagemol_plus_visual_semantic",
    "all_features",
}


def _safe_feature_stem(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text)
    return cleaned[:80] or "molecule"


class FeatureBuilder:
    def __init__(
        self,
        output_dir: str | Path = "outputs/features",
        feature_set: str = "all_features",
        imagemol_encoder: Optional[ImageMolEncoder] = None,
        imagemol_checkpoint_path: Optional[str] = None,
        smiles_encoder: Optional[SmilesEncoder] = None,
        use_smiles_encoder: bool = False,
        smiles_encoder_model_name: Optional[str] = None,
        smiles_encoder_device: str = "auto",
        smiles_pooling: str = "cls",
        smiles_local_files_only: bool = True,
        visual_extractor: Optional[VisualSemanticExtractor] = None,
        use_mllm_visual_semantic: bool = False,
        mllm_profile: Optional[str] = None,
        visual_semantic_cache_dir: Optional[str] = None,
        visual_semantic_cache_read: bool = True,
        visual_semantic_cache_write: bool = True,
        visual_semantic_cache_only: bool = False,
        generate_grad_cam: bool = False,
    ):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature_set: {feature_set}")
        self.output_dir = ensure_dir(output_dir)
        self.image_dir = ensure_dir(self.output_dir / "images")
        self.feature_set = feature_set
        self.rule_executor = RuleExecutor()
        needs_imagemol_encoder = feature_set in IMAGEMOL_FEATURE_SETS
        self.imagemol_encoder = imagemol_encoder or (
            load_imagemol_encoder(imagemol_checkpoint_path) if needs_imagemol_encoder else None
        )
        self.imagemol_checkpoint_path = imagemol_checkpoint_path
        self.use_smiles_encoder = bool(use_smiles_encoder or feature_set in SMILES_FEATURE_SETS)
        self.smiles_encoder = smiles_encoder or load_smiles_encoder(
            model_name=smiles_encoder_model_name or DEFAULT_CHEMBERTA2_MODEL,
            use_smiles_encoder=self.use_smiles_encoder,
            device=smiles_encoder_device,
            pooling=smiles_pooling,
            local_files_only=smiles_local_files_only,
        )
        self.smiles_encoder_model_name = smiles_encoder_model_name or DEFAULT_CHEMBERTA2_MODEL
        self.smiles_encoder_device = smiles_encoder_device
        self.smiles_pooling = smiles_pooling
        self.smiles_local_files_only = smiles_local_files_only
        self.visual_extractor = visual_extractor or VisualSemanticExtractor(
            use_mllm=use_mllm_visual_semantic,
            mllm_profile=mllm_profile,
            cache_dir=visual_semantic_cache_dir,
            cache_read=visual_semantic_cache_read,
            cache_write=visual_semantic_cache_write,
            cache_only=visual_semantic_cache_only,
        )
        self.visual_semantic_cache_dir = visual_semantic_cache_dir
        self.visual_semantic_cache_read = visual_semantic_cache_read
        self.visual_semantic_cache_write = visual_semantic_cache_write
        self.visual_semantic_cache_only = visual_semantic_cache_only
        self.generate_grad_cam = generate_grad_cam
        self._last_ddi_schema_signature: Optional[tuple[Any, ...]] = None
        self._last_ddi_schema: Optional[FeatureSchema] = None

    def build_single(
        self,
        record: MoleculeRecord,
        rules: Iterable[Rule] = (),
        task_type: str = "property_prediction",
        dataset_name: Optional[str] = None,
    ) -> SingleMoleculeFeatureBundle:
        canonical = record.canonical_smiles or record.smiles
        mol = mol_from_smiles(canonical or "")
        if mol is None:
            raise ValueError(f"Cannot build features for invalid molecule: {record.original_input}")

        descriptors = compute_rdkit_descriptors(mol)
        morgan = morgan_fingerprint(mol)
        maccs = maccs_fingerprint(mol)
        needs_rules = self.feature_set in RULE_FEATURE_SETS
        needs_smiles = self.use_smiles_encoder and (self.feature_set in SMILES_FEATURE_SETS or self.feature_set == "all_features")
        needs_imagemol = self.feature_set in IMAGEMOL_FEATURE_SETS
        needs_visual = self.feature_set in VISUAL_FEATURE_SETS
        needs_image = needs_imagemol or (needs_visual and bool(getattr(self.visual_extractor, "use_mllm", False)))

        smiles_bundle = self._encode_smiles(canonical or "") if needs_smiles else None
        smiles_embedding = smiles_bundle.embedding if smiles_bundle else []
        image_path = generate_molecule_image(mol, canonical or "molecule", self.image_dir) if needs_image else ""
        image_embedding = self.imagemol_encoder.encode(image_path, metadata=descriptors) if needs_imagemol and self.imagemol_encoder else []
        grad_cam_path = self.imagemol_encoder.grad_cam(image_path) if needs_imagemol and self.generate_grad_cam and self.imagemol_encoder else None
        image_bundle = ImageFeatureBundle(
            image_path=image_path,
            embedding=image_embedding,
            encoder_name=self.imagemol_encoder.encoder_name if self.imagemol_encoder else "disabled",
            checkpoint_path=self.imagemol_encoder.checkpoint_path if self.imagemol_encoder else None,
            grad_cam_path=grad_cam_path,
            grad_cam=None,
        )
        visual = (
            self.visual_extractor.extract(
                mol,
                descriptors,
                task_type=task_type,
                image_path=image_path or None,
                smiles=canonical,
                dataset_name=dataset_name,
            )
            if needs_visual
            else VisualSemanticBundle(provider="disabled", metadata={"reason": f"feature_set={self.feature_set}"})
        )
        rule_features = (
            self.rule_executor.execute(mol, descriptors, self.rule_executor.filter_executable_unique(rules))
            if needs_rules
            else RuleFeatureVector()
        )

        feature_names, vector, blocks = self._single_vector(
            descriptors=descriptors,
            morgan=morgan,
            maccs=maccs,
            rule_features=rule_features,
            smiles_embedding=smiles_embedding,
            image_embedding=image_embedding,
            visual_embedding=visual.embedding,
        )
        schema = FeatureSchema(
            feature_set=self.feature_set,
            feature_names=feature_names,
            blocks=blocks,
            metadata=self._single_schema_metadata(
                descriptors=descriptors,
                morgan=morgan,
                maccs=maccs,
                rule_features=rule_features,
                smiles_bundle=smiles_bundle,
                image_bundle=image_bundle,
                visual=visual,
                task_type=task_type,
                dataset_name=dataset_name,
                total_dim=len(vector),
            ),
        )
        bundle = SingleMoleculeFeatureBundle(
            molecule=record,
            rdkit_descriptors=descriptors,
            morgan_fingerprint=morgan,
            maccs_fingerprint=maccs,
            llm_rule_features=rule_features,
            smiles_embedding=smiles_embedding,
            imagemol_embedding=image_embedding,
            visual_semantic_embedding=visual.embedding,
            visual_semantic_text=visual.to_text(),
            feature_vector=vector,
            feature_schema=schema,
            smiles=smiles_bundle,
            image=image_bundle,
            visual_semantics=visual,
        )
        write_json(self.output_dir / "feature_schema.json", schema.to_dict())
        if needs_visual:
            stem = Path(image_path).stem if image_path else _safe_feature_stem(canonical or "molecule")
            write_json(self.output_dir / "visual_semantics" / f"{stem}.json", visual.to_dict())
        return bundle

    def build_ddi_pair(
        self,
        drug_a: SingleMoleculeFeatureBundle,
        drug_b: SingleMoleculeFeatureBundle,
    ) -> DDIPairFeatureBundle:
        components = self._ddi_pair_components(drug_a, drug_b)
        schema = self._cached_ddi_schema(
            drug_a.feature_schema,
            drug_b.feature_schema,
            components["similarities"],
            components["admet"],
            components["cyp"],
            components["toxicity"],
            components["mechanism"],
        )
        pair_vector = self._ddi_pair_vector_from_components(components)
        return DDIPairFeatureBundle(
            drug_a_features=drug_a,
            drug_b_features=drug_b,
            concat_features=components["concat"].astype(float).tolist(),
            absolute_difference=components["abs_diff"].astype(float).tolist(),
            elementwise_product=components["product"].astype(float).tolist(),
            fingerprint_similarity=components["similarities"],
            admet_risk_features=components["admet"],
            cyp_related_features=components["cyp"],
            toxicity_overlap_features=components["toxicity"],
            llm_ddi_mechanism_features=components["mechanism"],
            feature_vector=pair_vector.astype(float).tolist(),
            feature_schema=schema,
        )

    def build_ddi_pair_vector(
        self,
        drug_a: SingleMoleculeFeatureBundle,
        drug_b: SingleMoleculeFeatureBundle,
    ) -> tuple[np.ndarray, FeatureSchema]:
        components = self._ddi_pair_components(drug_a, drug_b)
        schema = self._cached_ddi_schema(
            drug_a.feature_schema,
            drug_b.feature_schema,
            components["similarities"],
            components["admet"],
            components["cyp"],
            components["toxicity"],
            components["mechanism"],
        )
        return self._ddi_pair_vector_from_components(components), schema

    def _ddi_pair_components(
        self,
        drug_a: SingleMoleculeFeatureBundle,
        drug_b: SingleMoleculeFeatureBundle,
    ) -> Dict[str, Any]:
        vec_a = np.asarray(drug_a.feature_vector, dtype=np.float32)
        vec_b = np.asarray(drug_b.feature_vector, dtype=np.float32)
        if vec_a.shape != vec_b.shape:
            raise ValueError("DDI pair features require matching single-molecule schema dimensions.")
        similarities = {
            "morgan_tanimoto": tanimoto_from_bits(drug_a.morgan_fingerprint, drug_b.morgan_fingerprint),
            "maccs_tanimoto": tanimoto_from_bits(drug_a.maccs_fingerprint, drug_b.maccs_fingerprint),
        }
        return {
            "concat": np.concatenate([vec_a, vec_b]),
            "abs_diff": np.abs(vec_a - vec_b),
            "product": vec_a * vec_b,
            "similarities": similarities,
            "admet": self._admet_risk_pair(drug_a, drug_b),
            "cyp": self._cyp_pair(drug_a, drug_b),
            "toxicity": self._toxicity_overlap(drug_a, drug_b),
            "mechanism": self._mechanism_features(drug_a, drug_b),
        }

    def _ddi_pair_vector_from_components(self, components: Dict[str, Any]) -> np.ndarray:
        return np.concatenate(
            [
                components["concat"],
                components["abs_diff"],
                components["product"],
                np.asarray(list(components["similarities"].values()), dtype=np.float32),
                np.asarray(list(components["admet"].values()), dtype=np.float32),
                np.asarray(list(components["cyp"].values()), dtype=np.float32),
                np.asarray(list(components["toxicity"].values()), dtype=np.float32),
                np.asarray(list(components["mechanism"].values()), dtype=np.float32),
            ]
        ).astype(np.float32, copy=False)

    def _cached_ddi_schema(
        self,
        schema_a: FeatureSchema,
        schema_b: FeatureSchema,
        similarities: Dict[str, float],
        admet: Dict[str, float],
        cyp: Dict[str, float],
        toxicity: Dict[str, float],
        mechanism: Dict[str, float],
    ) -> FeatureSchema:
        signature = (
            self.feature_set,
            tuple(schema_a.feature_names),
            tuple(schema_b.feature_names),
            tuple(similarities.keys()),
            tuple(admet.keys()),
            tuple(cyp.keys()),
            tuple(toxicity.keys()),
            tuple(mechanism.keys()),
        )
        if self._last_ddi_schema_signature == signature and self._last_ddi_schema is not None:
            return self._last_ddi_schema
        names = (
            [f"a_{n}" for n in schema_a.feature_names]
            + [f"b_{n}" for n in schema_b.feature_names]
            + [f"absdiff_{n}" for n in schema_a.feature_names]
            + [f"product_{n}" for n in schema_a.feature_names]
            + list(similarities.keys())
            + list(admet.keys())
            + list(cyp.keys())
            + list(toxicity.keys())
            + list(mechanism.keys())
        )
        schema = FeatureSchema(
            feature_set=f"ddi_{self.feature_set}",
            feature_names=names,
            blocks=self._ddi_blocks(schema_a, similarities, admet, cyp, toxicity, mechanism),
            metadata=self._ddi_schema_metadata(schema_a, schema_b),
        )
        write_json(self.output_dir / "ddi_feature_schema.json", schema.to_dict())
        self._last_ddi_schema_signature = signature
        self._last_ddi_schema = schema
        return schema

    def _single_vector(
        self,
        descriptors: Dict[str, float],
        morgan: List[int],
        maccs: List[int],
        rule_features: RuleFeatureVector,
        smiles_embedding: List[float],
        image_embedding: List[float],
        visual_embedding: List[float],
    ) -> tuple[List[str], List[float], Dict[str, Dict[str, Any]]]:
        selected_blocks: List[tuple[str, List[str], List[float]]] = []
        rdkit_block = ("rdkit", [f"rdkit_{name}" for name in DESCRIPTOR_NAMES], descriptor_vector(descriptors))
        fp_block = (
            "fingerprint",
            [f"morgan_{i}" for i in range(len(morgan))] + [f"maccs_{i}" for i in range(len(maccs))],
            [float(v) for v in morgan + maccs],
        )
        rule_names = sorted(rule_features.values)
        rule_block = ("llm_rules", [f"llm_rule_{name}" for name in rule_names], [float(rule_features.values[name]) for name in rule_names])
        smiles_block = ("smiles", [f"smiles_{i}" for i in range(len(smiles_embedding))], [float(v) for v in smiles_embedding])
        image_block = ("imagemol", [f"imagemol_{i}" for i in range(len(image_embedding))], [float(v) for v in image_embedding])
        visual_block = ("visual_semantic", [f"visual_semantic_{i}" for i in range(len(visual_embedding))], [float(v) for v in visual_embedding])

        if self.feature_set == "rdkit_only":
            selected_blocks.append(rdkit_block)
        elif self.feature_set == "fingerprint_only":
            selected_blocks.append(fp_block)
        elif self.feature_set == "llm_rules_only":
            selected_blocks.append(rule_block)
        elif self.feature_set == "smiles_only":
            selected_blocks.append(smiles_block)
        elif self.feature_set == "imagemol_only":
            selected_blocks.append(image_block)
        elif self.feature_set == "visual_semantic_only":
            selected_blocks.append(visual_block)
        elif self.feature_set == "rdkit_plus_llm_rules":
            selected_blocks.extend([rdkit_block, rule_block])
        elif self.feature_set == "rdkit_plus_smiles":
            selected_blocks.extend([rdkit_block, smiles_block])
        elif self.feature_set == "llm_rules_plus_smiles":
            selected_blocks.extend([rule_block, smiles_block])
        elif self.feature_set == "llm_rules_plus_imagemol":
            selected_blocks.extend([rule_block, image_block])
        elif self.feature_set == "smiles_plus_imagemol":
            selected_blocks.extend([smiles_block, image_block])
        elif self.feature_set == "smiles_plus_imagemol_plus_visual_semantic":
            selected_blocks.extend([smiles_block, image_block, visual_block])
        elif self.feature_set == "llm_rules_plus_imagemol_plus_visual_semantic":
            selected_blocks.extend([rule_block, image_block, visual_block])
        elif self.feature_set == "llm_rules_plus_smiles_plus_imagemol_plus_visual_semantic":
            selected_blocks.extend([rule_block, smiles_block, image_block, visual_block])
        elif self.feature_set == "all_features":
            selected_blocks.extend([rdkit_block, fp_block, rule_block])
            if self.use_smiles_encoder:
                selected_blocks.append(smiles_block)
            selected_blocks.extend([image_block, visual_block])

        names: List[str] = []
        values: List[float] = []
        block_ranges: Dict[str, Dict[str, Any]] = {}
        cursor = 0
        for block_name, block_names, block_values in selected_blocks:
            start = cursor
            names.extend(block_names)
            values.extend(block_values)
            cursor += len(block_values)
            block_ranges[block_name] = {
                "start": start,
                "end": cursor,
                "dim": len(block_values),
            }
        return names, values, block_ranges

    def _single_schema_metadata(
        self,
        *,
        descriptors: Dict[str, float],
        morgan: List[int],
        maccs: List[int],
        rule_features: RuleFeatureVector,
        smiles_bundle: Optional[SmilesFeatureBundle],
        image_bundle: ImageFeatureBundle,
        visual,
        task_type: str,
        dataset_name: Optional[str],
        total_dim: int,
    ) -> Dict[str, Any]:
        return {
            "task_type": task_type,
            "dataset_name": dataset_name,
            "total_dim": total_dim,
            "block_sources": {
                "rdkit": {
                    "source": "rdkit_descriptors",
                    "descriptor_names": list(DESCRIPTOR_NAMES),
                    "dim": len(DESCRIPTOR_NAMES),
                },
                "fingerprint": {
                    "source": "rdkit_fingerprints",
                    "morgan_dim": len(morgan),
                    "maccs_dim": len(maccs),
                    "dim": len(morgan) + len(maccs),
                },
                "llm_rules": {
                    "source": "compiled_rules",
                    "n_features": len(rule_features.values),
                    "n_executed_rules": len(rule_features.executed_rules),
                    "n_failed_rules": len(rule_features.failed_rules),
                    "feature_names": sorted(rule_features.values),
                },
                "smiles": {
                    "source": "smiles_pretrained_encoder",
                    "enabled": smiles_bundle is not None,
                    "encoder": smiles_bundle.encoder_name if smiles_bundle else None,
                    "model_name": smiles_bundle.model_name if smiles_bundle else self.smiles_encoder_model_name,
                    "pooling": smiles_bundle.pooling if smiles_bundle else self.smiles_pooling,
                    "dim": len(smiles_bundle.embedding) if smiles_bundle else 0,
                    "metadata": smiles_bundle.metadata if smiles_bundle else {},
                },
                "imagemol": {
                    "source": "imagemol_encoder",
                    "encoder": image_bundle.encoder_name,
                    "checkpoint_path": image_bundle.checkpoint_path or self.imagemol_checkpoint_path,
                    "dim": len(image_bundle.embedding),
                    "grad_cam_available": bool(image_bundle.grad_cam_path),
                },
                "visual_semantic": {
                    "source": visual.provider,
                    "provider": visual.provider,
                    "dim": len(visual.embedding),
                    "mllm_profile": visual.metadata.get("mllm_profile") if visual.metadata else None,
                    "mllm_model": visual.metadata.get("mllm_model") if visual.metadata else None,
                    "cache_path": visual.metadata.get("cache_path") if visual.metadata else None,
                    "cache_hit": visual.metadata.get("cache_hit") if visual.metadata else None,
                    "cache_only": visual.metadata.get("cache_only") if visual.metadata else None,
                    "mllm_cache_miss": visual.metadata.get("mllm_cache_miss") if visual.metadata else None,
                    "fallback_error": visual.metadata.get("mllm_error") if visual.metadata else None,
                },
            },
            "descriptor_snapshot": {
                key: descriptors.get(key)
                for key in ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "QED"]
            },
        }

    def _encode_smiles(self, smiles: str) -> Optional[SmilesFeatureBundle]:
        if not self.use_smiles_encoder or self.smiles_encoder is None:
            return None
        embedding = self.smiles_encoder.encode(smiles)
        return SmilesFeatureBundle(
            smiles=smiles,
            embedding=embedding,
            encoder_name=self.smiles_encoder.encoder_name,
            model_name=self.smiles_encoder.model_name,
            pooling=self.smiles_encoder.pooling,
            metadata=dict(self.smiles_encoder.metadata),
        )

    def _ddi_blocks(
        self,
        single_schema: FeatureSchema,
        similarities: Dict[str, float],
        admet: Dict[str, float],
        cyp: Dict[str, float],
        toxicity: Dict[str, float],
        mechanism: Dict[str, float],
    ) -> Dict[str, Dict[str, Any]]:
        single_dim = len(single_schema.feature_names)
        blocks: Dict[str, Dict[str, Any]] = {
            "drug_a": {"start": 0, "end": single_dim, "dim": single_dim},
            "drug_b": {"start": single_dim, "end": single_dim * 2, "dim": single_dim},
            "absolute_difference": {"start": single_dim * 2, "end": single_dim * 3, "dim": single_dim},
            "elementwise_product": {"start": single_dim * 3, "end": single_dim * 4, "dim": single_dim},
        }
        cursor = single_dim * 4
        for name, values in [
            ("fingerprint_similarity", similarities),
            ("admet_risk_features", admet),
            ("cyp_related_features", cyp),
            ("toxicity_overlap_features", toxicity),
            ("llm_ddi_mechanism_features", mechanism),
        ]:
            start = cursor
            cursor += len(values)
            blocks[name] = {"start": start, "end": cursor, "dim": len(values)}
        return blocks

    def _ddi_schema_metadata(self, schema_a: FeatureSchema, schema_b: FeatureSchema) -> Dict[str, Any]:
        return {
            "source": "ddi_pair_fusion",
            "single_feature_set": self.feature_set,
            "drug_a_single_schema": schema_a.to_dict(),
            "drug_b_single_schema": schema_b.to_dict(),
        }

    def _admet_risk_pair(self, a: SingleMoleculeFeatureBundle, b: SingleMoleculeFeatureBundle) -> Dict[str, float]:
        alerts_a = set(a.visual_semantics.admet_alerts if a.visual_semantics else [])
        alerts_b = set(b.visual_semantics.admet_alerts if b.visual_semantics else [])
        overlap = alerts_a & alerts_b
        return {
            "admet_alert_overlap_count": float(len(overlap)),
            "both_high_logp": float("high_logp" in alerts_a and "high_logp" in alerts_b),
            "either_high_tpsa": float("high_tpsa" in alerts_a or "high_tpsa" in alerts_b),
        }

    def _cyp_pair(self, a: SingleMoleculeFeatureBundle, b: SingleMoleculeFeatureBundle) -> Dict[str, float]:
        desc_a = a.rdkit_descriptors
        desc_b = b.rdkit_descriptors
        return {
            "both_cyp_lipophilic": float(desc_a.get("MolLogP", 0.0) > 4 and desc_b.get("MolLogP", 0.0) > 4),
            "aromatic_ring_total": float(desc_a.get("NumAromaticRings", 0.0) + desc_b.get("NumAromaticRings", 0.0)),
            "amine_pair": float(
                a.visual_semantics and b.visual_semantics and "amine" in a.visual_semantics.functional_groups and "amine" in b.visual_semantics.functional_groups
            ),
        }

    def _toxicity_overlap(self, a: SingleMoleculeFeatureBundle, b: SingleMoleculeFeatureBundle) -> Dict[str, float]:
        groups_a = set(a.visual_semantics.functional_groups if a.visual_semantics else [])
        groups_b = set(b.visual_semantics.functional_groups if b.visual_semantics else [])
        return {
            "shared_toxicophore_count": float(len((groups_a & groups_b) & {"nitro", "halogen"})),
            "either_nitro": float("nitro" in groups_a or "nitro" in groups_b),
        }

    def _mechanism_features(self, a: SingleMoleculeFeatureBundle, b: SingleMoleculeFeatureBundle) -> Dict[str, float]:
        text = " ".join([a.visual_semantic_text, b.visual_semantic_text]).lower()
        return {
            "llm_mechanism_cyp": float("cyp" in text or "lipophilicity" in text),
            "llm_mechanism_transporter": float("transporter" in text),
            "llm_mechanism_bleeding": 0.0,
        }
