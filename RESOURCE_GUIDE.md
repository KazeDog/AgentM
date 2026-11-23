# Resource preparation

Paths below are relative to the repository root. Obtain external assets through authorized sources and record their hashes. No downloads are performed by the offline checks.

| Resource | Location or configuration |
| --- | --- |
| ChemBERTa-2 | `checkpoints/chemberta2/DeepChem-ChemBERTa-77M-MTR` |
| ImageMol | `checkpoints/imagemol/ImageMol.pth.tar` |
| SciBERT retrieval encoder | `checkpoints/rag/scibert_scivocab_uncased` |
| MoleculeNet files | `data/moleculenet/raw` |
| TDC processed files | `data/tdc/<dataset>/<dataset>.csv` |
| Trained endpoint models | Explicit `--model-path` or `--model-registry` |
| ADMET/DDI evidence corpora | `data/rag_corpus/tdc_real`, `data/rag_corpus/ddi_real` |
| Elasticsearch | `AGENTM_ES_URL`, `AGENTM_ES_INDEX` |
| Reaction source data | `data/reaction_sources` or `AGENTM_REACTION_DATA_ROOT` |
| Reaction models and index | `outputs/reaction_models_full`, `data/reaction_indices` |

Obtain these external assets separately and verify their content hashes alongside filenames.

## Data and representations

MoleculeNet target definitions and download URLs are in `agentm/runtime/moleculenet.py`. TDC preparation scripts are `scripts/download_tdc_admet.py` and `scripts/download_tdc_ddi.py`; their optional dataset dependencies are listed separately. Preserve target definitions, units and split membership when preparing data.

Obtain ChemBERTa-2, SciBERT and ImageMol through their authorized upstream sources. The retrieval download helper is `scripts/download_rag_embedding_model.py`. Model and dependency identifiers retain their necessary technical specifications.

A model registry can map endpoint keys to trusted model paths:

```json
{
  "admet_prediction:HIA_Hou:all_features": "/your/models/HIA_Hou/model.pkl",
  "ddi_prediction:DDI:all_features": "/your/models/ddi/model.pkl"
}
```

Match the trained endpoint, feature schema and dataset. Load pickle or checkpoint files only from trusted sources. In strict mode, missing external assets halt execution and record the unmet dependency explicitly.

## Evidence and reactions

Use the evidence-building and indexing scripts in `scripts/` to prepare a corpus and index. Creating an index changes its identity. Real-RAG acceptance checks retrieval provenance and source-body support; synthetic query headers alone do not qualify.

Reaction workflows expect MIT, ORDerly, Schneider reaction classification and Buchwald-Hartwig yield resources under the configured reaction root. `agentm/runtime/reaction.py` lists the required filenames. Provide an explicit root when the resources live elsewhere.

## Before a paid experiment

Freeze the source, effective configuration, case manifest, split assignments, trained resources and retrieval index. Confirm credentials, model identity, reflection settings and finite call/token caps. Use separately authorized small checks before advancing in bounded stages. Preserve failed attempts and replacements as separate records.
