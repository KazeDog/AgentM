"""Project-local default paths for optional pretrained representations."""

DEFAULT_FEATURE_SET = "all_features"
DEFAULT_IMAGEMOL_CHECKPOINT_PATH = "checkpoints/imagemol/ImageMol.pth.tar"
DEFAULT_CHEMBERTA2_MODEL_PATH = "checkpoints/chemberta2/DeepChem-ChemBERTa-77M-MTR"
DEFAULT_RAG_EMBEDDING_MODEL_PATH = "checkpoints/rag/scibert_scivocab_uncased"
DEFAULT_RAG_EMBEDDING_MODEL_ID = "allenai/scibert_scivocab_uncased"
DEFAULT_RAG_BACKEND = "dense_faiss"
DEFAULT_RAG_CORPUS_DIR = "data/rag_corpus/tdc_real"
DEFAULT_DDI_RAG_CORPUS_DIR = "data/rag_corpus/ddi_real"
DEFAULT_RAG_TOP_K = 5
DEFAULT_LLM_PROFILE = "qwen_flash"
DEFAULT_MLLM_PROFILE = "kimi_k2_6_bailian"
DEFAULT_USE_SMILES_ENCODER = True
DEFAULT_USE_MLLM_VISUAL_SEMANTIC = True
DEFAULT_USE_LLM_PLANNER = True
DEFAULT_USE_LLM_DECISION_LAYER = True
