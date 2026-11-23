# Operational notes and external requirements

The default offline test profile runs without external downloads or credentials. Integration workflows that use trained ADMET/DDI models, populated evidence corpora, Elasticsearch or reaction datasets run separately once those assets are configured as described in [RESOURCE_GUIDE.md](RESOURCE_GUIDE.md).

The required-LLM terminal-stop path is covered by offline regression tests: when an online LLM is required and fallback is disabled, LLM unavailability halts execution terminally even when reflection is enabled, while preserving failure artifacts and normal healthy-LLM reflection behavior.

Dependency versions in `requirements.txt` record the validated Linux and Python 3.10 environment. Install optional dataset-preparation packages (`requirements-optional.txt` or `PyTDC`) in an isolated environment when preparing upstream datasets.

New runs use the configuration templates in `configs/` and `.env.example`. Historical execution archives retain their original frozen source identities and schemas.
