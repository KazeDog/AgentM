# Reproducibility

## Source identity

Python imports, command entry points, environment variables and output fields use unified project-relative names. `tools/verify_release.py` checks Python syntax, import structure, naming hygiene and common credential patterns offline.

The experiment harness hashes the `agentm` package and reliability harness using repository-relative paths, so relocating the checkout preserves this source identity. Dataset, model, manifest and retrieval-index identities are recorded separately for resource-equipped experiments.

## Case definitions

`configs/` contains the molecular-safety and reaction case definitions, and `experiments/agentm_reliability/manifests/` contains the clean, controlled-stress, open-task and multi-turn manifests. Evaluated molecules, endpoints, target values, injected faults and scientific settings are preserved in the packaged manifests, with content hashes computed over the packaged JSON-lines records.

Archived experimental runs retain the frozen source and manifest identities recorded at execution time. Re-running numerical benchmarks in a new environment uses the external assets in [RESOURCE_GUIDE.md](RESOURCE_GUIDE.md) together with a newly frozen run manifest.

## Validation

Offline checks run against synthetic fixtures and mock responses with Python networking blocked and inherited API credentials cleared. They verify scientific output semantics, tool guards, evidence checks, reflection control, budget handling, import roots, path resolution and report serialization, including a temporary-directory relocation check.

Validation summaries in `validation/` record these offline checks. Predictive benchmarking and full reliability-matrix execution require the external datasets, model weights and retrieval index described in [RESOURCE_GUIDE.md](RESOURCE_GUIDE.md).
