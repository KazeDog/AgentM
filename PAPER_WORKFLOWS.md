# Manuscript workflows

This map identifies the retained code for the AgentM manuscript and Supplementary Information. Run commands from the repository root. Model weights, source datasets, retrieval contents and archived experimental records are supplied separately as described in [RESOURCE_GUIDE.md](RESOURCE_GUIDE.md).

## Implementation and experiment entry points

| Manuscript component | Implementation or entry point | Configuration and inputs |
| --- | --- | --- |
| Guarded multi-agent reporting | `agentm/runtime/`, `scripts/run_multi_agent.py` | Task arguments, model registry and retrieval resources |
| Web demonstration and interaction | `scripts/run_web_demo.py` | The same main controller; bundled local HTML template |
| Eight MoleculeNet tasks | `scripts/run_selected_moleculenet_multiseed.py` | `all_features`, seeds 1–5, scaffold and random-scaffold splits |
| Seven selected TDC endpoints | `scripts/run_selected_tdc_multiseed.py` | The seven selected files in `configs/tdc/`; seeds 1–5 |
| Molecular-safety cases | `configs/molecular_safety_cases.jsonl`, `scripts/run_benchmark_suite.py` | Expanded 100-case set; the retained 50-case reference supports selection checks |
| Reaction tools and cases | `scripts/train_reaction_models.py`, `scripts/run_reaction_model_benchmark.py`, `scripts/run_reaction_benchmark.py` | `configs/reaction_cases.jsonl` and external reaction resources |
| Clean, stability, stress and open tasks | `experiments/agentm_reliability/run_matrix.py` | Matching matrix configuration and manifest under `experiments/agentm_reliability/` |
| Multi-turn state updates | `experiments/agentm_reliability/multi_turn_runner.py` | `configs/multi_turn_tasks.yaml` within the reliability directory |
| Prospective paired stress design | `prepare_single_build_stress.py`, `preflight_single_build_stress.py` within the reliability directory | `configs/stress_single_build.yaml`, the 64-case controlled-stress manifest and frozen external resources |
| Exploratory reflection depth | `reflection_depth.py`, `prepare_depth_jobs.py` within the reliability directory | Explicit depth-aware jobs and accepted paired records; partial cohorts remain separate |
| Evidence preparation and indexing | `scripts/build_real_rag_corpus.py`, `scripts/build_ddi_rag_corpus.py`, `scripts/build_paper_evidence_qc.py`, `scripts/index_evidence_es.py` | Authorized source documents and an explicitly configured retrieval index |
| Evidence acceptance and provenance | `scripts/run_evidence_acceptance_suite.py`, `experiments/agentm_reliability/evidence_provenance.py` | Retrieval traces, retained source records and target-alignment rules |
| Independent evaluation and statistics | `canonicalize_runs.py`, `offline_evaluator.py`, `statistics.py` within the reliability directory | Complete raw run records and the appropriate frozen design |
| Reliability figures and tables | `render_outputs.py`, `render_manuscript_composites.py` within the reliability directory | Evaluated case-level tables and statistical outputs |

The core molecular components in `agentm/agents`, `features`, `models`, `data`, `rules` and `tools` support the main runtime and training workflows, and regression tests cover these scientific interfaces alongside the full case manifests, labels and evaluation rules.

## Select explicit hardware and scope

The following predictive commands require prepared datasets and feature-model assets. Select GPU identifiers available on the target machine:

```bash
python scripts/run_selected_moleculenet_multiseed.py --gpus 0 --seeds 1,2,3,4,5 --splits scaffold,random_scaffold
python scripts/run_selected_tdc_multiseed.py --gpus 0 --seeds 1,2,3,4,5
```

Check options without starting an experiment:

```bash
python scripts/run_multi_agent.py --help
python -m experiments.agentm_reliability.run_matrix --help
python -m experiments.agentm_reliability.multi_turn_runner --help
python -m experiments.agentm_reliability.prepare_depth_jobs --help
```

Runtime configuration determines whether model APIs or live retrieval are used. Use finite budgets and a frozen job list for paid experiments. Repository validation does not launch these workflows.

## Analyze an existing reliability run directory

For a directory whose records match the packaged evaluator and design:

```bash
python -m experiments.agentm_reliability.canonicalize_runs /path/to/run-directory
python -m experiments.agentm_reliability.offline_evaluator /path/to/run-directory
python -m experiments.agentm_reliability.statistics /path/to/run-directory
python -m experiments.agentm_reliability.render_outputs /path/to/run-directory
```

These commands write derived analysis files. Retain the original raw records and keep historical and prospective cohorts separate. Completion and evidence gates remain active on evaluated run directories; complete inputs are required for replication statistics. Archived raw execution records are maintained separately from this source repository.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for validation scope. Resource-equipped numerical reproduction additionally requires the external assets listed in [RESOURCE_GUIDE.md](RESOURCE_GUIDE.md).
