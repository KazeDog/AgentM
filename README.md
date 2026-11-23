# AgentM

Official source repository for **AgentM: a guarded multi-agent system for molecular safety and reaction reasoning** (*Science China Information Sciences*).

AgentM coordinates registered scientific tools, task planning, evidence retrieval, verification and bounded reflection to produce provenance-constrained, auditable reports across ADMET, drug--drug interaction (DDI) and chemical reaction tasks.

## Repository layout

```text
AgentM/
├── agentm/                  # Unified Python package
│   ├── agents/             # Molecular analysis components
│   ├── features/           # Fingerprints and learned representations
│   ├── models/             # Prediction models and training interfaces
│   └── runtime/            # Main controller, task graph, guards and web interface
├── scripts/                # Training, prediction and application entry points
├── configs/                # Prediction settings and case definitions
├── experiments/            # Reliability protocols, evaluators and analysis
├── tests/                  # Regression and repository-layout tests
├── tools/                  # Network-blocked validation tools
└── validation/             # Recorded offline validation results
```

All components import from `agentm`. Run commands from the repository root. The public controller is available as:

```python
from agentm import AgentMMultiAgentController
```

The source package retains the manuscript's prediction, reporting, retrieval, verification, reflection and evaluation workflows. The main controller is the single application entry point; its web page is stored locally in `agentm/runtime/web/templates/`. The lower-level molecular components also support model training and feature construction.

## Installation

Use Linux and Python 3.10. Install a hardware-compatible PyTorch build and the listed dependencies in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependency pins record the validated Linux and Python 3.10 software environment. Optional dataset and retrieval dependencies are listed separately in `requirements-optional.txt`.

## Offline checks

```bash
python -B tools/verify_release.py
python -B tools/run_offline_tests.py --report /tmp/agentm-tests.json
python -B tools/run_offline_tests.py --modules test_repository_layout --report /tmp/agentm-layout.json
python -B tools/run_offline_tests.py --modules test_evidence_manifest test_web_demo --report /tmp/agentm-support.json
python -B tools/check_entrypoints.py --report /tmp/agentm-entrypoints.json
python -B tools/check_web.py --report /tmp/agentm-web.json
python -B tools/smoke_offline.py --output /tmp/agentm-smoke
```

Choose a new or empty smoke output directory. The source-package check reads files only. The execution checks clear inherited API credentials, hide GPU devices and block Python network operations. They use synthetic fixtures and mocks; no paid model calls or resource downloads are required. The smoke test exercises input guarding and report serialization on an unsupported request. Validation outcomes are recorded in `validation/`.

## Applications and experiments

Inspect the available options before configuring resources:

```bash
python scripts/run_multi_agent.py --help
python scripts/run_reaction_agent.py --help
python scripts/run_web_demo.py --help
python scripts/run_selected_moleculenet_multiseed.py --help
python scripts/run_selected_tdc_multiseed.py --help
python -m experiments.agentm_reliability.run_matrix --help
```

The MoleculeNet study uses BACE, BBBP, Tox21, ToxCast, SIDER, ESOL, FreeSolv and Lipophilicity, with `all_features`, five seeds, scaffold and random-scaffold partitions. The selected TDC study uses HIA_Hou, Pgp_Broccatelli, BBB_Martins, CYP2C9_Veith, LD50_Zhu, hERG and DILI with the project's scaffold protocol. Pass the manuscript protocol arguments explicitly when launching benchmark scripts.

Research workflows require the relevant data, trained weights and retrieval resources. Some research defaults enable online retrieval or language models. Configure credentials and finite call/token budgets before intentionally running those workflows. The offline checks above do not launch them.

See [PAPER_WORKFLOWS.md](PAPER_WORKFLOWS.md) for the manuscript-to-code entry-point map, [RESOURCE_GUIDE.md](RESOURCE_GUIDE.md) for asset locations, and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for experiment identities and validation scope.

## Credentials and distribution

The repository contains no API credentials, trained weights, raw restricted datasets or retrieval-index contents. `.env.example` documents configuration variables; it is not loaded automatically. Keep keys and service credentials in a private environment, and do not commit `.env`, `secret_keys.py`, model files or private outputs.

Project source code is released under the [MIT License](LICENSE). External datasets, model weights and retrieval resources remain subject to their upstream terms; see [LICENSE_STATUS.md](LICENSE_STATUS.md).
