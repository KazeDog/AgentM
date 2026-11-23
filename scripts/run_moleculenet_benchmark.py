from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentm.runtime.moleculenet import MOLECULENET_SPECS, run_one_benchmark, summarize_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run canonical MoleculeNet scaffold and random-scaffold benchmarks."
    )
    parser.add_argument("--datasets", default="all", help="Comma-separated names or 'all'.")
    parser.add_argument("--splits", default="scaffold,random_scaffold")
    parser.add_argument("--raw-dir", default="data/moleculenet/raw")
    parser.add_argument("--processed-dir", default="data/moleculenet/processed")
    parser.add_argument("--output-root", default="outputs/moleculenet_agentm_structural_seed13")
    parser.add_argument("--feature-set", choices=["agentm_structural", "all_features"], default="agentm_structural")
    parser.add_argument(
        "--chemberta-model-path",
        default="checkpoints/chemberta2/DeepChem-ChemBERTa-77M-MTR",
    )
    parser.add_argument(
        "--imagemol-checkpoint-path",
        default="checkpoints/imagemol/ImageMol.pth.tar",
    )
    parser.add_argument("--max-rule-features", type=int, default=23)
    parser.add_argument(
        "--clintox-risk-labels",
        action="store_true",
        help="Use FDA_NOT_APPROVED=1-FDA_APPROVED so both ClinTox positives are risk events.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", default="512,256")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _names(value: str, allowed: list[str]) -> list[str]:
    names = allowed.copy() if value.strip().lower() == "all" else [item.strip().lower() for item in value.split(",")]
    unknown = sorted(set(names) - set(allowed))
    if unknown:
        raise SystemExit(f"Unknown values: {', '.join(unknown)}")
    return names


def main() -> None:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    args = parse_args()
    datasets = _names(args.datasets, list(MOLECULENET_SPECS))
    splits = _names(args.splits, ["scaffold", "random_scaffold"])
    hidden_dims = tuple(int(value) for value in args.hidden_dims.split(",") if value.strip())
    results = []
    for dataset in datasets:
        for split_type in splits:
            result_path = (
                ROOT
                / args.output_root
                / dataset
                / split_type
                / f"seed_{args.seed}"
                / "result.json"
            )
            if result_path.exists() and not args.no_resume:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                results.append(result)
                print(f"[resume] dataset={dataset} split={split_type}", flush=True)
                continue
            print(f"[start] dataset={dataset} split={split_type}", flush=True)
            result = run_one_benchmark(
                dataset,
                split_type,
                raw_dir=ROOT / args.raw_dir,
                processed_dir=ROOT / args.processed_dir,
                output_root=ROOT / args.output_root,
                seed=args.seed,
                device=args.device,
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                hidden_dims=hidden_dims,
                dropout=args.dropout,
                download=args.download,
                force_features=args.force_features,
                feature_set=args.feature_set,
                chemberta_model_path=ROOT / args.chemberta_model_path,
                imagemol_checkpoint_path=ROOT / args.imagemol_checkpoint_path,
                max_rule_features=args.max_rule_features,
                clintox_risk_labels=args.clintox_risk_labels,
            )
            results.append(result)
            print(
                f"[done] dataset={dataset} split={split_type} "
                f"best_epoch={result['best_epoch']} metrics={result['metrics']}",
                flush=True,
            )
    summary = summarize_results(results, ROOT / args.output_root)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
