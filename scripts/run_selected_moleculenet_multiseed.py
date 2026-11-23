#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_DATASETS = ("bace", "bbbp", "tox21", "toxcast", "sider", "esol", "freesolv", "lipophilicity")
CLASSIFICATION_DATASETS = {"bace", "bbbp", "tox21", "toxcast", "sider"}


def _write_markdown(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame[columns].to_dict(orient="records"):
        lines.append("| " + " | ".join(cell(row[column]) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the selected MoleculeNet tasks over repeated seeds.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--splits", default="scaffold,random_scaffold")
    parser.add_argument("--gpus", default="6")
    parser.add_argument("--output-root", default="outputs/moleculenet_agentm_all_features_multiseed")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _run_dataset_group(
    datasets: list[str],
    seeds: list[int],
    splits: list[str],
    gpu: str,
    output_root: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            result_paths = [output_root / dataset / split / f"seed_{seed}" / "result.json" for split in splits]
            if resume and all(path.exists() for path in result_paths):
                statuses.append({"dataset": dataset, "seed": seed, "status": "resumed"})
                continue
            log_path = output_root / "logs" / f"{dataset}_seed_{seed}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_moleculenet_benchmark.py"),
                "--datasets",
                dataset,
                "--splits",
                ",".join(splits),
                "--output-root",
                str(output_root.relative_to(ROOT)),
                "--feature-set",
                "all_features",
                "--seed",
                str(seed),
                "--device",
                "cuda",
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            env.setdefault("OMP_NUM_THREADS", "8")
            env.setdefault("MKL_NUM_THREADS", "8")
            with log_path.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            status = "completed" if completed.returncode == 0 and all(path.exists() for path in result_paths) else "failed"
            statuses.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "gpu": gpu,
                    "status": status,
                    "returncode": completed.returncode,
                    "log": str(log_path),
                }
            )
            print(f"[{status}] dataset={dataset} seed={seed} gpu={gpu}", flush=True)
            if status == "failed":
                break
    return statuses


def summarize(output_root: Path, datasets: list[str], seeds: list[int], splits: list[str]) -> tuple[Path, Path]:
    results: list[dict[str, Any]] = []
    for dataset in datasets:
        for split in splits:
            for seed in seeds:
                path = output_root / dataset / split / f"seed_{seed}" / "result.json"
                if path.exists():
                    results.append(json.loads(path.read_text(encoding="utf-8")))

    from agentm.runtime.moleculenet import summarize_results

    long_frame = summarize_results(results, output_root)
    long_path = output_root / "multiseed_results_long.csv"
    long_frame.to_csv(long_path, index=False)
    summary_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        metric = "roc_auc_macro" if dataset in CLASSIFICATION_DATASETS else "rmse_macro"
        direction = "higher" if dataset in CLASSIFICATION_DATASETS else "lower"
        for split in splits:
            group = long_frame[(long_frame["dataset"] == dataset) & (long_frame["split_type"] == split)]
            values = [float(value) for value in group[metric].dropna().tolist()]
            mean = statistics.mean(values) if values else None
            std = statistics.stdev(values) if len(values) >= 2 else None
            summary_rows.append(
                {
                    "dataset": dataset,
                    "split_type": split,
                    "metric": metric,
                    "direction": direction,
                    "n_seeds": len(values),
                    "seeds": ",".join(str(value) for value in sorted(group["seed"].astype(int).tolist())),
                    "mean": mean,
                    "std": std,
                    "mean_std": f"{mean:.4f} ± {std:.4f}" if mean is not None and std is not None else "",
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                    "complete_5_seeds": len(values) == len(seeds),
                }
            )
    summary_path = output_root / "multiseed_summary.csv"
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(summary_path, index=False)
    _write_markdown(
        summary_frame,
        output_root / "multiseed_summary.md",
        ["dataset", "split_type", "metric", "direction", "n_seeds", "mean_std", "min", "max"],
    )
    return long_path, summary_path


def main() -> int:
    args = parse_args()
    datasets = _csv_values(args.datasets)
    unknown = sorted(set(datasets) - set(DEFAULT_DATASETS))
    if unknown:
        raise SystemExit(f"Unsupported selected dataset(s): {', '.join(unknown)}")
    seeds = [int(value) for value in _csv_values(args.seeds)]
    splits = _csv_values(args.splits)
    gpus = _csv_values(args.gpus)
    if not seeds or not splits or not gpus:
        raise SystemExit("At least one seed, split, and GPU are required.")
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    assignments = [datasets[index:: len(gpus)] for index in range(len(gpus))]
    statuses: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(
                _run_dataset_group,
                assignment,
                seeds,
                splits,
                gpu,
                output_root,
                not args.no_resume,
            )
            for assignment, gpu in zip(assignments, gpus)
            if assignment
        ]
        for future in as_completed(futures):
            statuses.extend(future.result())
    (output_root / "job_status.json").write_text(
        json.dumps(statuses, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    long_path, summary_path = summarize(output_root, datasets, seeds, splits)
    failures = [row for row in statuses if row.get("status") == "failed"]
    print(f"[summary] long={long_path} summary={summary_path} failures={len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
