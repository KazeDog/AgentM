#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = (
    "HIA_Hou",
    "Pgp_Broccatelli",
    "BBB_Martins",
    "CYP2C9_Veith",
    "LD50_Zhu",
    "hERG",
    "DILI",
)
PRIMARY_METRICS = {
    "HIA_Hou": ("roc_auc", "higher"),
    "Pgp_Broccatelli": ("roc_auc", "higher"),
    "BBB_Martins": ("roc_auc", "higher"),
    "CYP2C9_Veith": ("aupr", "higher"),
    "LD50_Zhu": ("mae", "lower"),
    "hERG": ("roc_auc", "higher"),
    "DILI": ("roc_auc", "higher"),
}


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
    parser = argparse.ArgumentParser(description="Run the selected TDC ADMET tasks over repeated seeds.")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--gpus", default="7")
    parser.add_argument("--output-root", default="outputs/tdc_selected_multiseed")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _run_task_group(
    tasks: list[str],
    seeds: list[int],
    gpu: str,
    output_root: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for task in tasks:
        config_path = ROOT / "configs" / "tdc" / f"admet_{task}.yaml"
        if not config_path.exists():
            statuses.append({"task": task, "status": "missing_config", "config": str(config_path)})
            continue
        cache_dir = output_root / "_feature_cache" / task
        first_rules_path = output_root / task / f"seed_{seeds[0]}" / "rules_used.json"
        for seed in seeds:
            run_dir = output_root / task / f"seed_{seed}"
            result_path = run_dir / "training_result.json"
            if resume and result_path.exists():
                statuses.append({"task": task, "seed": seed, "status": "resumed", "result": str(result_path)})
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_root / "logs" / f"{task}_seed_{seed}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(ROOT / "scripts" / "train_admet.py"),
                "--config",
                str(config_path),
                "--seed",
                str(seed),
                "--output-dir",
                str(run_dir),
                "--persistent-feature-cache-dir",
                str(cache_dir),
            ]
            if seed != seeds[0] and first_rules_path.exists():
                command.extend(["--precomputed-rules-path", str(first_rules_path)])
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
            status = "completed" if completed.returncode == 0 and result_path.exists() else "failed"
            statuses.append(
                {
                    "task": task,
                    "seed": seed,
                    "gpu": gpu,
                    "status": status,
                    "returncode": completed.returncode,
                    "result": str(result_path),
                    "log": str(log_path),
                }
            )
            print(f"[{status}] task={task} seed={seed} gpu={gpu}", flush=True)
            if status == "failed":
                break
    return statuses


def summarize(output_root: Path, tasks: list[str], seeds: list[int]) -> tuple[Path, Path]:
    long_rows: list[dict[str, Any]] = []
    for task in tasks:
        primary_metric, direction = PRIMARY_METRICS[task]
        for seed in seeds:
            result_path = output_root / task / f"seed_{seed}" / "training_result.json"
            if not result_path.exists():
                continue
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            metrics = dict(payload.get("metrics") or {})
            row = {
                "dataset": task,
                "seed": seed,
                "split_type": payload.get("split_type"),
                "n_train": payload.get("n_train"),
                "n_valid": payload.get("n_valid"),
                "n_test": payload.get("n_test"),
                "primary_metric": primary_metric,
                "direction": direction,
                "primary_value": metrics.get(primary_metric),
            }
            row.update(metrics)
            long_rows.append(row)

    long_path = output_root / "multiseed_results_long.csv"
    fieldnames = sorted({key for row in long_rows for key in row}) if long_rows else [
        "dataset",
        "seed",
        "primary_metric",
        "primary_value",
    ]
    with long_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(long_rows)

    summary_rows: list[dict[str, Any]] = []
    for task in tasks:
        rows = [row for row in long_rows if row["dataset"] == task and row.get("primary_value") is not None]
        values = [float(row["primary_value"]) for row in rows]
        metric, direction = PRIMARY_METRICS[task]
        mean = statistics.mean(values) if values else None
        std = statistics.stdev(values) if len(values) >= 2 else None
        summary_rows.append(
            {
                "dataset": task,
                "metric": metric,
                "direction": direction,
                "n_seeds": len(values),
                "seeds": ",".join(str(row["seed"]) for row in rows),
                "mean": mean,
                "std": std,
                "mean_std": f"{mean:.4f} ± {std:.4f}" if mean is not None and std is not None else "",
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "complete_5_seeds": len(values) == len(seeds),
                "protocol": "local deterministic scaffold 80/10/10; not TDC BenchmarkGroup fixed test",
            }
        )
    summary_path = output_root / "multiseed_summary.csv"
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(summary_path, index=False)
    _write_markdown(
        summary_frame,
        output_root / "multiseed_summary.md",
        ["dataset", "metric", "direction", "n_seeds", "mean_std", "min", "max", "protocol"],
    )
    return long_path, summary_path


def main() -> int:
    args = parse_args()
    tasks = _csv_values(args.tasks)
    unknown = sorted(set(tasks) - set(DEFAULT_TASKS))
    if unknown:
        raise SystemExit(f"Unsupported selected task(s): {', '.join(unknown)}")
    seeds = [int(value) for value in _csv_values(args.seeds)]
    gpus = _csv_values(args.gpus)
    if not seeds or not gpus:
        raise SystemExit("At least one seed and one GPU are required.")
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    assignments = [tasks[index:: len(gpus)] for index in range(len(gpus))]
    statuses: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(
                _run_task_group,
                assignment,
                seeds,
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
    long_path, summary_path = summarize(output_root, tasks, seeds)
    failures = [row for row in statuses if row.get("status") in {"failed", "missing_config"}]
    print(f"[summary] long={long_path} summary={summary_path} failures={len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
