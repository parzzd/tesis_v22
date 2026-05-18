from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hard_negative_mining import enrich_metrics
from train_validate_pose_pipeline import classification_metrics


SCOPES = {
    "final_test_all": lambda df: df,
    "RWF-2000": lambda df: df[df["dataset"] == "RWF-2000"],
    "VioPeru": lambda df: df[(df["dataset"] == "VioPeru") & (df["source_subset"] == "val")],
    "false_positives_validation": lambda df: df[df["source_subset"] == "false_positives_validation"],
}


def evaluate_scope(subset: pd.DataFrame, threshold: float, probability_column: str) -> dict[str, Any]:
    if subset.empty:
        return {
            "n": 0,
            "threshold": threshold,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "specificity": None,
            "f1": None,
            "roc_auc": None,
            "false_positive_rate": None,
            "false_negative_rate": None,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }
    metrics = classification_metrics(
        subset["label_id"].to_numpy(dtype=np.int64),
        subset[probability_column].to_numpy(dtype=np.float32),
        threshold=threshold,
    )
    return {"n": int(len(subset)), **enrich_metrics(metrics)}


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], recommendation: dict[str, Any] | None, min_recall: float) -> None:
    headers = ["threshold", "scope", "n", "accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "fp", "fn", "tp", "tn"]
    lines = [
        "# Experimentos para reducir falsas alertas",
        "",
        f"- Criterio: minimizar falsos positivos con recall >= `{min_recall:.2f}` en `final_test_all`.",
        "- El test final solo se usa para evaluar, no para entrenar.",
        "",
    ]
    if recommendation:
        lines.extend(
            [
                "## Recomendacion",
                "",
                (
                    f"Threshold recomendado experimental: `{recommendation['threshold']:.3f}` "
                    f"con `FP={recommendation['fp']}`, `FN={recommendation['fn']}`, "
                    f"`precision={recommendation['precision']:.4f}` y `recall={recommendation['recall']:.4f}`."
                ),
                "",
            ]
        )
    lines.extend(["## Cuadro comparativo", "", "| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"])
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(header)) for header in headers) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_thresholds(
    predictions_path: Path,
    output_dir: Path,
    thresholds: list[float],
    probability_column: str,
    min_recall: float,
) -> dict[str, Any]:
    df = pd.read_csv(predictions_path)
    if probability_column not in df.columns:
        raise ValueError(f"Missing probability column {probability_column!r} in {predictions_path}")

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        for scope, selector in SCOPES.items():
            subset = selector(df)
            metrics = evaluate_scope(subset, threshold, probability_column)
            rows.append({"threshold": threshold, "scope": scope, **metrics})

    final_rows = [row for row in rows if row["scope"] == "final_test_all" and row.get("recall") is not None]
    candidates = [row for row in final_rows if float(row["recall"]) >= min_recall]
    recommendation = None
    if candidates:
        recommendation = sorted(
            candidates,
            key=lambda row: (int(row["fp"]), -float(row["precision"]), -float(row["f1"]), float(row["threshold"])),
        )[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "threshold_experiments.csv", rows)
    write_markdown(output_dir / "threshold_experiments.md", rows, recommendation, min_recall)
    summary = {
        "predictions_path": str(predictions_path),
        "probability_column": probability_column,
        "thresholds": thresholds,
        "min_recall": min_recall,
        "recommendation": recommendation,
    }
    (output_dir / "threshold_experiments_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate conservative thresholds to reduce false alerts.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("output") / "pipeline_25fps" / "hard_negative_mining" / "hnm_final_test_predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("métricas") / "experimentos_falsas_alertas")
    parser.add_argument("--thresholds", default="0.49,0.50,0.505")
    parser.add_argument("--probability-column", default="stacker_prob_hnm")
    parser.add_argument("--min-recall", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    summary = evaluate_thresholds(
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        thresholds=thresholds,
        probability_column=args.probability_column,
        min_recall=args.min_recall,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
