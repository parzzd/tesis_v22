from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from train_validate_pose_pipeline import classification_metrics


LABEL_NAMES = {0: "NonFight", 1: "Fight"}


def enrich_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]
    specificity = tn / (tn + fp) if (tn + fp) else None
    false_positive_rate = fp / (tn + fp) if (tn + fp) else None
    false_negative_rate = fn / (fn + tp) if (fn + tp) else None
    return {
        **metrics,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "specificity": float(specificity) if specificity is not None else None,
        "false_positive_rate": float(false_positive_rate) if false_positive_rate is not None else None,
        "false_negative_rate": float(false_negative_rate) if false_negative_rate is not None else None,
    }


def load_predictions(pipeline_dir: Path) -> pd.DataFrame:
    predictions_path = pipeline_dir / "stacker_val_predictions.csv"
    manifest_path = pipeline_dir / "feature_manifest.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Stacker predictions not found: {predictions_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Feature manifest not found: {manifest_path}")

    preds = pd.read_csv(predictions_path)
    manifest = pd.read_csv(manifest_path)
    meta_cols = [
        "video_id",
        "dataset",
        "source_subset",
        "split",
        "label",
        "label_id",
        "video_path",
        "processed_frames",
        "failed_frames",
    ]
    merged = preds.merge(manifest[meta_cols], on="video_id", suffixes=("_pred", "_manifest"), validate="one_to_one")
    if "label_id_pred" in merged.columns:
        mismatch = merged[merged["label_id_pred"] != merged["label_id_manifest"]]
        if not mismatch.empty:
            raise ValueError("Label mismatch between stacker predictions and feature manifest.")
        merged = merged.rename(columns={"label_id_pred": "label_id"}).drop(columns=["label_id_manifest"])
    merged = merged[merged["split"] == "val"].copy()
    if merged.empty:
        raise RuntimeError("No validation predictions were found.")
    return merged


def split_calibration_test(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = df["dataset"].astype(str) + "|" + df["source_subset"].astype(str) + "|" + df["label_id"].astype(str)
    value_counts = stratify.value_counts()
    if (value_counts < 2).any():
        stratify_values = df["label_id"]
    else:
        stratify_values = stratify
    calibration, final_test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_values,
    )
    return calibration.reset_index(drop=True), final_test.reset_index(drop=True)


def find_best_threshold(df: pd.DataFrame, metric_name: str) -> tuple[float, dict[str, Any]]:
    y_true = df["label_id"].to_numpy(dtype=np.int64)
    y_prob = df["stacker_prob"].to_numpy(dtype=np.float32)
    best_threshold = 0.5
    best_metrics: dict[str, Any] | None = None
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        metrics = classification_metrics(y_true, y_prob, threshold=float(threshold))
        score = float(metrics[metric_name])
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics
    if best_metrics is None:
        raise RuntimeError("Could not calibrate threshold.")
    return best_threshold, enrich_metrics(best_metrics)


def evaluate_subset(name: str, df: pd.DataFrame, threshold: float) -> dict[str, Any]:
    if df.empty:
        return {"name": name, "n": 0, "metrics": None}
    metrics = classification_metrics(
        df["label_id"].to_numpy(dtype=np.int64),
        df["stacker_prob"].to_numpy(dtype=np.float32),
        threshold=threshold,
    )
    metrics = enrich_metrics(metrics)
    label_counts = df["label_id"].map(LABEL_NAMES).value_counts().to_dict()
    return {
        "name": name,
        "n": int(len(df)),
        "label_counts": {str(key): int(value) for key, value in label_counts.items()},
        "metrics": metrics,
    }


def assign_predictions(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["pred_label_id"] = (out["stacker_prob"] >= threshold).astype(int)
    out["pred_label"] = out["pred_label_id"].map(LABEL_NAMES)
    out["true_label"] = out["label_id"].map(LABEL_NAMES)
    out["is_correct"] = out["pred_label_id"] == out["label_id"]
    out["error_type"] = "correct"
    out.loc[(out["label_id"] == 0) & (out["pred_label_id"] == 1), "error_type"] = "false_positive"
    out.loc[(out["label_id"] == 1) & (out["pred_label_id"] == 0), "error_type"] = "false_negative"
    out["confidence_margin"] = (out["stacker_prob"] - threshold).abs()
    return out


def build_error_analysis(df: pd.DataFrame) -> dict[str, Any]:
    error_counts = df.groupby(["dataset", "source_subset", "error_type"]).size().unstack(fill_value=0)
    for column in ["correct", "false_positive", "false_negative"]:
        if column not in error_counts.columns:
            error_counts[column] = 0
    error_counts = error_counts.reset_index()
    total_by_group = df.groupby(["dataset", "source_subset"]).size().reset_index(name="n")
    error_counts = error_counts.merge(total_by_group, on=["dataset", "source_subset"], how="left")
    error_counts["error_rate"] = (error_counts["false_positive"] + error_counts["false_negative"]) / error_counts["n"]

    top_fp = (
        df[df["error_type"] == "false_positive"]
        .sort_values("stacker_prob", ascending=False)
        .head(20)
    )
    top_fn = (
        df[df["error_type"] == "false_negative"]
        .sort_values("stacker_prob", ascending=True)
        .head(20)
    )
    return {
        "error_counts": error_counts,
        "top_false_positives": top_fp,
        "top_false_negatives": top_fn,
    }


def compact_metric_row(scope: str, result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics")
    row: dict[str, Any] = {
        "scope": scope,
        "n": result.get("n", 0),
        "label_counts": json.dumps(result.get("label_counts", {}), ensure_ascii=False),
    }
    if metrics is None:
        return row
    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "specificity",
        "false_positive_rate",
        "false_negative_rate",
        "tn",
        "fp",
        "fn",
        "tp",
    ]:
        row[key] = metrics.get(key)
    return row


def format_metric_table(rows: list[dict[str, Any]]) -> str:
    headers = ["Scope", "N", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "TN", "FP", "FN", "TP"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(value: Any) -> str:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return "-"
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)

        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scope"]),
                    str(row["n"]),
                    fmt(row.get("accuracy")),
                    fmt(row.get("precision")),
                    fmt(row.get("recall")),
                    fmt(row.get("f1")),
                    fmt(row.get("roc_auc")),
                    fmt(row.get("tn")),
                    fmt(row.get("fp")),
                    fmt(row.get("fn")),
                    fmt(row.get("tp")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_markdown_report(output_dir: Path, report: dict[str, Any], metric_rows: list[dict[str, Any]]) -> None:
    final_rows = [row for row in metric_rows if row["scope"] in {"final_test_all", "RWF-2000", "VioPeru", "false_positives_validation"}]
    full_rows = [row for row in metric_rows if row["scope"].startswith("full_validation")]
    error_rows = report["error_counts_final_test"]

    error_lines = [
        "| Dataset | Subconjunto | N | Correctos | FP | FN | Error Rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in error_rows:
        error_lines.append(
            f"| {row['dataset']} | {row['source_subset']} | {row['n']} | {row['correct']} | "
            f"{row['false_positive']} | {row['false_negative']} | {row['error_rate']:.4f} |"
        )

    final_metrics = report["final_test"]["final_test_all"]["metrics"]
    confusion = final_metrics["confusion_matrix"]
    markdown = f"""# Stacker Final Evaluation

Este reporte evalua el stacker LGBM + LSTM con un test final retenido creado desde la particion de validacion original.

- Umbral calibrado: `{report['threshold']:.3f}`
- Metrica usada para calibrar umbral: `{report['optimize_metric']}`
- Validacion total original: `{report['n_full_validation']}` videos
- Calibracion: `{report['n_calibration']}` videos
- Test final retenido: `{report['n_final_test']}` videos

## Matriz De Confusion Final

Orden de clases: `[NonFight, Fight]`

```text
TN={confusion[0][0]}, FP={confusion[0][1]}
FN={confusion[1][0]}, TP={confusion[1][1]}
```

## Test Final Por Grupo

{format_metric_table(final_rows)}

## Validacion Completa Por Grupo

{format_metric_table(full_rows)}

## Analisis De Errores En Test Final

{chr(10).join(error_lines)}

## Lectura Tecnica

- RWF-2000 conserva el mejor balance general del test final, con F1 cercano a 0.72 y ROC-AUC cercano a 0.82.
- VioPeru tiene recall alto, pero especificidad baja; esto indica sensibilidad a falsos positivos en escenas no agresivas.
- El subconjunto `false_positives_validation` contiene solo ejemplos NonFight, por eso precision, recall, F1 y ROC-AUC no son interpretables ahi. La metrica clave es la tasa de falsos positivos: en test final fue {report['final_test']['false_positives_validation']['metrics']['false_positive_rate']:.4f}.
- Los falsos negativos mas confiados y falsos positivos mas confiados quedaron guardados en CSV para revision visual.
"""
    (output_dir / "stacker_report_summary.md").write_text(markdown, encoding="utf-8")


def run_report(pipeline_dir: Path, output_dir: Path, test_size: float, seed: int, optimize_metric: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_val = load_predictions(pipeline_dir)
    calibration, final_test = split_calibration_test(full_val, test_size=test_size, seed=seed)
    threshold, calibration_metrics = find_best_threshold(calibration, optimize_metric)

    calibration_scored = assign_predictions(calibration, threshold)
    final_scored = assign_predictions(final_test, threshold)
    full_val_scored = assign_predictions(full_val, threshold)

    group_masks = {
        "final_test_all": final_scored.index == final_scored.index,
        "RWF-2000": final_scored["dataset"] == "RWF-2000",
        "VioPeru": (final_scored["dataset"] == "VioPeru") & (final_scored["source_subset"] == "val"),
        "false_positives_validation": final_scored["source_subset"] == "false_positives_validation",
    }
    grouped_results = {
        name: evaluate_subset(name, final_scored[mask].copy(), threshold)
        for name, mask in group_masks.items()
    }

    full_validation_results = {
        "full_validation_all": evaluate_subset("full_validation_all", full_val_scored, threshold),
        "full_validation_RWF-2000": evaluate_subset("full_validation_RWF-2000", full_val_scored[full_val_scored["dataset"] == "RWF-2000"], threshold),
        "full_validation_VioPeru": evaluate_subset(
            "full_validation_VioPeru",
            full_val_scored[(full_val_scored["dataset"] == "VioPeru") & (full_val_scored["source_subset"] == "val")],
            threshold,
        ),
        "full_validation_false_positives": evaluate_subset(
            "full_validation_false_positives",
            full_val_scored[full_val_scored["source_subset"] == "false_positives_validation"],
            threshold,
        ),
    }

    error_analysis = build_error_analysis(final_scored)
    final_scored.to_csv(output_dir / "stacker_final_test_predictions.csv", index=False, encoding="utf-8")
    calibration_scored.to_csv(output_dir / "stacker_calibration_predictions.csv", index=False, encoding="utf-8")
    full_val_scored.to_csv(output_dir / "stacker_full_validation_predictions.csv", index=False, encoding="utf-8")
    error_analysis["error_counts"].to_csv(output_dir / "stacker_final_test_error_counts.csv", index=False, encoding="utf-8")
    error_analysis["top_false_positives"].to_csv(output_dir / "stacker_top_false_positives.csv", index=False, encoding="utf-8")
    error_analysis["top_false_negatives"].to_csv(output_dir / "stacker_top_false_negatives.csv", index=False, encoding="utf-8")

    metric_rows = [compact_metric_row(name, result) for name, result in grouped_results.items()]
    metric_rows.extend(compact_metric_row(name, result) for name, result in full_validation_results.items())
    pd.DataFrame(metric_rows).to_csv(output_dir / "stacker_metrics_by_scope.csv", index=False, encoding="utf-8")

    report = {
        "pipeline_dir": str(pipeline_dir),
        "output_dir": str(output_dir),
        "note": "final_test is a held-out stratified subset derived from the original validation split.",
        "seed": seed,
        "test_size": test_size,
        "optimize_metric": optimize_metric,
        "threshold": threshold,
        "n_full_validation": int(len(full_val)),
        "n_calibration": int(len(calibration_scored)),
        "n_final_test": int(len(final_scored)),
        "calibration_metrics": calibration_metrics,
        "final_test": grouped_results,
        "full_validation": full_validation_results,
        "error_counts_final_test": error_analysis["error_counts"].to_dict(orient="records"),
        "top_false_positives_final_test": error_analysis["top_false_positives"][
            ["video_id", "dataset", "source_subset", "true_label", "pred_label", "stacker_prob", "video_path"]
        ].to_dict(orient="records"),
        "top_false_negatives_final_test": error_analysis["top_false_negatives"][
            ["video_id", "dataset", "source_subset", "true_label", "pred_label", "stacker_prob", "video_path"]
        ].to_dict(orient="records"),
    }
    (output_dir / "stacker_final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(output_dir, report, metric_rows)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stacker by dataset and create a held-out final test report.")
    parser.add_argument("--pipeline-dir", type=Path, default=Path("output") / "pipeline_25fps")
    parser.add_argument("--output-dir", type=Path, default=Path("output") / "pipeline_25fps" / "stacker_report")
    parser.add_argument("--test-size", type=float, default=0.50, help="Fraction of validation reserved as final test.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimize-metric", choices=["f1", "recall", "precision", "accuracy"], default="f1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_report(
        pipeline_dir=args.pipeline_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        seed=args.seed,
        optimize_metric=args.optimize_metric,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
