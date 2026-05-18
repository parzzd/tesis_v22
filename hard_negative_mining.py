from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_validate_pose_pipeline import LSTMAggressionClassifier, classification_metrics, resolve_torch_device


META_FEATURES = ["lgbm_prob", "lstm_prob"]
LABEL_NAMES = {0: "NonFight", 1: "Fight"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_lgbm_feature_names(pipeline_dir: Path) -> list[str]:
    bundle_path = pipeline_dir / "models" / "lgbm_classifier.joblib"
    if bundle_path.exists():
        return list(joblib.load(bundle_path)["feature_names"])

    train_table_path = pipeline_dir / "lgbm_train_table.csv"
    if not train_table_path.exists():
        raise FileNotFoundError(f"Missing LGBM bundle and train table: {bundle_path}, {train_table_path}")
    table = pd.read_csv(train_table_path, nrows=1)
    metadata_columns = {"video_id", "dataset", "source_subset", "split", "label", "label_id", "video_path", "feature_npz"}
    return [column for column in table.columns if column not in metadata_columns]


def load_lstm_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing sequence file: {path}")
    data = np.load(path, allow_pickle=True)
    return {
        "sequences": np.asarray(data["sequences"], dtype=np.float32),
        "labels": np.asarray(data["labels"], dtype=np.int64),
        "video_ids": np.asarray(data["video_ids"]).astype(str),
    }


def enrich_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]
    metrics = dict(metrics)
    metrics.update(
        {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
            "false_positive_rate": float(fp / (tn + fp)) if (tn + fp) else None,
            "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else None,
        }
    )
    return metrics


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric_name: str = "f1") -> tuple[float, dict[str, Any]]:
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
        raise RuntimeError("Could not find a threshold.")
    return best_threshold, enrich_metrics(best_metrics)


def mine_hard_negatives(
    report_dir: Path,
    threshold: float,
    max_items: int | None = None,
    probability_column: str = "stacker_prob",
    calibration_filename: str = "stacker_calibration_predictions.csv",
) -> pd.DataFrame:
    calibration_path = report_dir / calibration_filename
    if not calibration_path.exists():
        raise FileNotFoundError(f"Missing calibration predictions: {calibration_path}")

    calibration = pd.read_csv(calibration_path)
    if probability_column not in calibration.columns:
        raise ValueError(f"Missing probability column {probability_column!r} in {calibration_path}")
    hard_negatives = calibration[
        (calibration["label_id"] == 0)
        & (calibration[probability_column] >= threshold)
    ].copy()
    hard_negatives["mining_probability"] = hard_negatives[probability_column]
    hard_negatives = hard_negatives.sort_values("mining_probability", ascending=False)
    if max_items is not None and max_items > 0:
        hard_negatives = hard_negatives.head(max_items)
    return hard_negatives.reset_index(drop=True)


def normalize_sequences(train_seq: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    flat = train_seq.reshape(-1, train_seq.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    normalized_train = ((train_seq - mean) / std).astype(np.float32)
    normalized_others = [((seq - mean) / std).astype(np.float32) for seq in others]
    return normalized_train, normalized_others, mean, std


def train_hnm_lgbm(
    train_table: pd.DataFrame,
    hard_table: pd.DataFrame,
    feature_names: list[str],
    hard_negative_weight: float,
    estimators: int,
    learning_rate: float,
    seed: int,
) -> tuple[LGBMClassifier, pd.DataFrame, np.ndarray]:
    combined = pd.concat([train_table, hard_table], ignore_index=True)
    sample_weight = np.ones(len(combined), dtype=np.float32)
    if len(hard_table):
        sample_weight[len(train_table) :] = float(hard_negative_weight)

    model = LGBMClassifier(
        objective="binary",
        n_estimators=estimators,
        learning_rate=learning_rate,
        num_leaves=31,
        min_child_samples=5,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(combined[feature_names], combined["label_id"].astype(int), sample_weight=sample_weight)
    return model, combined, sample_weight


def gather_hard_sequences(val_data: dict[str, np.ndarray], hard_ids: set[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index_by_id = {video_id: idx for idx, video_id in enumerate(val_data["video_ids"])}
    indices = [index_by_id[video_id] for video_id in hard_ids if video_id in index_by_id]
    if not indices:
        return (
            np.empty((0, val_data["sequences"].shape[1], val_data["sequences"].shape[2]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=str),
        )
    return (
        val_data["sequences"][indices],
        val_data["labels"][indices],
        val_data["video_ids"][indices],
    )


def train_hnm_lstm(
    train_data: dict[str, np.ndarray],
    hard_sequences: np.ndarray,
    hard_labels: np.ndarray,
    calibration_sequences: np.ndarray,
    calibration_labels: np.ndarray,
    final_sequences: np.ndarray,
    final_labels: np.ndarray,
    device: torch.device,
    hard_negative_repeats: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    patience: int,
) -> tuple[LSTMAggressionClassifier, dict[str, Any], np.ndarray, np.ndarray]:
    train_sequences = train_data["sequences"]
    train_labels = train_data["labels"]
    if len(hard_sequences) and hard_negative_repeats > 0:
        train_sequences = np.concatenate(
            [train_sequences, np.repeat(hard_sequences, hard_negative_repeats, axis=0)],
            axis=0,
        )
        train_labels = np.concatenate(
            [train_labels, np.repeat(hard_labels, hard_negative_repeats, axis=0)],
            axis=0,
        )

    train_norm, others_norm, feature_mean, feature_std = normalize_sequences(
        train_sequences,
        calibration_sequences,
        final_sequences,
    )
    calibration_norm, final_norm = others_norm

    model = LSTMAggressionClassifier(
        input_size=train_norm.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    y_train = train_labels.astype(np.float32)
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_norm), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1.0
    best_epoch = 0
    stagnant = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        cal_prob = predict_lstm_from_normalized(model, calibration_norm, device, batch_size)
        _, cal_best = find_best_threshold(calibration_labels, cal_prob, metric_name="f1")
        score = float(cal_best["f1"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else None,
                "calibration_f1": score,
                "calibration_recall": cal_best["recall"],
                "calibration_precision": cal_best["precision"],
            }
        )
        print(
            f"[HNM-LSTM] epoch {epoch:03d}/{epochs} "
            f"loss={np.mean(losses):.4f} cal_f1={score:.4f}"
        )

        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= patience:
                break

    if best_state is None:
        raise RuntimeError("LSTM hard-negative training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    metadata = {
        "input_size": int(train_norm.shape[-1]),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "best_epoch": int(best_epoch),
        "best_calibration_f1": float(best_score),
        "history": history,
        "n_train_sequences_after_hnm": int(len(train_norm)),
    }
    return model, metadata, calibration_norm, final_norm


def predict_lstm_from_normalized(
    model: LSTMAggressionClassifier,
    sequences: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = torch.from_numpy(sequences[start : start + batch_size]).to(device)
            logits = model(batch)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs).astype(np.float32) if probs else np.empty((0,), dtype=np.float32)


def build_sequence_subset(val_data: dict[str, np.ndarray], video_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    index_by_id = {video_id: idx for idx, video_id in enumerate(val_data["video_ids"])}
    indices = [index_by_id[video_id] for video_id in video_ids]
    return val_data["sequences"][indices], val_data["labels"][indices]


def build_meta_table(
    df: pd.DataFrame,
    lgbm_model: LGBMClassifier,
    feature_table: pd.DataFrame,
    feature_names: list[str],
    lstm_prob: np.ndarray,
) -> pd.DataFrame:
    lookup = feature_table.set_index("video_id")
    ordered_features = lookup.loc[df["video_id"], feature_names]
    lgbm_prob = lgbm_model.predict_proba(ordered_features)[:, 1].astype(np.float32)
    meta = df.copy().reset_index(drop=True)
    meta["lgbm_prob_hnm"] = lgbm_prob
    meta["lstm_prob_hnm"] = lstm_prob.astype(np.float32)
    meta["lgbm_prob"] = meta["lgbm_prob_hnm"]
    meta["lstm_prob"] = meta["lstm_prob_hnm"]
    return meta


def evaluate_scopes(df: pd.DataFrame, threshold: float) -> dict[str, Any]:
    scopes = {
        "final_test_all": df,
        "RWF-2000": df[df["dataset"] == "RWF-2000"],
        "VioPeru": df[(df["dataset"] == "VioPeru") & (df["source_subset"] == "val")],
        "false_positives_validation": df[df["source_subset"] == "false_positives_validation"],
    }
    results: dict[str, Any] = {}
    for name, subset in scopes.items():
        if subset.empty:
            results[name] = {"n": 0, "metrics": None}
            continue
        metrics = classification_metrics(
            subset["label_id"].to_numpy(dtype=np.int64),
            subset["stacker_prob_hnm"].to_numpy(dtype=np.float32),
            threshold=threshold,
        )
        results[name] = {
            "n": int(len(subset)),
            "label_counts": {
                LABEL_NAMES[int(key)]: int(value)
                for key, value in subset["label_id"].value_counts().to_dict().items()
            },
            "metrics": enrich_metrics(metrics),
        }
    return results


def add_predictions(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["pred_label_id_hnm"] = (out["stacker_prob_hnm"] >= threshold).astype(int)
    out["pred_label_hnm"] = out["pred_label_id_hnm"].map(LABEL_NAMES)
    out["true_label"] = out["label_id"].map(LABEL_NAMES)
    out["error_type_hnm"] = "correct"
    out.loc[(out["label_id"] == 0) & (out["pred_label_id_hnm"] == 1), "error_type_hnm"] = "false_positive"
    out.loc[(out["label_id"] == 1) & (out["pred_label_id_hnm"] == 0), "error_type_hnm"] = "false_negative"
    return out


def metrics_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope, result in results.items():
        row: dict[str, Any] = {"scope": scope, "n": result["n"]}
        metrics = result.get("metrics")
        if metrics is not None:
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
        rows.append(row)
    return rows


def write_summary_md(output_dir: Path, report: dict[str, Any]) -> None:
    def fmt(value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "-"
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    lines = [
        "# Hard Negative Mining Report",
        "",
        f"- Hard negatives usados: `{report['n_hard_negatives']}`",
        f"- Umbral de mineria: `{report['mining_threshold']:.3f}`",
        f"- Umbral final calibrado: `{report['threshold']:.3f}`",
        f"- Test final retenido: `{report['n_final_test']}` videos",
        "",
        "## Metricas En Test Final",
        "",
        "| Scope | N | Accuracy | Precision | Recall | F1 | ROC-AUC | TN | FP | FN | TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_rows(report["final_test"]):
        lines.append(
            f"| {row['scope']} | {row['n']} | {fmt(row.get('accuracy'))} | {fmt(row.get('precision'))} | "
            f"{fmt(row.get('recall'))} | {fmt(row.get('f1'))} | {fmt(row.get('roc_auc'))} | "
            f"{fmt(row.get('tn'))} | {fmt(row.get('fp'))} | {fmt(row.get('fn'))} | {fmt(row.get('tp'))} |"
        )

    before = report["baseline_final_test"]["final_test_all"]["metrics"]
    after = report["final_test"]["final_test_all"]["metrics"]
    lines.extend(
        [
            "",
            "## Comparacion General",
            "",
            "| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | FP | FN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| Stacker original | {fmt(before['accuracy'])} | {fmt(before['precision'])} | {fmt(before['recall'])} | "
            f"{fmt(before['f1'])} | {fmt(before['roc_auc'])} | {before['fp']} | {before['fn']} |",
            f"| Stacker HNM | {fmt(after['accuracy'])} | {fmt(after['precision'])} | {fmt(after['recall'])} | "
            f"{fmt(after['f1'])} | {fmt(after['roc_auc'])} | {after['fp']} | {after['fn']} |",
        ]
    )
    (output_dir / "hnm_summary.md").write_text("\n".join(lines), encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def run_hard_negative_mining(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_torch_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False
        print(f"[CUDA] Usando GPU para LSTM HNM: {torch.cuda.get_device_name(device)}")

    baseline_report = load_json(args.report_dir / "stacker_final_report.json")
    mining_source_report: dict[str, Any] | None = None
    mining_probability_column = "stacker_prob"
    mining_report_dir = args.report_dir
    mining_calibration_file = "stacker_calibration_predictions.csv"
    if args.mining_source == "previous_hnm":
        mining_source_report = load_json(args.hnm_source_dir / "hnm_report.json")
        mining_report_dir = args.hnm_source_dir
        mining_calibration_file = "hnm_calibration_predictions.csv"
        mining_probability_column = "stacker_prob_hnm"
    mining_threshold = (
        float(args.mining_threshold)
        if args.mining_threshold is not None
        else float((mining_source_report or baseline_report)["threshold"])
    )
    hard_negatives = mine_hard_negatives(
        mining_report_dir,
        mining_threshold,
        args.max_hard_negatives,
        probability_column=mining_probability_column,
        calibration_filename=mining_calibration_file,
    )
    hard_negatives.to_csv(output_dir / "hard_negatives_used.csv", index=False, encoding="utf-8")
    hard_ids = set(hard_negatives["video_id"].astype(str))
    print(f"[HNM] hard negatives mined from calibration: {len(hard_ids)}")

    feature_names = load_lgbm_feature_names(args.pipeline_dir)
    train_table = pd.read_csv(args.pipeline_dir / "lgbm_train_table.csv")
    val_table = pd.read_csv(args.pipeline_dir / "lgbm_val_table.csv")
    hard_table = val_table[val_table["video_id"].isin(hard_ids)].copy()

    lgbm_model, lgbm_combined, sample_weight = train_hnm_lgbm(
        train_table=train_table,
        hard_table=hard_table,
        feature_names=feature_names,
        hard_negative_weight=args.hard_negative_weight,
        estimators=args.lgbm_estimators,
        learning_rate=args.lgbm_learning_rate,
        seed=args.seed,
    )

    train_seq = load_lstm_npz(args.pipeline_dir / "lstm_train_sequences.npz")
    val_seq = load_lstm_npz(args.pipeline_dir / "lstm_val_sequences.npz")
    hard_sequences, hard_labels, hard_video_ids = gather_hard_sequences(val_seq, hard_ids)

    calibration_df = pd.read_csv(args.report_dir / "stacker_calibration_predictions.csv")
    final_df = pd.read_csv(args.report_dir / "stacker_final_test_predictions.csv")
    calibration_sequences, calibration_labels = build_sequence_subset(val_seq, calibration_df["video_id"].astype(str).tolist())
    final_sequences, final_labels = build_sequence_subset(val_seq, final_df["video_id"].astype(str).tolist())

    lstm_model, lstm_meta, calibration_norm, final_norm = train_hnm_lstm(
        train_data=train_seq,
        hard_sequences=hard_sequences,
        hard_labels=hard_labels,
        calibration_sequences=calibration_sequences,
        calibration_labels=calibration_labels,
        final_sequences=final_sequences,
        final_labels=final_labels,
        device=device,
        hard_negative_repeats=args.hard_negative_repeats,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
    )

    calibration_lstm_prob = predict_lstm_from_normalized(lstm_model, calibration_norm, device, args.batch_size)
    final_lstm_prob = predict_lstm_from_normalized(lstm_model, final_norm, device, args.batch_size)
    calibration_meta = build_meta_table(calibration_df, lgbm_model, val_table, feature_names, calibration_lstm_prob)
    final_meta = build_meta_table(final_df, lgbm_model, val_table, feature_names, final_lstm_prob)

    stacker = LogisticRegression(C=args.stacker_c, class_weight="balanced", max_iter=1000, random_state=args.seed)
    stacker.fit(calibration_meta[META_FEATURES], calibration_meta["label_id"])
    calibration_meta["stacker_prob_hnm"] = stacker.predict_proba(calibration_meta[META_FEATURES])[:, 1].astype(np.float32)
    final_meta["stacker_prob_hnm"] = stacker.predict_proba(final_meta[META_FEATURES])[:, 1].astype(np.float32)
    threshold, calibration_metrics = find_best_threshold(
        calibration_meta["label_id"].to_numpy(dtype=np.int64),
        calibration_meta["stacker_prob_hnm"].to_numpy(dtype=np.float32),
        metric_name=args.optimize_metric,
    )
    final_scored = add_predictions(final_meta, threshold)
    calibration_scored = add_predictions(calibration_meta, threshold)

    final_results = evaluate_scopes(final_scored, threshold)
    baseline_final_results = evaluate_scopes(
        final_df.rename(columns={"stacker_prob": "stacker_prob_hnm"}).copy(),
        baseline_report["threshold"],
    )

    error_counts = (
        final_scored.groupby(["dataset", "source_subset", "error_type_hnm"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for column in ["correct", "false_positive", "false_negative"]:
        if column not in error_counts.columns:
            error_counts[column] = 0
    error_counts["n"] = error_counts[["correct", "false_positive", "false_negative"]].sum(axis=1)
    error_counts["error_rate"] = (error_counts["false_positive"] + error_counts["false_negative"]) / error_counts["n"]

    top_fp = final_scored[final_scored["error_type_hnm"] == "false_positive"].sort_values("stacker_prob_hnm", ascending=False).head(30)
    top_fn = final_scored[final_scored["error_type_hnm"] == "false_negative"].sort_values("stacker_prob_hnm", ascending=True).head(30)

    joblib.dump(
        {
            "model": lgbm_model,
            "feature_names": feature_names,
            "hard_negative_weight": args.hard_negative_weight,
        },
        models_dir / "lgbm_hnm.joblib",
    )
    torch.save(
        {
            "model_state_dict": {key: value.detach().cpu() for key, value in lstm_model.state_dict().items()},
            **lstm_meta,
        },
        models_dir / "lstm_hnm.pt",
    )
    joblib.dump(
        {
            "model": stacker,
            "meta_features": META_FEATURES,
            "threshold": threshold,
            "stacker_c": args.stacker_c,
        },
        models_dir / "stacker_hnm.joblib",
    )

    calibration_scored.to_csv(output_dir / "hnm_calibration_predictions.csv", index=False, encoding="utf-8")
    final_scored.to_csv(output_dir / "hnm_final_test_predictions.csv", index=False, encoding="utf-8")
    pd.DataFrame(metrics_rows(final_results)).to_csv(output_dir / "hnm_metrics_by_scope.csv", index=False, encoding="utf-8")
    error_counts.to_csv(output_dir / "hnm_error_counts.csv", index=False, encoding="utf-8")
    top_fp.to_csv(output_dir / "hnm_top_false_positives.csv", index=False, encoding="utf-8")
    top_fn.to_csv(output_dir / "hnm_top_false_negatives.csv", index=False, encoding="utf-8")

    report = {
        "pipeline_dir": str(args.pipeline_dir),
        "report_dir": str(args.report_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "n_original_train": int(len(train_table)),
        "n_hard_negatives": int(len(hard_negatives)),
        "n_hard_negatives_matched_lgbm": int(len(hard_table)),
        "n_hard_negatives_matched_lstm": int(len(hard_sequences)),
        "hard_negative_weight": float(args.hard_negative_weight),
        "hard_negative_repeats": int(args.hard_negative_repeats),
        "mining_threshold": mining_threshold,
        "mining_source": args.mining_source,
        "mining_probability_column": mining_probability_column,
        "threshold": threshold,
        "optimize_metric": args.optimize_metric,
        "n_calibration": int(len(calibration_scored)),
        "n_final_test": int(len(final_scored)),
        "calibration_metrics": calibration_metrics,
        "final_test": final_results,
        "baseline_final_test": baseline_final_results,
        "lstm": lstm_meta,
        "error_counts": error_counts.to_dict(orient="records"),
    }
    report = json_safe(report)
    (output_dir / "hnm_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_summary_md(output_dir, report)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain LGBM/LSTM/stacker with hard negative mining.")
    parser.add_argument("--pipeline-dir", type=Path, default=Path("output") / "pipeline_25fps")
    parser.add_argument("--report-dir", type=Path, default=Path("output") / "pipeline_25fps" / "stacker_report")
    parser.add_argument("--output-dir", type=Path, default=Path("output") / "pipeline_25fps" / "hard_negative_mining")
    parser.add_argument("--mining-source", choices=["baseline", "previous_hnm"], default="baseline")
    parser.add_argument("--hnm-source-dir", type=Path, default=Path("output") / "pipeline_25fps" / "hard_negative_mining")
    parser.add_argument("--mining-threshold", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-hard-negatives", type=int, default=0, help="0 means all mined hard negatives.")
    parser.add_argument("--hard-negative-weight", type=float, default=4.0)
    parser.add_argument("--hard-negative-repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lgbm-estimators", type=int, default=400)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.03)
    parser.add_argument("--stacker-c", type=float, default=0.01)
    parser.add_argument("--optimize-metric", choices=["f1", "recall", "precision", "accuracy"], default="f1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_hard_negatives <= 0:
        args.max_hard_negatives = None
    return args


def main() -> None:
    args = parse_args()
    report = run_hard_negative_mining(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
