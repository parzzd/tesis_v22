from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from train_validate_pose_pipeline import LSTMAggressionClassifier, classification_metrics, resolve_torch_device


META_FEATURES = ["lgbm_prob", "lstm_prob"]


def load_lgbm_bundle(pipeline_dir: Path) -> dict[str, Any]:
    path = pipeline_dir / "models" / "lgbm_classifier.joblib"
    if not path.exists():
        raise FileNotFoundError(f"LightGBM model not found: {path}")
    return joblib.load(path)


def load_lstm_model(pipeline_dir: Path, device: torch.device) -> tuple[LSTMAggressionClassifier, dict[str, Any]]:
    path = pipeline_dir / "models" / "lstm_classifier.pt"
    if not path.exists():
        raise FileNotFoundError(f"LSTM model not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = LSTMAggressionClassifier(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_lgbm_table(pipeline_dir: Path, split: str, feature_names: list[str]) -> pd.DataFrame:
    path = pipeline_dir / f"lgbm_{split}_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"LightGBM feature table not found: {path}")
    table = pd.read_csv(path)
    missing = [name for name in feature_names if name not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing feature columns: {missing[:5]}")
    return table


def load_lstm_sequences(pipeline_dir: Path, split: str) -> dict[str, np.ndarray]:
    path = pipeline_dir / f"lstm_{split}_sequences.npz"
    if not path.exists():
        raise FileNotFoundError(f"LSTM sequence file not found: {path}")
    data = np.load(path, allow_pickle=True)
    return {
        "sequences": np.asarray(data["sequences"], dtype=np.float32),
        "labels": np.asarray(data["labels"], dtype=np.int64),
        "video_ids": np.asarray(data["video_ids"]).astype(str),
    }


def predict_lgbm(bundle: dict[str, Any], table: pd.DataFrame) -> pd.DataFrame:
    feature_names = list(bundle["feature_names"])
    probs = bundle["model"].predict_proba(table[feature_names])[:, 1].astype(np.float32)
    return pd.DataFrame(
        {
            "video_id": table["video_id"].astype(str),
            "label_id": table["label_id"].astype(int),
            "lgbm_prob": probs,
        }
    )


def normalize_for_lstm(sequences: np.ndarray, checkpoint: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return ((sequences - mean) / std).astype(np.float32)


def predict_lstm(
    model: LSTMAggressionClassifier,
    checkpoint: dict[str, Any],
    sequence_data: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    sequences = normalize_for_lstm(sequence_data["sequences"], checkpoint)
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = torch.from_numpy(sequences[start : start + batch_size]).to(device)
            logits = model(batch)
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())

    probs = np.concatenate(probabilities).astype(np.float32) if probabilities else np.empty((0,), dtype=np.float32)
    return pd.DataFrame(
        {
            "video_id": sequence_data["video_ids"],
            "label_id": sequence_data["labels"].astype(int),
            "lstm_prob": probs,
        }
    )


def build_meta_table(lgbm_probs: pd.DataFrame, lstm_probs: pd.DataFrame) -> pd.DataFrame:
    merged = lgbm_probs.merge(lstm_probs, on="video_id", suffixes=("_lgbm", "_lstm"), validate="one_to_one")
    mismatched = merged[merged["label_id_lgbm"] != merged["label_id_lstm"]]
    if not mismatched.empty:
        raise ValueError("Label mismatch between LGBM and LSTM probability tables.")
    return merged.rename(columns={"label_id_lgbm": "label_id"})[
        ["video_id", "label_id", "lgbm_prob", "lstm_prob"]
    ].sort_values("video_id")


def save_probability_table(path: Path, table: pd.DataFrame, stacker_probs: np.ndarray) -> None:
    output = table.copy()
    output["stacker_prob"] = stacker_probs.astype(np.float32)
    output.to_csv(path, index=False, encoding="utf-8")


def train_stacker(
    pipeline_dir: Path,
    device_name: str,
    batch_size: int,
    class_weight: str | None,
    c_value: float,
    threshold: float,
    seed: int,
) -> dict[str, Any]:
    device = resolve_torch_device(device_name)
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False
        print(f"[CUDA] Usando GPU para inferencia LSTM: {torch.cuda.get_device_name(device)}")

    lgbm_bundle = load_lgbm_bundle(pipeline_dir)
    lstm_model, lstm_checkpoint = load_lstm_model(pipeline_dir, device)

    train_lgbm_table = load_lgbm_table(pipeline_dir, "train", list(lgbm_bundle["feature_names"]))
    val_lgbm_table = load_lgbm_table(pipeline_dir, "val", list(lgbm_bundle["feature_names"]))
    train_lstm_sequences = load_lstm_sequences(pipeline_dir, "train")
    val_lstm_sequences = load_lstm_sequences(pipeline_dir, "val")

    train_meta = build_meta_table(
        predict_lgbm(lgbm_bundle, train_lgbm_table),
        predict_lstm(lstm_model, lstm_checkpoint, train_lstm_sequences, device, batch_size),
    )
    val_meta = build_meta_table(
        predict_lgbm(lgbm_bundle, val_lgbm_table),
        predict_lstm(lstm_model, lstm_checkpoint, val_lstm_sequences, device, batch_size),
    )

    stacker = LogisticRegression(
        C=c_value,
        max_iter=1000,
        class_weight=class_weight,
        random_state=seed,
    )
    stacker.fit(train_meta[META_FEATURES], train_meta["label_id"])

    train_prob = stacker.predict_proba(train_meta[META_FEATURES])[:, 1]
    val_prob = stacker.predict_proba(val_meta[META_FEATURES])[:, 1]

    train_metrics = classification_metrics(train_meta["label_id"].to_numpy(), train_prob, threshold=threshold)
    val_metrics = classification_metrics(val_meta["label_id"].to_numpy(), val_prob, threshold=threshold)
    lgbm_val_metrics = classification_metrics(val_meta["label_id"].to_numpy(), val_meta["lgbm_prob"].to_numpy())
    lstm_val_metrics = classification_metrics(val_meta["label_id"].to_numpy(), val_meta["lstm_prob"].to_numpy())

    model_path = pipeline_dir / "models" / "stacker_logistic.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": stacker,
            "meta_features": META_FEATURES,
            "threshold": threshold,
            "class_weight": class_weight,
            "metrics_val": val_metrics,
            "base_metrics_val": {
                "lgbm": lgbm_val_metrics,
                "lstm": lstm_val_metrics,
            },
        },
        model_path,
    )

    save_probability_table(pipeline_dir / "stacker_train_predictions.csv", train_meta, train_prob)
    save_probability_table(pipeline_dir / "stacker_val_predictions.csv", val_meta, val_prob)

    summary = {
        "model": "LogisticRegressionStacker",
        "pipeline_dir": str(pipeline_dir),
        "device": str(device),
        "meta_features": META_FEATURES,
        "threshold": threshold,
        "class_weight": class_weight,
        "c_value": c_value,
        "n_train_videos": int(len(train_meta)),
        "n_val_videos": int(len(val_meta)),
        "coefficients": stacker.coef_[0].astype(float).tolist(),
        "intercept": float(stacker.intercept_[0]),
        "train": train_metrics,
        "val": val_metrics,
        "base_val": {
            "lgbm": lgbm_val_metrics,
            "lstm": lstm_val_metrics,
        },
    }
    (pipeline_dir / "metrics_stacker.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a stacker using LGBM and LSTM probabilities.")
    parser.add_argument("--pipeline-dir", type=Path, default=Path("output") / "pipeline_25fps")
    parser.add_argument("--device", default="cuda", help='Device for LSTM inference, e.g. "cuda" or "cpu".')
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--c", type=float, default=1.0, help="Inverse regularization strength for LogisticRegression.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_weight = None if args.class_weight == "none" else args.class_weight
    summary = train_stacker(
        pipeline_dir=args.pipeline_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        class_weight=class_weight,
        c_value=args.c,
        threshold=args.threshold,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
