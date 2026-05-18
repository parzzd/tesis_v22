from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from preprocess_cctv_pose import FRAME_FEATURE_COLUMNS, VIDEO_EXTENSIONS, feature_columns, process_video


LABEL_TO_ID = {"NonFight": 0, "Fight": 1}
DEFAULT_OUTPUT_DIR = Path("output") / "pipeline_25fps"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_videos(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS],
        key=lambda path: str(path).lower(),
    )


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._") or "item"


def stable_video_id(dataset: str, split: str, label: str, path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{safe_name(dataset)}_{safe_name(split)}_{safe_name(label)}_{digest}"


def build_manifest(rwf_dir: Path, vioperu_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    def add_folder(dataset: str, source_subset: str, split: str, label: str, folder: Path) -> None:
        for video_path in discover_videos(folder):
            records.append(
                {
                    "video_id": stable_video_id(dataset, split, label, video_path),
                    "dataset": dataset,
                    "source_subset": source_subset,
                    "split": split,
                    "label": label,
                    "label_id": LABEL_TO_ID[label],
                    "video_path": str(video_path),
                }
            )

    for split in ("train", "val"):
        for label in ("Fight", "NonFight"):
            add_folder("RWF-2000", split, split, label, rwf_dir / split / label)
            add_folder("VioPeru", split, split, label, vioperu_dir / split / label)

    add_folder(
        "VioPeru",
        "false_positives_validation",
        "val",
        "NonFight",
        vioperu_dir / "false_positives_validation",
    )

    manifest = pd.DataFrame.from_records(records)
    if manifest.empty:
        raise RuntimeError("No videos were found in RWF-2000 or VioPeru-main.")
    return manifest.sort_values(["dataset", "source_subset", "split", "label", "video_path"]).reset_index(drop=True)


def sample_manifest(manifest: pd.DataFrame, sample_per_group: int, seed: int) -> pd.DataFrame:
    if sample_per_group <= 0:
        return manifest

    sampled: list[pd.DataFrame] = []
    groups = ["dataset", "source_subset", "split", "label"]
    for _, group in manifest.groupby(groups, sort=False, dropna=False):
        sampled.append(group.sample(n=min(sample_per_group, len(group)), random_state=seed))
    return pd.concat(sampled, ignore_index=True).sort_values(groups + ["video_path"]).reset_index(drop=True)


def feature_paths(output_dir: Path, record: dict[str, Any]) -> tuple[Path, Path]:
    folder = (
        output_dir
        / "features"
        / safe_name(str(record["dataset"]))
        / safe_name(str(record["source_subset"]))
        / safe_name(str(record["split"]))
        / safe_name(str(record["label"]))
    )
    stem = f"{record['video_id']}_{safe_name(Path(str(record['video_path'])).stem)}"
    return folder / f"{stem}.csv", folder / f"{stem}.npz"


def extract_features_for_manifest(
    manifest: pd.DataFrame,
    output_dir: Path,
    model_path: Path,
    target_fps: float,
    imgsz: int,
    conf: float,
    mode: str,
    tile_size: int | tuple[int, int],
    overlap: float,
    device: str | None,
    enhance_low_light: bool,
    overwrite: bool,
    feature_version: str,
    use_tracking: bool,
) -> pd.DataFrame:
    from ultralytics import YOLO

    yolo_model = YOLO(str(model_path))
    rows: list[dict[str, Any]] = []

    for record in tqdm(manifest.to_dict("records"), desc="Extrayendo features YOLO-pose"):
        csv_path, npz_path = feature_paths(output_dir, record)
        row = {**record, "feature_csv": str(csv_path), "feature_npz": str(npz_path)}

        if csv_path.exists() and npz_path.exists() and not overwrite and feature_file_matches(npz_path, feature_version):
            rows.append({**row, "status": "ok", "error": "", "processed_frames": None, "failed_frames": None})
            continue

        try:
            summary = process_video(
                input_path=record["video_path"],
                model_path=yolo_model,
                features_csv=csv_path,
                npz_output=npz_path,
                target_fps=target_fps,
                imgsz=imgsz,
                conf=conf,
                mode=mode,
                tile_size=tile_size,
                overlap=overlap,
                enhance_low_light=enhance_low_light,
                draw_results=False,
                device=device,
                feature_version=feature_version,
                use_tracking=use_tracking,
                save_csv=True,
                save_npz=True,
                save_json=False,
                save_video=False,
            )
            rows.append(
                {
                    **row,
                    "status": "ok",
                    "error": "",
                    "processed_frames": summary["processed_frames"],
                    "failed_frames": summary["failed_frames"],
                }
            )
        except Exception as exc:
            rows.append({**row, "status": "failed", "error": str(exc), "processed_frames": 0, "failed_frames": 0})

    return pd.DataFrame.from_records(rows)


def feature_file_matches(npz_path: str | Path, feature_version: str) -> bool:
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            saved_version = str(data["feature_version"]) if "feature_version" in data else "v1"
            saved_features = [str(item) for item in data["feature_names"]] if "feature_names" in data else FRAME_FEATURE_COLUMNS
        return saved_version == feature_version and saved_features == feature_columns(feature_version)
    except Exception:
        return False


def load_sequence(npz_path: str | Path, expected_feature_names: list[str]) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as data:
        sequence = np.asarray(data["sequence"], dtype=np.float32)
        saved_feature_names = [str(item) for item in data["feature_names"]] if "feature_names" in data else list(FRAME_FEATURE_COLUMNS)
    if sequence.ndim != 2:
        sequence = np.empty((0, len(saved_feature_names)), dtype=np.float32)
    aligned = np.zeros((sequence.shape[0], len(expected_feature_names)), dtype=np.float32)
    saved_index = {name: idx for idx, name in enumerate(saved_feature_names)}
    for out_idx, name in enumerate(expected_feature_names):
        in_idx = saved_index.get(name)
        if in_idx is not None and in_idx < sequence.shape[1]:
            aligned[:, out_idx] = sequence[:, in_idx]
    return np.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0)


def resample_sequence(sequence: np.ndarray, sequence_length: int, feature_count: int | None = None) -> np.ndarray:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than 0.")
    if len(sequence) == 0:
        return np.zeros((sequence_length, int(feature_count or len(FRAME_FEATURE_COLUMNS))), dtype=np.float32)
    if len(sequence) == sequence_length:
        return sequence.astype(np.float32, copy=False)
    indices = np.linspace(0, len(sequence) - 1, sequence_length).round().astype(np.int64)
    return sequence[indices].astype(np.float32, copy=False)


def aggregate_sequence(sequence: np.ndarray, feature_count: int | None = None) -> np.ndarray:
    if len(sequence) == 0:
        return np.zeros(int(feature_count or len(FRAME_FEATURE_COLUMNS)) * 6, dtype=np.float32)
    sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
    values = [
        sequence.mean(axis=0),
        sequence.std(axis=0),
        sequence.min(axis=0),
        sequence.max(axis=0),
        sequence[-1],
        sequence[-1] - sequence[0],
    ]
    return np.concatenate(values).astype(np.float32)


def aggregate_feature_names(frame_feature_names: list[str] | None = None) -> list[str]:
    features = frame_feature_names or list(FRAME_FEATURE_COLUMNS)
    stats = ("mean", "std", "min", "max", "last", "delta")
    return [f"{feature}_{stat}" for stat in stats for feature in features]


def build_split_arrays(
    feature_manifest: pd.DataFrame,
    split: str,
    sequence_length: int,
    frame_feature_names: list[str] | None = None,
) -> dict[str, Any]:
    frame_feature_names = frame_feature_names or list(FRAME_FEATURE_COLUMNS)
    split_df = feature_manifest[(feature_manifest["split"] == split) & (feature_manifest["status"] == "ok")].copy()
    split_df = split_df[split_df["feature_npz"].map(lambda path: Path(str(path)).exists())]
    if split_df.empty:
        raise RuntimeError(f"No valid feature files were found for split={split}.")

    sequences: list[np.ndarray] = []
    aggregates: list[np.ndarray] = []
    labels: list[int] = []
    meta_rows: list[dict[str, Any]] = []

    for record in split_df.to_dict("records"):
        raw_sequence = load_sequence(record["feature_npz"], frame_feature_names)
        sequences.append(resample_sequence(raw_sequence, sequence_length, len(frame_feature_names)))
        aggregates.append(aggregate_sequence(raw_sequence, len(frame_feature_names)))
        labels.append(int(record["label_id"]))
        meta_rows.append(
            {
                "video_id": record["video_id"],
                "dataset": record["dataset"],
                "source_subset": record["source_subset"],
                "split": record["split"],
                "label": record["label"],
                "label_id": int(record["label_id"]),
                "video_path": record["video_path"],
                "feature_npz": record["feature_npz"],
            }
        )

    return {
        "X_seq": np.stack(sequences).astype(np.float32),
        "X_agg": np.stack(aggregates).astype(np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "meta": pd.DataFrame(meta_rows),
        "feature_names": list(frame_feature_names),
    }


def save_training_tables(output_dir: Path, split_name: str, split_data: dict[str, Any]) -> None:
    frame_feature_names = list(split_data.get("feature_names", FRAME_FEATURE_COLUMNS))
    agg_names = aggregate_feature_names(frame_feature_names)
    table = pd.concat(
        [
            split_data["meta"].reset_index(drop=True),
            pd.DataFrame(split_data["X_agg"], columns=agg_names),
        ],
        axis=1,
    )
    table.to_csv(output_dir / f"lgbm_{split_name}_table.csv", index=False, encoding="utf-8")
    np.savez_compressed(
        output_dir / f"lstm_{split_name}_sequences.npz",
        sequences=split_data["X_seq"],
        labels=split_data["y"],
        video_ids=split_data["meta"]["video_id"].to_numpy(),
        feature_names=np.asarray(frame_feature_names),
    )


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(np.int64)

    metrics: dict[str, Any] = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix_labels": ["NonFight", "Fight"],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else None
    return metrics


def train_lgbm(
    train_data: dict[str, Any],
    val_data: dict[str, Any],
    output_dir: Path,
    estimators: int,
    learning_rate: float,
    requested_device: str,
    seed: int,
) -> dict[str, Any]:
    if len(np.unique(train_data["y"])) < 2:
        raise RuntimeError("LightGBM needs both classes in the training split.")

    frame_feature_names = list(train_data.get("feature_names", FRAME_FEATURE_COLUMNS))
    feature_names = aggregate_feature_names(frame_feature_names)
    X_train = pd.DataFrame(train_data["X_agg"], columns=feature_names)
    X_val = pd.DataFrame(val_data["X_agg"], columns=feature_names)

    def make_model(device_type: str) -> LGBMClassifier:
        params: dict[str, Any] = {
            "objective": "binary",
            "n_estimators": estimators,
            "learning_rate": learning_rate,
            "num_leaves": 31,
            "min_child_samples": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "class_weight": "balanced",
            "random_state": seed,
            "n_jobs": -1,
            "verbosity": -1,
        }
        if device_type != "cpu":
            params["device_type"] = device_type
        return LGBMClassifier(**params)

    actual_device = requested_device
    model = make_model(actual_device)
    try:
        model.fit(X_train, train_data["y"])
    except Exception as exc:
        if requested_device == "cpu":
            raise
        print(f"[WARN] LightGBM no pudo usar device_type={requested_device}: {exc}")
        print("[WARN] Reentrenando LightGBM en CPU. YOLO y LSTM siguen usando CUDA.")
        actual_device = "cpu_fallback"
        model = make_model("cpu")
        model.fit(X_train, train_data["y"])

    val_prob = model.predict_proba(X_val)[:, 1]
    metrics = classification_metrics(val_data["y"], val_prob)
    metrics.update(
        {
            "model": "LGBMClassifier",
            "requested_device": requested_device,
            "actual_device": actual_device,
            "n_train_videos": int(len(train_data["y"])),
            "n_val_videos": int(len(val_data["y"])),
            "feature_count": int(train_data["X_agg"].shape[1]),
        }
    )

    model_path = output_dir / "models" / "lgbm_classifier.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "frame_feature_names": frame_feature_names,
            "metrics": metrics,
        },
        model_path,
    )
    (output_dir / "metrics_lgbm.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


class LSTMAggressionClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        last_forward = hidden[-2]
        last_backward = hidden[-1]
        return self.head(torch.cat([last_forward, last_backward], dim=1)).squeeze(1)


def resolve_torch_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested in {"auto", "cuda"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "cuda":
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        return torch.device("cpu")
    if requested.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {requested} was requested but CUDA is not available.")
        return torch.device(f"cuda:{requested}")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{requested} was requested but CUDA is not available.")
        return torch.device(requested)
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {requested}")


def normalize_sequences(train_seq: np.ndarray, val_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat = train_seq.reshape(-1, train_seq.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return ((train_seq - mean) / std).astype(np.float32), ((val_seq - mean) / std).astype(np.float32), mean, std


def evaluate_lstm(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    criterion: nn.Module | None = None,
    batch_size: int = 64,
) -> tuple[dict[str, Any], float | None]:
    model.eval()
    probabilities: list[np.ndarray] = []
    losses: list[float] = []
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))), batch_size=batch_size)
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            if criterion is not None:
                losses.append(float(criterion(logits, yb).detach().cpu()))
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
    y_prob = np.concatenate(probabilities) if probabilities else np.empty((0,), dtype=np.float32)
    return classification_metrics(y, y_prob), float(np.mean(losses)) if losses else None


def train_lstm(
    train_data: dict[str, Any],
    val_data: dict[str, Any],
    output_dir: Path,
    device_request: str,
    target_fps: float,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    patience: int,
) -> dict[str, Any]:
    if len(np.unique(train_data["y"])) < 2:
        raise RuntimeError("LSTM needs both classes in the training split.")

    device = resolve_torch_device(device_request)
    frame_feature_names = list(train_data.get("feature_names", FRAME_FEATURE_COLUMNS))
    if device.type == "cuda":
        # On some Windows CUDA builds, cuDNN-backed LSTMs can finish training
        # correctly but crash during interpreter shutdown. Keep CUDA enabled
        # while using the non-cuDNN LSTM path for a cleaner pipeline exit.
        torch.backends.cudnn.enabled = False
    train_seq, val_seq, feature_mean, feature_std = normalize_sequences(train_data["X_seq"], val_data["X_seq"])

    model = LSTMAggressionClassifier(
        input_size=train_seq.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    y_train = train_data["y"].astype(np.float32)
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_seq), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    best_score = -1.0
    best_epoch = 0
    stagnant_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_metrics, val_loss = evaluate_lstm(model, val_seq, val_data["y"], device, criterion, batch_size=batch_size)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else None,
            "val_loss": val_loss,
            "val_f1": val_metrics["f1"],
            "val_recall": val_metrics["recall"],
            "val_precision": val_metrics["precision"],
        }
        history.append(epoch_summary)
        print(
            f"[LSTM] epoch {epoch:03d}/{epochs} "
            f"loss={epoch_summary['train_loss']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_recall={val_metrics['recall']:.4f}"
        )

        score = float(val_metrics["f1"])
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_metrics = val_metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stagnant_epochs = 0
        else:
            stagnant_epochs += 1
            if stagnant_epochs >= patience:
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("LSTM training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    final_metrics, final_loss = evaluate_lstm(model, val_seq, val_data["y"], device, criterion, batch_size=batch_size)
    final_metrics.update(
        {
            "model": "LSTM",
            "device": str(device),
            "best_epoch": best_epoch,
            "best_val_f1": best_score,
            "final_val_loss": final_loss,
            "n_train_videos": int(len(train_data["y"])),
            "n_val_videos": int(len(val_data["y"])),
            "sequence_length": int(sequence_length),
            "target_fps": float(target_fps),
            "feature_count": int(train_seq.shape[-1]),
            "history": history,
        }
    )

    model_path = output_dir / "models" / "lstm_classifier.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "input_size": int(train_seq.shape[-1]),
            "hidden_size": int(hidden_size),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "feature_names": frame_feature_names,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "sequence_length": int(sequence_length),
            "target_fps": float(target_fps),
            "metrics": final_metrics,
        },
        model_path,
    )
    (output_dir / "metrics_lstm.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return final_metrics


def _parse_tile_size(value: str) -> int | tuple[int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise argparse.ArgumentTypeError("--tile-size must be N or W,H")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and validate the YOLOv11s-pose + LSTM + LGBM aggression detection pipeline."
    )
    parser.add_argument("--rwf-dir", type=Path, default=Path("RWF-2000"), help="RWF-2000 dataset directory.")
    parser.add_argument("--vioperu-dir", type=Path, default=Path("VioPeru-main"), help="VioPeru-main dataset directory.")
    parser.add_argument("--model", type=Path, default=Path("yolo11s-pose.pt"), help="YOLO pose model path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Pipeline output directory.")
    parser.add_argument("--target-fps", type=float, default=25.0, help="Effective FPS for feature extraction.")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO pose inference size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--mode", choices=["full_frame", "tiles"], default="full_frame", help="Pose extraction mode.")
    parser.add_argument("--tile-size", type=_parse_tile_size, default=960, help="Tile size as N or W,H.")
    parser.add_argument("--overlap", type=float, default=0.20, help="Tile overlap ratio or pixels if >= 1.")
    parser.add_argument("--device", default="cuda", help='CUDA/CPU device for YOLO and LSTM, e.g. "cuda", "cuda:0", "0", "cpu".')
    parser.add_argument("--lgbm-device", choices=["cpu", "gpu", "cuda"], default="cpu", help="LightGBM device_type; use cuda only with a CUDA-enabled LightGBM build.")
    parser.add_argument("--enhance-low-light", action="store_true", help="Apply low-light enhancement before pose extraction.")
    parser.add_argument("--feature-version", choices=["v1", "v2"], default="v1", help="Frame feature schema. v1 matches current production models; v2 adds interaction features.")
    parser.add_argument("--use-tracking", action="store_true", help="Use Ultralytics ByteTrack IDs during full_frame feature extraction.")
    parser.add_argument("--overwrite-features", action="store_true", help="Recompute feature files even if they already exist.")
    parser.add_argument("--skip-extraction", action="store_true", help="Reuse output feature_manifest.csv and train only.")
    parser.add_argument("--sample-per-group", type=int, default=0, help="Small debug sample per dataset/split/label group.")
    parser.add_argument("--sequence-length", type=int, default=125, help="Fixed LSTM sequence length; 125 = 5 seconds at 25 FPS.")
    parser.add_argument("--epochs", type=int, default=15, help="LSTM training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="LSTM batch size.")
    parser.add_argument("--hidden-size", type=int, default=96, help="LSTM hidden size.")
    parser.add_argument("--num-layers", type=int, default=2, help="LSTM layer count.")
    parser.add_argument("--dropout", type=float, default=0.25, help="LSTM dropout.")
    parser.add_argument("--lr", type=float, default=1e-3, help="LSTM learning rate.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--lgbm-estimators", type=int, default=400, help="LightGBM number of trees.")
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.03, help="LightGBM learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch_device = resolve_torch_device(args.device)
    if torch_device.type == "cuda":
        print(f"[CUDA] Usando GPU: {torch.cuda.get_device_name(torch_device)}")
    else:
        print("[WARN] El pipeline esta usando CPU. Para esta tesis se recomienda --device cuda.")

    if not args.model.exists():
        raise FileNotFoundError(f"YOLO pose model not found: {args.model}")

    if args.skip_extraction:
        feature_manifest_path = args.output_dir / "feature_manifest.csv"
        if not feature_manifest_path.exists():
            raise FileNotFoundError(f"--skip-extraction requires {feature_manifest_path}")
        feature_manifest = pd.read_csv(feature_manifest_path)
    else:
        manifest = build_manifest(args.rwf_dir, args.vioperu_dir)
        manifest = sample_manifest(manifest, args.sample_per_group, args.seed)
        manifest.to_csv(args.output_dir / "manifest.csv", index=False, encoding="utf-8")
        print(f"[DATA] Videos en manifiesto: {len(manifest)}")
        print(manifest.groupby(["split", "label"]).size().to_string())

        yolo_device = None if str(args.device).lower() == "auto" else str(args.device)
        feature_manifest = extract_features_for_manifest(
            manifest=manifest,
            output_dir=args.output_dir,
            model_path=args.model,
            target_fps=args.target_fps,
            imgsz=args.imgsz,
            conf=args.conf,
            mode=args.mode,
            tile_size=args.tile_size,
            overlap=args.overlap,
            device=yolo_device,
            enhance_low_light=args.enhance_low_light,
            overwrite=args.overwrite_features,
            feature_version=args.feature_version,
            use_tracking=args.use_tracking,
        )
        feature_manifest.to_csv(args.output_dir / "feature_manifest.csv", index=False, encoding="utf-8")

    ok_count = int((feature_manifest["status"] == "ok").sum())
    failed_count = int((feature_manifest["status"] != "ok").sum())
    print(f"[FEATURES] OK={ok_count} fallidos={failed_count}")
    if ok_count == 0:
        raise RuntimeError("No valid features were extracted.")

    frame_feature_names = feature_columns(args.feature_version)
    train_data = build_split_arrays(feature_manifest, "train", args.sequence_length, frame_feature_names)
    val_data = build_split_arrays(feature_manifest, "val", args.sequence_length, frame_feature_names)
    save_training_tables(args.output_dir, "train", train_data)
    save_training_tables(args.output_dir, "val", val_data)

    print(f"[TRAIN] videos train={len(train_data['y'])} val={len(val_data['y'])} fps={args.target_fps}")
    lgbm_metrics = train_lgbm(
        train_data=train_data,
        val_data=val_data,
        output_dir=args.output_dir,
        estimators=args.lgbm_estimators,
        learning_rate=args.lgbm_learning_rate,
        requested_device=args.lgbm_device,
        seed=args.seed,
    )
    lstm_metrics = train_lstm(
        train_data=train_data,
        val_data=val_data,
        output_dir=args.output_dir,
        device_request=args.device,
        target_fps=args.target_fps,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
    )

    summary = {
        "target_fps": float(args.target_fps),
        "imgsz": int(args.imgsz),
        "mode": args.mode,
        "feature_version": args.feature_version,
        "use_tracking": bool(args.use_tracking),
        "device": str(torch_device),
        "n_features_ok": ok_count,
        "n_features_failed": failed_count,
        "train_videos": int(len(train_data["y"])),
        "val_videos": int(len(val_data["y"])),
        "lgbm": lgbm_metrics,
        "lstm": lstm_metrics,
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
