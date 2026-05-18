from __future__ import annotations

import asyncio
import base64
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import joblib
import numpy as np
import pandas as pd
import torch

from app.model_profiles import MODEL_PROFILES, ROOT
from preprocess_cctv_pose import (
    FRAME_FEATURE_COLUMNS,
    _draw_detections,
    extract_frame_tabular_features,
    feature_columns,
    run_yolo_pose_on_frame,
)
from train_validate_pose_pipeline import LSTMAggressionClassifier, aggregate_sequence, resample_sequence


SESSION_STATUS: dict[str, dict[str, Any]] = {}


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


class HNMInferenceService:
    def __init__(self) -> None:
        self._pose_models: dict[str, Any] = {}
        self._bundles: dict[str, dict[str, Any]] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.backends.cudnn.enabled = False

    def _load_pose(self, pose_model: str) -> Any:
        if pose_model not in self._pose_models:
            from ultralytics import YOLO

            model_path = resolve_path(pose_model)
            if not model_path.exists():
                raise FileNotFoundError(f"Pose model not found: {model_path}")
            self._pose_models[pose_model] = YOLO(str(model_path))
        return self._pose_models[pose_model]

    def _load_bundle(self, profile_id: str) -> dict[str, Any]:
        if profile_id in self._bundles:
            return self._bundles[profile_id]
        if profile_id not in MODEL_PROFILES:
            raise KeyError(f"Unknown model profile: {profile_id}")

        profile = MODEL_PROFILES[profile_id]
        lgbm_bundle = joblib.load(resolve_path(profile["lgbm_model"]))
        stacker_bundle = joblib.load(resolve_path(profile["stacker_model"]))
        lstm_checkpoint = torch.load(resolve_path(profile["lstm_model"]), map_location=self.device, weights_only=False)
        lstm_model = LSTMAggressionClassifier(
            input_size=int(lstm_checkpoint["input_size"]),
            hidden_size=int(lstm_checkpoint["hidden_size"]),
            num_layers=int(lstm_checkpoint["num_layers"]),
            dropout=float(lstm_checkpoint["dropout"]),
        ).to(self.device)
        lstm_model.load_state_dict(lstm_checkpoint["model_state_dict"])
        lstm_model.eval()

        bundle = {
            "lgbm_model": lgbm_bundle["model"],
            "lgbm_feature_names": list(lgbm_bundle["feature_names"]),
            "stacker_model": stacker_bundle["model"],
            "stacker_features": list(stacker_bundle.get("meta_features", ["lgbm_prob", "lstm_prob"])),
            "lstm_model": lstm_model,
            "lstm_mean": np.asarray(lstm_checkpoint["feature_mean"], dtype=np.float32),
            "lstm_std": np.asarray(lstm_checkpoint["feature_std"], dtype=np.float32),
            "sequence_length": int(lstm_checkpoint.get("sequence_length", 125)),
        }
        bundle["lstm_std"] = np.where(bundle["lstm_std"] < 1e-6, 1.0, bundle["lstm_std"]).astype(np.float32)
        self._bundles[profile_id] = bundle
        return bundle

    def extract_frame_features(
        self,
        frame: np.ndarray,
        config: dict[str, Any],
        previous_centers: np.ndarray | None,
        previous_timestamp: float | None,
        previous_frame_features: dict[str, float] | None,
        timestamp: float,
    ) -> tuple[list[float], np.ndarray, list[dict[str, Any]], np.ndarray, dict[str, float]]:
        active_feature_columns = feature_columns(str(config.get("feature_version", "v1")))
        pose_model = self._load_pose(str(config["pose_model"]))
        detections = run_yolo_pose_on_frame(
            pose_model,
            frame,
            imgsz=int(config["imgsz"]),
            conf=0.25,
            device="cuda" if self.device.type == "cuda" else "cpu",
        )
        frame_features, centers = extract_frame_tabular_features(
            detections=detections,
            frame_shape=frame.shape,
            previous_centers=previous_centers,
            previous_timestamp=previous_timestamp,
            timestamp=timestamp,
            previous_frame_features=previous_frame_features,
            output_columns=active_feature_columns,
        )
        row = [float(frame_features[name]) for name in active_feature_columns]
        return row, centers, detections, frame, frame_features

    def predict_sequence(self, sequence_rows: list[list[float]], profile_id: str) -> dict[str, float]:
        bundle = self._load_bundle(profile_id)
        sequence = np.asarray(sequence_rows, dtype=np.float32)
        if sequence.ndim != 2 or sequence.shape[0] == 0:
            sequence = np.zeros((1, len(bundle["lstm_mean"])), dtype=np.float32)

        person_count_idx = FRAME_FEATURE_COLUMNS.index("person_count")
        recent_people = sequence[-min(5, sequence.shape[0]) :, person_count_idx]
        if recent_people.size == 0 or float(np.nanmax(recent_people)) < 1.0:
            return {
                "lgbm_prob": 0.0,
                "lstm_prob": 0.0,
                "stacker_prob": 0.0,
            }

        aggregated = aggregate_sequence(sequence)
        X_lgbm = pd.DataFrame([aggregated], columns=bundle["lgbm_feature_names"])
        lgbm_prob = float(bundle["lgbm_model"].predict_proba(X_lgbm)[:, 1][0])

        seq_fixed = resample_sequence(sequence, int(bundle["sequence_length"]))
        seq_norm = ((seq_fixed - bundle["lstm_mean"]) / bundle["lstm_std"]).astype(np.float32)
        with torch.no_grad():
            xb = torch.from_numpy(seq_norm[None, :, :]).to(self.device)
            lstm_logit = bundle["lstm_model"](xb)
            lstm_prob = float(torch.sigmoid(lstm_logit).detach().cpu().numpy()[0])

        X_stack = pd.DataFrame([[lgbm_prob, lstm_prob]], columns=bundle["stacker_features"])
        stacker_prob = float(bundle["stacker_model"].predict_proba(X_stack)[:, 1][0])
        return {
            "lgbm_prob": lgbm_prob,
            "lstm_prob": lstm_prob,
            "stacker_prob": stacker_prob,
        }


SERVICE = HNMInferenceService()


def source_for_opencv(src: str) -> str | int:
    stripped = str(src).strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped


def open_video_capture(src: str) -> cv2.VideoCapture:
    source = source_for_opencv(src)
    if isinstance(source, int):
        return cv2.VideoCapture(source)
    return cv2.VideoCapture(source, cv2.CAP_FFMPEG)


def encode_frame(frame: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise RuntimeError("Could not encode frame as JPEG.")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def smoothed_probability(score_window: list[float], score: float, window_size: int) -> float:
    window_size = max(int(window_size), 1)
    score_window.append(float(score))
    del score_window[:-window_size]
    return float(np.mean(score_window)) if score_window else float(score)


async def stream_camera(websocket: Any, serial_number: str, src: str, config_loader: Any) -> None:
    cap = open_video_capture(src)
    if not cap.isOpened():
        await websocket.send_json({"type": "error", "serial_number": serial_number, "error": "No se pudo abrir la fuente de video"})
        return

    SESSION_STATUS[serial_number] = {
        "running": True,
        "started_at": time.time(),
        "last_score": 0.0,
        "last_error": None,
    }
    sequence_rows: list[list[float]] = []
    score_window: list[float] = []
    positive_windows = 0
    last_alert_at = 0.0
    previous_centers: np.ndarray | None = None
    previous_timestamp: float | None = None
    previous_frame_features: dict[str, float] | None = None
    last_processed_at = 0.0

    try:
        while True:
            ok, frame = await asyncio.to_thread(cap.read)
            if not ok or frame is None:
                await websocket.send_json({"type": "error", "serial_number": serial_number, "error": "Frame no disponible"})
                break

            config = config_loader()
            fps = max(float(config.get("fps", 10.0)), 1.0)
            now = time.time()
            interval = 1.0 / fps
            if now - last_processed_at < interval:
                await asyncio.sleep(0.005)
                continue
            last_processed_at = now

            try:
                row, centers, detections, display_frame, frame_features = await asyncio.to_thread(
                    SERVICE.extract_frame_features,
                    frame,
                    config,
                    previous_centers,
                    previous_timestamp,
                    previous_frame_features,
                    now,
                )
                previous_centers = centers
                previous_timestamp = now
                previous_frame_features = frame_features
                sequence_rows.append(row)
                sequence_rows = sequence_rows[-125:]

                probs = await asyncio.to_thread(SERVICE.predict_sequence, sequence_rows, config["model_profile"])
                score = probs["stacker_prob"]
                threshold = float(config.get("threshold", 0.49))
                smoothing_windows = max(int(config.get("smoothing_windows", 3)), 1)
                p_smooth = smoothed_probability(score_window, score, smoothing_windows)

                if p_smooth >= threshold:
                    positive_windows += 1
                else:
                    positive_windows = 0

                alert_windows = max(int(config.get("alert_windows", 2)), 1)
                is_alert = positive_windows >= alert_windows
                cooldown = float(config.get("cooldown_seconds", 30.0))
                should_emit_alert = is_alert and (now - last_alert_at >= cooldown)
                if should_emit_alert:
                    last_alert_at = now

                if config.get("overlay"):
                    display_frame = _draw_detections(display_frame, detections)

                jpg_b64 = await asyncio.to_thread(encode_frame, display_frame)
                payload = {
                    "type": "frame",
                    "serial_number": serial_number,
                    "jpg_b64": jpg_b64,
                    "ts": now,
                    "p_win": score,
                    "p_vid": p_smooth,
                    "smoothed_prob": p_smooth,
                    "prob": p_smooth,
                    "on": is_alert,
                    "alert": is_alert,
                    "threshold": threshold,
                    "smoothing_windows": smoothing_windows,
                    "model_profile": config["model_profile"],
                    "fps": fps,
                    "imgsz": int(config["imgsz"]),
                    "pose_model": config["pose_model"],
                    **probs,
                }
                SESSION_STATUS[serial_number].update({"last_score": p_smooth, "last_raw_score": score, "last_error": None, "config": config})
                await websocket.send_json(payload)
                if should_emit_alert:
                    await websocket.send_json(
                        {
                            "type": "alert",
                            "serial_number": serial_number,
                            "prob": p_smooth,
                            "smoothed_prob": p_smooth,
                            "stacker_prob": score,
                            "ts": now,
                            "threshold": threshold,
                            "smoothing_windows": smoothing_windows,
                            "model_profile": config["model_profile"],
                        }
                    )
            except Exception as exc:
                SESSION_STATUS[serial_number]["last_error"] = str(exc)
                await websocket.send_json({"type": "error", "serial_number": serial_number, "error": str(exc)})
                if isinstance(exc, PermissionError):
                    break
                await asyncio.sleep(0.25)
    finally:
        cap.release()
        SESSION_STATUS[serial_number] = {
            **SESSION_STATUS.get(serial_number, {}),
            "running": False,
            "stopped_at": time.time(),
        }
