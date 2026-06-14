from __future__ import annotations

from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HNM_MODEL_DIR = ROOT / "output" / "pipeline_25fps" / "hard_negative_mining" / "models"


MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "hnm_balanced": {
        "id": "hnm_balanced",
        "name": "HNM balanceado",
        "description": "Perfil original recomendado: mayor sensibilidad para no perder agresiones.",
        "recommended": True,
        "threshold": 0.49,
        "fps": 10.0,
        "imgsz": 960,
        "pose_model": "yolo11s-pose.pt",
        "feature_version": "v1",
        "smoothing_windows": 3,
        "alert_windows": 2,
        "cooldown_seconds": 30.0,
        "stacker_model": str(HNM_MODEL_DIR / "stacker_hnm.joblib"),
        "lgbm_model": str(HNM_MODEL_DIR / "lgbm_hnm.joblib"),
        "lstm_model": str(HNM_MODEL_DIR / "lstm_hnm.pt"),
    },
    "high_accuracy": {
        "id": "high_accuracy",
        "name": "Alta precision",
        "description": "Mayor resolucion para zonas criticas; consume mas GPU.",
        "recommended": False,
        "threshold": 0.49,
        "fps": 10.0,
        "imgsz": 1280,
        "pose_model": "yolo11s-pose.pt",
        "feature_version": "v1",
        "smoothing_windows": 3,
        "alert_windows": 2,
        "cooldown_seconds": 30.0,
        "stacker_model": str(HNM_MODEL_DIR / "stacker_hnm.joblib"),
        "lgbm_model": str(HNM_MODEL_DIR / "lgbm_hnm.joblib"),
        "lstm_model": str(HNM_MODEL_DIR / "lstm_hnm.pt"),
    },
}

PUBLIC_PROFILE_FIELDS = (
    "id",
    "name",
    "description",
    "recommended",
    "threshold",
    "fps",
    "imgsz",
    "pose_model",
    "feature_version",
    "smoothing_windows",
    "alert_windows",
    "cooldown_seconds",
)


def _path_exists(path_value: str) -> bool:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path.exists()


def profile_available(profile: dict[str, Any]) -> tuple[bool, list[str]]:
    required = ["pose_model", "stacker_model", "lgbm_model", "lstm_model"]
    missing: list[str] = []
    for key in required:
        value = str(profile[key])
        if not _path_exists(value):
            missing.append(value)
    return not missing, missing


def serialize_profile(profile_id: str) -> dict[str, Any]:
    if profile_id not in MODEL_PROFILES:
        profile_id = "hnm_balanced"
    if profile_id not in MODEL_PROFILES:
        raise KeyError(profile_id)
    profile = dict(MODEL_PROFILES[profile_id])
    available, missing = profile_available(profile)
    public = {key: profile[key] for key in PUBLIC_PROFILE_FIELDS}
    public["available"] = available
    public["missing"] = [Path(item).name for item in missing]
    return public


def list_profiles() -> list[dict[str, Any]]:
    return [serialize_profile(profile_id) for profile_id in MODEL_PROFILES]


def default_config() -> dict[str, Any]:
    profile = serialize_profile("hnm_balanced")
    return {
        "model_profile": profile["id"],
        "threshold": float(profile["threshold"]),
        "fps": float(profile["fps"]),
        "imgsz": int(profile["imgsz"]),
        "pose_model": profile["pose_model"],
        "feature_version": profile["feature_version"],
        "smoothing_windows": int(profile["smoothing_windows"]),
        "alert_windows": int(profile["alert_windows"]),
        "cooldown_seconds": float(profile["cooldown_seconds"]),
        "overlay": False,
    }


def config_from_profile(profile_id: str, overlay: bool = False) -> dict[str, Any]:
    profile = serialize_profile(profile_id)
    return {
        "model_profile": profile["id"],
        "threshold": float(profile["threshold"]),
        "fps": float(profile["fps"]),
        "imgsz": int(profile["imgsz"]),
        "pose_model": profile["pose_model"],
        "feature_version": profile["feature_version"],
        "smoothing_windows": int(profile["smoothing_windows"]),
        "alert_windows": int(profile["alert_windows"]),
        "cooldown_seconds": float(profile["cooldown_seconds"]),
        "overlay": bool(overlay),
    }
