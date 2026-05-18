from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mpeg", ".mpg", ".mov", ".mkv"}
KEYPOINT_CONF_THRESHOLD = 0.20
COCO_POSE_EDGES = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]

FRAME_FEATURE_COLUMNS_V1 = [
    "person_count",
    "mean_detection_conf",
    "max_detection_conf",
    "sum_bbox_area_norm",
    "mean_bbox_area_norm",
    "max_bbox_area_norm",
    "min_pair_distance_norm",
    "mean_pair_distance_norm",
    "max_iou_between_people",
    "mean_center_speed_norm_per_sec",
    "max_center_speed_norm_per_sec",
    "visible_keypoints_total",
    "visible_keypoints_mean",
    "mean_keypoint_score",
    "min_keypoint_score",
]
FRAME_FEATURE_COLUMNS_V2_EXTRA = [
    "min_pair_distance_delta",
    "mean_pair_distance_delta",
    "max_iou_delta",
    "mean_center_accel_norm_per_sec2",
    "max_center_accel_norm_per_sec2",
    "mean_bbox_area_delta",
    "max_bbox_area_delta",
    "close_pair_count_norm",
    "overlap_pair_count_norm",
    "wrist_to_other_head_min_norm",
    "wrist_to_other_torso_min_norm",
]
FRAME_FEATURE_COLUMNS_V2 = [*FRAME_FEATURE_COLUMNS_V1, *FRAME_FEATURE_COLUMNS_V2_EXTRA]

# Backward-compatible default used by the active production model.
FRAME_FEATURE_COLUMNS = FRAME_FEATURE_COLUMNS_V1


def feature_columns(feature_version: str = "v1") -> list[str]:
    """Return the frame-level feature schema for a training/inference version."""
    version = str(feature_version or "v1").lower()
    if version == "v1":
        return list(FRAME_FEATURE_COLUMNS_V1)
    if version == "v2":
        return list(FRAME_FEATURE_COLUMNS_V2)
    raise ValueError('feature_version must be "v1" or "v2".')


def _valid_keypoint_mask(keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray:
    if keypoints.size == 0 or scores.size == 0:
        return np.zeros((0,), dtype=bool)
    return (
        (keypoints[:, 0] > 0)
        & (keypoints[:, 1] > 0)
        & (scores >= KEYPOINT_CONF_THRESHOLD)
    )


def get_video_info(video_path: str | Path) -> dict[str, Any]:
    """Read basic metadata and validate that the video can be opened."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {path.suffix}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if fps <= 0:
        raise ValueError(f"Invalid FPS reported by OpenCV: {fps}")
    if frame_count <= 0:
        raise ValueError("Invalid frame count reported by OpenCV.")
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid resolution reported by OpenCV: {width}x{height}")

    return {
        "path": str(path),
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps,
        "width": width,
        "height": height,
    }


def sample_frames_by_time(video_path: str | Path, target_fps: float) -> Iterable[tuple[int, float, np.ndarray]]:
    """Yield frames sampled by timestamp at a fixed effective FPS."""
    if target_fps <= 0:
        raise ValueError("target_fps must be greater than 0.")

    info = get_video_info(video_path)
    source_fps = float(info["fps"])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    target_interval = 1.0 / target_fps
    next_timestamp = 0.0
    frame_idx = -1

    while True:
        ok, frame = cap.read()
        frame_idx += 1
        if not ok:
            break

        timestamp = frame_idx / source_fps
        if timestamp + 1e-9 >= next_timestamp:
            yield frame_idx, timestamp, frame
            next_timestamp += target_interval

    cap.release()


def letterbox_frame(
    frame: np.ndarray,
    target_size: int | tuple[int, int],
    pad_color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize without deformation, then pad to target size."""
    if isinstance(target_size, int):
        target_w = target_h = target_size
    else:
        target_w, target_h = target_size

    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Cannot letterbox an empty frame.")

    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    out = np.full((target_h, target_w, 3), pad_color, dtype=frame.dtype)
    out[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return out, scale, (pad_x, pad_y)


def _gamma_correction(frame: np.ndarray, gamma: float = 0.75) -> np.ndarray:
    inv_gamma = 1.0 / max(gamma, 1e-6)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(frame, table)


def apply_low_light_enhancement(
    frame: np.ndarray,
    brightness_threshold: float = 85.0,
    gamma: float = 0.75,
    clahe_clip_limit: float = 2.0,
    clahe_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE + gamma only when mean luminance is low."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    mean_luma = float(np.mean(l_channel))
    if mean_luma >= brightness_threshold:
        return frame

    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)
    l_enhanced = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return _gamma_correction(enhanced, gamma=gamma)


def generate_tiles(
    frame: np.ndarray,
    tile_size: int | tuple[int, int] = 960,
    overlap: float = 0.20,
) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """Split frame into overlapping tiles and return each crop with its original coordinates."""
    if isinstance(tile_size, int):
        tile_w = tile_h = tile_size
    else:
        tile_w, tile_h = tile_size

    h, w = frame.shape[:2]
    tile_w = min(max(1, int(tile_w)), w)
    tile_h = min(max(1, int(tile_h)), h)
    if 0 <= overlap < 1:
        step_x = max(1, int(tile_w * (1.0 - overlap)))
        step_y = max(1, int(tile_h * (1.0 - overlap)))
    else:
        overlap_px = int(overlap)
        step_x = max(1, tile_w - overlap_px)
        step_y = max(1, tile_h - overlap_px)

    x_starts = list(range(0, max(w - tile_w + 1, 1), step_x))
    y_starts = list(range(0, max(h - tile_h + 1, 1), step_y))
    if x_starts[-1] != w - tile_w:
        x_starts.append(w - tile_w)
    if y_starts[-1] != h - tile_h:
        y_starts.append(h - tile_h)

    tiles: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for y1 in y_starts:
        for x1 in x_starts:
            x2 = x1 + tile_w
            y2 = y1 + tile_h
            tiles.append((frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)))
    return tiles


def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_detections(detections: list[dict[str, Any]], iou_threshold: float = 0.55) -> list[dict[str, Any]]:
    """Remove duplicated people produced by overlapping tiles."""
    if not detections:
        return []
    ordered = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    keep: list[dict[str, Any]] = []
    for det in ordered:
        box = np.array(det["bbox_xyxy"], dtype=np.float32)
        if all(_bbox_iou(box, np.array(prev["bbox_xyxy"], dtype=np.float32)) < iou_threshold for prev in keep):
            keep.append(det)
    return keep


def run_yolo_pose_on_frame(
    model: Any,
    frame: np.ndarray,
    imgsz: int = 1280,
    conf: float = 0.25,
    device: str | int | None = None,
    offset: tuple[int, int] = (0, 0),
    use_tracking: bool = False,
    tracker: str = "bytetrack.yaml",
    persist: bool = True,
) -> list[dict[str, Any]]:
    """Run YOLO pose on one frame/crop and return detections mapped to original coordinates."""
    if frame is None or frame.size == 0:
        return []

    predict_kwargs: dict[str, Any] = {"imgsz": imgsz, "conf": conf, "verbose": False}
    if device is not None:
        predict_kwargs["device"] = device
    if use_tracking and offset == (0, 0) and hasattr(model, "track"):
        results = model.track(frame, tracker=tracker, persist=persist, **predict_kwargs)
    else:
        results = model.predict(frame, **predict_kwargs)
    if not results:
        return []

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    confidences = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    track_ids: np.ndarray | None = None
    if getattr(result.boxes, "id", None) is not None:
        track_ids = result.boxes.id.detach().cpu().numpy().astype(np.int64)

    keypoints_xy = np.empty((len(boxes), 0, 2), dtype=np.float32)
    keypoint_scores = np.empty((len(boxes), 0), dtype=np.float32)
    if getattr(result, "keypoints", None) is not None and result.keypoints is not None:
        if getattr(result.keypoints, "xy", None) is not None:
            keypoints_xy = result.keypoints.xy.detach().cpu().numpy().astype(np.float32)
        if getattr(result.keypoints, "conf", None) is not None:
            keypoint_scores = result.keypoints.conf.detach().cpu().numpy().astype(np.float32)
        elif keypoints_xy.size:
            keypoint_scores = (np.any(keypoints_xy > 0, axis=-1)).astype(np.float32)

    offset_x, offset_y = offset
    detections: list[dict[str, Any]] = []
    for idx, box in enumerate(boxes):
        mapped_box = box.copy()
        mapped_box[[0, 2]] += offset_x
        mapped_box[[1, 3]] += offset_y

        mapped_kps = keypoints_xy[idx].copy() if idx < len(keypoints_xy) else np.empty((0, 2), dtype=np.float32)
        if mapped_kps.size:
            mapped_kps[:, 0] += offset_x
            mapped_kps[:, 1] += offset_y

        kp_scores = keypoint_scores[idx].copy() if idx < len(keypoint_scores) else np.empty((0,), dtype=np.float32)
        detections.append(
            {
                "bbox_xyxy": mapped_box.tolist(),
                "confidence": float(confidences[idx]),
                "track_id": int(track_ids[idx]) if track_ids is not None and idx < len(track_ids) else None,
                "keypoints_xy": mapped_kps.tolist(),
                "keypoint_scores": kp_scores.tolist(),
            }
        )
    return detections


def _pairwise_center_distances(centers: np.ndarray) -> np.ndarray:
    if len(centers) < 2:
        return np.array([], dtype=np.float32)
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))
    return dist[np.triu_indices(len(centers), k=1)]


def _min_cross_person_keypoint_distance(
    detections: list[dict[str, Any]],
    source_indices: tuple[int, ...],
    target_indices: tuple[int, ...],
    image_diag: float,
) -> float:
    best: float | None = None
    for src_idx, src in enumerate(detections):
        src_kps = np.array(src.get("keypoints_xy", []), dtype=np.float32)
        src_scores = np.array(src.get("keypoint_scores", []), dtype=np.float32)
        if src_kps.size == 0 or src_scores.size == 0:
            continue
        src_valid = _valid_keypoint_mask(src_kps, src_scores)
        for dst_idx, dst in enumerate(detections):
            if src_idx == dst_idx:
                continue
            dst_kps = np.array(dst.get("keypoints_xy", []), dtype=np.float32)
            dst_scores = np.array(dst.get("keypoint_scores", []), dtype=np.float32)
            if dst_kps.size == 0 or dst_scores.size == 0:
                continue
            dst_valid = _valid_keypoint_mask(dst_kps, dst_scores)
            for source_idx in source_indices:
                if source_idx >= len(src_kps) or not src_valid[source_idx]:
                    continue
                for target_idx in target_indices:
                    if target_idx >= len(dst_kps) or not dst_valid[target_idx]:
                        continue
                    distance = float(np.linalg.norm(src_kps[source_idx] - dst_kps[target_idx]) / max(image_diag, 1.0))
                    best = distance if best is None else min(best, distance)
    return float(best) if best is not None else 0.0


def extract_frame_tabular_features(
    detections: list[dict[str, Any]],
    frame_shape: tuple[int, int, int],
    previous_centers: np.ndarray | None = None,
    previous_timestamp: float | None = None,
    timestamp: float = 0.0,
    previous_frame_features: dict[str, float] | None = None,
    output_columns: list[str] | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    """Aggregate detections into one numeric row per frame for LGBM and LSTM."""
    columns = output_columns or FRAME_FEATURE_COLUMNS
    h, w = frame_shape[:2]
    image_diag = math.sqrt(float(w * w + h * h))
    person_count = len(detections)

    if person_count == 0:
        return {name: 0.0 for name in columns}, np.empty((0, 2), dtype=np.float32)

    boxes = np.array([det["bbox_xyxy"] for det in detections], dtype=np.float32)
    confidences = np.array([det["confidence"] for det in detections], dtype=np.float32)
    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    areas_norm = (widths * heights) / max(float(w * h), 1.0)
    centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0))

    pair_distances = _pairwise_center_distances(centers) / max(image_diag, 1.0)
    pair_ious: list[float] = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            pair_ious.append(_bbox_iou(boxes[i], boxes[j]))
    pair_count = max(1, int(person_count * (person_count - 1) / 2))
    close_pair_count_norm = float(np.sum(pair_distances <= 0.12) / pair_count) if len(pair_distances) else 0.0
    overlap_pair_count_norm = float(np.sum(np.asarray(pair_ious, dtype=np.float32) >= 0.05) / pair_count) if pair_ious else 0.0

    dt = 0.0 if previous_timestamp is None else max(float(timestamp - previous_timestamp), 1e-6)
    center_speeds = np.array([], dtype=np.float32)
    if previous_centers is not None and len(previous_centers) and dt > 0:
        speed_values = []
        for center in centers:
            distances_prev = np.sqrt(((previous_centers - center) ** 2).sum(axis=1))
            speed_values.append(float(distances_prev.min() / max(image_diag, 1.0) / dt))
        center_speeds = np.array(speed_values, dtype=np.float32)

    visible_counts: list[int] = []
    visible_scores: list[float] = []
    for det in detections:
        keypoints = np.array(det.get("keypoints_xy", []), dtype=np.float32)
        scores = np.array(det.get("keypoint_scores", []), dtype=np.float32)
        if keypoints.size == 0 or scores.size == 0:
            visible_counts.append(0)
            continue
        valid = _valid_keypoint_mask(keypoints, scores)
        visible_counts.append(int(valid.sum()))
        visible_scores.extend(scores[valid].astype(float).tolist())

    base_features = {
        "person_count": float(person_count),
        "mean_detection_conf": float(confidences.mean()) if len(confidences) else 0.0,
        "max_detection_conf": float(confidences.max()) if len(confidences) else 0.0,
        "sum_bbox_area_norm": float(areas_norm.sum()) if len(areas_norm) else 0.0,
        "mean_bbox_area_norm": float(areas_norm.mean()) if len(areas_norm) else 0.0,
        "max_bbox_area_norm": float(areas_norm.max()) if len(areas_norm) else 0.0,
        "min_pair_distance_norm": float(pair_distances.min()) if len(pair_distances) else 0.0,
        "mean_pair_distance_norm": float(pair_distances.mean()) if len(pair_distances) else 0.0,
        "max_iou_between_people": float(max(pair_ious)) if pair_ious else 0.0,
        "mean_center_speed_norm_per_sec": float(center_speeds.mean()) if len(center_speeds) else 0.0,
        "max_center_speed_norm_per_sec": float(center_speeds.max()) if len(center_speeds) else 0.0,
        "visible_keypoints_total": float(sum(visible_counts)),
        "visible_keypoints_mean": float(np.mean(visible_counts)) if visible_counts else 0.0,
        "mean_keypoint_score": float(np.mean(visible_scores)) if visible_scores else 0.0,
        "min_keypoint_score": float(np.min(visible_scores)) if visible_scores else 0.0,
    }

    previous = previous_frame_features or {}
    v2_features = {
        "min_pair_distance_delta": float(base_features["min_pair_distance_norm"] - previous.get("min_pair_distance_norm", 0.0)),
        "mean_pair_distance_delta": float(base_features["mean_pair_distance_norm"] - previous.get("mean_pair_distance_norm", 0.0)),
        "max_iou_delta": float(base_features["max_iou_between_people"] - previous.get("max_iou_between_people", 0.0)),
        "mean_center_accel_norm_per_sec2": (
            float(abs(base_features["mean_center_speed_norm_per_sec"] - previous.get("mean_center_speed_norm_per_sec", 0.0)) / dt)
            if dt > 0
            else 0.0
        ),
        "max_center_accel_norm_per_sec2": (
            float(abs(base_features["max_center_speed_norm_per_sec"] - previous.get("max_center_speed_norm_per_sec", 0.0)) / dt)
            if dt > 0
            else 0.0
        ),
        "mean_bbox_area_delta": float(base_features["mean_bbox_area_norm"] - previous.get("mean_bbox_area_norm", 0.0)),
        "max_bbox_area_delta": float(base_features["max_bbox_area_norm"] - previous.get("max_bbox_area_norm", 0.0)),
        "close_pair_count_norm": close_pair_count_norm,
        "overlap_pair_count_norm": overlap_pair_count_norm,
        "wrist_to_other_head_min_norm": _min_cross_person_keypoint_distance(
            detections, source_indices=(9, 10), target_indices=(0, 1, 2, 3, 4), image_diag=image_diag
        ),
        "wrist_to_other_torso_min_norm": _min_cross_person_keypoint_distance(
            detections, source_indices=(9, 10), target_indices=(5, 6, 11, 12), image_diag=image_diag
        ),
    }
    all_features = {**base_features, **v2_features}
    return ({name: float(all_features.get(name, 0.0)) for name in columns}, centers)


def extract_pose_features(
    detections: list[dict[str, Any]],
    frame_shape: tuple[int, int, int],
    frame_idx: int,
    timestamp: float,
    video_path: str | Path,
    mode: str,
    previous_centers: np.ndarray | None = None,
    previous_timestamp: float | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Extract per-person, per-frame features for LSTM/LGBM-friendly CSV output."""
    h, w = frame_shape[:2]
    image_diag = math.sqrt(float(w * w + h * h))
    person_count = len(detections)

    boxes = np.array([det["bbox_xyxy"] for det in detections], dtype=np.float32) if detections else np.empty((0, 4))
    centers = (
        np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0))
        if len(boxes)
        else np.empty((0, 2), dtype=np.float32)
    )
    pair_distances = _pairwise_center_distances(centers) / max(image_diag, 1.0)
    mean_pair_distance = float(pair_distances.mean()) if len(pair_distances) else 0.0
    min_pair_distance = float(pair_distances.min()) if len(pair_distances) else 0.0

    dt = 0.0 if previous_timestamp is None else max(float(timestamp - previous_timestamp), 1e-6)
    rows: list[dict[str, Any]] = []

    if not detections:
        rows.append(
            {
                "video_path": str(video_path),
                "mode": mode,
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "person_count": 0,
                "person_id": -1,
                "track_id": "",
                "bbox_x1_norm": "",
                "bbox_y1_norm": "",
                "bbox_x2_norm": "",
                "bbox_y2_norm": "",
                "bbox_area_norm": 0.0,
                "center_x_norm": "",
                "center_y_norm": "",
                "center_speed_norm_per_sec": 0.0,
                "max_iou_with_other": 0.0,
                "min_pair_distance_norm": 0.0,
                "mean_pair_distance_norm": 0.0,
                "visible_keypoints": 0,
                "mean_keypoint_score": 0.0,
                "bbox_xyxy_json": "[]",
                "keypoints_norm_json": "[]",
                "keypoint_scores_json": "[]",
            }
        )
        return rows, centers

    for idx, det in enumerate(detections):
        box = boxes[idx]
        x1, y1, x2, y2 = box
        width = max(0.0, float(x2 - x1))
        height = max(0.0, float(y2 - y1))
        area_norm = (width * height) / max(float(w * h), 1.0)
        center = centers[idx]

        if previous_centers is not None and len(previous_centers) and dt > 0:
            distances_prev = np.sqrt(((previous_centers - center) ** 2).sum(axis=1))
            center_speed = float(distances_prev.min() / max(image_diag, 1.0) / dt)
        else:
            center_speed = 0.0

        ious = [
            _bbox_iou(box, boxes[other_idx])
            for other_idx in range(len(boxes))
            if other_idx != idx
        ]
        max_iou = float(max(ious)) if ious else 0.0

        keypoints_abs = np.array(det.get("keypoints_xy", []), dtype=np.float32)
        keypoint_scores = np.array(det.get("keypoint_scores", []), dtype=np.float32)
        if keypoints_abs.size:
            keypoints_norm = keypoints_abs.copy()
            keypoints_norm[:, 0] /= max(float(w), 1.0)
            keypoints_norm[:, 1] /= max(float(h), 1.0)
            valid_mask = (
                (keypoints_abs[:, 0] > 0)
                & (keypoints_abs[:, 1] > 0)
                & (keypoint_scores >= KEYPOINT_CONF_THRESHOLD)
            )
            visible_keypoints = int(valid_mask.sum())
            mean_kp_score = float(keypoint_scores[valid_mask].mean()) if visible_keypoints else 0.0
        else:
            keypoints_norm = np.empty((0, 2), dtype=np.float32)
            visible_keypoints = 0
            mean_kp_score = 0.0

        rows.append(
            {
                "video_path": str(video_path),
                "mode": mode,
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "person_count": person_count,
                "person_id": idx,
                "track_id": det.get("track_id") if det.get("track_id") is not None else "",
                "bbox_x1_norm": float(x1 / max(w, 1)),
                "bbox_y1_norm": float(y1 / max(h, 1)),
                "bbox_x2_norm": float(x2 / max(w, 1)),
                "bbox_y2_norm": float(y2 / max(h, 1)),
                "bbox_area_norm": float(area_norm),
                "center_x_norm": float(center[0] / max(w, 1)),
                "center_y_norm": float(center[1] / max(h, 1)),
                "center_speed_norm_per_sec": center_speed,
                "max_iou_with_other": max_iou,
                "min_pair_distance_norm": min_pair_distance,
                "mean_pair_distance_norm": mean_pair_distance,
                "visible_keypoints": visible_keypoints,
                "mean_keypoint_score": mean_kp_score,
                "bbox_xyxy_json": json.dumps([float(v) for v in box]),
                "keypoints_norm_json": json.dumps(keypoints_norm.round(6).tolist()),
                "keypoint_scores_json": json.dumps(keypoint_scores.round(6).tolist()),
            }
        )

    return rows, centers


def _draw_detections(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    origin: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Draw detections on the current view frame. Origin maps original coords into this view."""
    out = frame.copy()
    ox, oy = origin
    for det in detections:
        box = np.array(det["bbox_xyxy"], dtype=np.float32)
        x1, y1, x2, y2 = (box - np.array([ox, oy, ox, oy], dtype=np.float32)).astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)

        keypoints = np.array(det.get("keypoints_xy", []), dtype=np.float32)
        scores = np.array(det.get("keypoint_scores", []), dtype=np.float32)
        for start, end in COCO_POSE_EDGES:
            if start >= len(keypoints) or end >= len(keypoints):
                continue
            start_score = scores[start] if start < len(scores) else 1.0
            end_score = scores[end] if end < len(scores) else 1.0
            if start_score < KEYPOINT_CONF_THRESHOLD or end_score < KEYPOINT_CONF_THRESHOLD:
                continue
            p1 = keypoints[start]
            p2 = keypoints[end]
            if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                cv2.line(
                    out,
                    (int(p1[0] - ox), int(p1[1] - oy)),
                    (int(p2[0] - ox), int(p2[1] - oy)),
                    (255, 180, 0),
                    2,
                )
        for idx, point in enumerate(keypoints):
            score = scores[idx] if idx < len(scores) else 1.0
            if score >= KEYPOINT_CONF_THRESHOLD and point[0] > 0 and point[1] > 0:
                px = int(point[0] - ox)
                py = int(point[1] - oy)
                cv2.circle(out, (px, py), 3, (0, 255, 255), -1)
    return out


def process_video(
    input_path: str | Path,
    output_path: str | Path | None = None,
    model_path: str | Path | None = "yolo11s-pose.pt",
    features_csv: str | Path | None = None,
    npz_output: str | Path | None = None,
    json_output: str | Path | None = None,
    video_output: str | Path | None = None,
    target_fps: float = 10.0,
    imgsz: int = 1280,
    conf: float = 0.25,
    mode: str = "full_frame",
    tile_size: int | tuple[int, int] = 960,
    overlap: float = 0.20,
    enhance_low_light: bool = False,
    draw_results: bool = False,
    device: str | int | None = None,
    feature_version: str = "v1",
    use_tracking: bool = False,
    tracker: str = "bytetrack.yaml",
    save_csv: bool = False,
    save_npz: bool = False,
    save_json: bool = False,
    save_video: bool = False,
) -> dict[str, Any]:
    """Preprocess a video, run YOLOv11 pose, and save requested outputs."""
    if mode not in {"full_frame", "tiles"}:
        raise ValueError('mode must be one of: "full_frame", "tiles"')
    if not any([save_csv, save_npz, save_json, save_video]):
        raise ValueError("Select at least one output: --save-csv, --save-npz, --save-json, or --save-video.")

    info = get_video_info(input_path)
    input_path = Path(input_path)
    frame_feature_columns = feature_columns(feature_version)
    if use_tracking and mode == "tiles":
        raise ValueError("ByteTrack/YOLO tracking is only supported with mode=full_frame.")

    model = None
    if model_path:
        if hasattr(model_path, "predict"):
            model = model_path
        else:
            from ultralytics import YOLO

            model = YOLO(str(model_path))

    csv_path = Path(features_csv) if features_csv else input_path.with_suffix(".lgbm_features.csv")
    npz_path = Path(npz_output) if npz_output else input_path.with_suffix(".lstm_sequence.npz")
    json_path = Path(json_output) if json_output else input_path.with_suffix(".detections.json")
    legacy_video_path = output_path
    resolved_video_output = video_output or legacy_video_path
    video_path = Path(resolved_video_output) if resolved_video_output else input_path.with_suffix(".pose.mp4")

    if save_csv:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    if save_npz:
        npz_path.parent.mkdir(parents=True, exist_ok=True)
    if save_json:
        json_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    if save_video:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, target_fps, (imgsz, imgsz))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {video_path}")

    processed_frames = 0
    failed_frames = 0
    previous_centers: np.ndarray | None = None
    previous_timestamp: float | None = None
    previous_frame_features: dict[str, float] | None = None
    csv_fh = csv_path.open("w", encoding="utf-8", newline="") if save_csv else None
    csv_writer = None
    if csv_fh is not None:
        csv_writer = csv.DictWriter(
            csv_fh,
            fieldnames=["video_path", "mode", "feature_version", "frame_idx", "timestamp", *frame_feature_columns],
        )
        csv_writer.writeheader()

    sequence_rows: list[list[float]] = []
    frame_indices: list[int] = []
    timestamps: list[float] = []
    json_frames: list[dict[str, Any]] = []

    try:
        for frame_idx, timestamp, frame in sample_frames_by_time(input_path, target_fps):
            try:
                working_frame = apply_low_light_enhancement(frame) if enhance_low_light else frame

                detections: list[dict[str, Any]] = []
                view_frame = working_frame
                view_origin = (0, 0)

                if model is not None and mode == "full_frame":
                    detections = run_yolo_pose_on_frame(
                        model,
                        working_frame,
                        imgsz=imgsz,
                        conf=conf,
                        device=device,
                        use_tracking=use_tracking,
                        tracker=tracker,
                        persist=True,
                    )

                elif model is not None and mode == "tiles":
                    tiled_detections: list[dict[str, Any]] = []
                    for tile, (x1, y1, _, _) in generate_tiles(working_frame, tile_size=tile_size, overlap=overlap):
                        tiled_detections.extend(
                            run_yolo_pose_on_frame(
                                model,
                                tile,
                                imgsz=imgsz,
                                conf=conf,
                                device=device,
                                offset=(x1, y1),
                            )
                        )
                    detections = nms_detections(tiled_detections)

                frame_features, current_centers = extract_frame_tabular_features(
                    detections=detections,
                    frame_shape=working_frame.shape,
                    previous_centers=previous_centers,
                    previous_timestamp=previous_timestamp,
                    timestamp=timestamp,
                    previous_frame_features=previous_frame_features,
                    output_columns=frame_feature_columns,
                )

                if csv_writer is not None:
                    csv_writer.writerow(
                        {
                            "video_path": str(input_path),
                            "mode": mode,
                            "feature_version": feature_version,
                            "frame_idx": frame_idx,
                            "timestamp": timestamp,
                            **frame_features,
                        }
                    )

                sequence_rows.append([float(frame_features[name]) for name in frame_feature_columns])
                frame_indices.append(frame_idx)
                timestamps.append(timestamp)

                if save_json:
                    json_frames.append(
                        {
                            "frame_idx": frame_idx,
                            "timestamp": timestamp,
                            "mode": mode,
                            "feature_version": feature_version,
                            "person_count": len(detections),
                            "detections": detections,
                        }
                    )

                previous_centers = current_centers
                previous_timestamp = timestamp
                previous_frame_features = frame_features

                if writer is not None:
                    output_frame = view_frame
                    if draw_results:
                        output_frame = _draw_detections(view_frame, detections, origin=view_origin)
                    letterboxed, _, _ = letterbox_frame(output_frame, imgsz)
                    writer.write(letterboxed)

                processed_frames += 1

            except Exception as exc:
                failed_frames += 1
                print(f"[WARN] frame {frame_idx} skipped: {exc}")
    finally:
        if csv_fh is not None:
            csv_fh.close()

    if writer is not None:
        writer.release()

    if save_npz:
        sequence = (
            np.array(sequence_rows, dtype=np.float32)
            if sequence_rows
            else np.empty((0, len(frame_feature_columns)), dtype=np.float32)
        )
        np.savez_compressed(
            npz_path,
            sequence=sequence,
            frame_indices=np.array(frame_indices, dtype=np.int64),
            timestamps=np.array(timestamps, dtype=np.float32),
            feature_names=np.array(frame_feature_columns),
            feature_version=str(feature_version),
            video_info=json.dumps(info),
            mode=mode,
            target_fps=float(target_fps),
            imgsz=int(imgsz),
            device=str(device) if device is not None else "auto",
        )

    if save_json:
        json_path.write_text(
            json.dumps(
                {
                    "video_info": info,
                    "target_fps": target_fps,
                    "imgsz": imgsz,
                    "device": str(device) if device is not None else "auto",
                    "mode": mode,
                    "feature_version": feature_version,
                    "use_tracking": bool(use_tracking),
                    "frames": json_frames,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return {
        "input": str(input_path),
        "csv_output": str(csv_path) if save_csv else None,
        "npz_output": str(npz_path) if save_npz else None,
        "json_output": str(json_path) if save_json else None,
        "video_output": str(video_path) if save_video else None,
        "source_info": info,
        "target_fps": target_fps,
        "imgsz": imgsz,
        "device": str(device) if device is not None else "auto",
        "mode": mode,
        "feature_version": feature_version,
        "use_tracking": bool(use_tracking),
        "feature_count": len(frame_feature_columns),
        "processed_frames": processed_frames,
        "failed_frames": failed_frames,
    }


def _parse_tile_size(value: str) -> int | tuple[int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise argparse.ArgumentTypeError("--tile-size must be N or W,H")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess CCTV videos and extract YOLOv11s-pose features.")
    parser.add_argument("--input", required=True, type=Path, help="Input video path.")
    parser.add_argument("--save-csv", action="store_true", help="Save numeric tabular features for LGBMClassifier.")
    parser.add_argument("--save-npz", action="store_true", help="Save compressed temporal sequence for LSTM.")
    parser.add_argument("--save-json", action="store_true", help="Save readable detections per frame.")
    parser.add_argument("--save-video", action="store_true", help="Save MP4 with pose skeletons for visual validation.")
    parser.add_argument("--output", type=Path, default=None, help="Legacy alias for --video-output.")
    parser.add_argument("--features-csv", type=Path, default=None, help="Legacy alias for --csv-output.")
    parser.add_argument("--csv-output", type=Path, default=None, help="CSV output path.")
    parser.add_argument("--npz-output", type=Path, default=None, help="NPZ output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="JSON output path.")
    parser.add_argument("--video-output", type=Path, default=None, help="MP4 output path.")
    parser.add_argument("--model", default="yolo11s-pose.pt", help="YOLO pose model path.")
    parser.add_argument("--target-fps", type=float, default=10.0, help="Effective sampling FPS.")
    parser.add_argument("--imgsz", type=int, default=1280, help="High YOLO inference size, e.g. 960 or 1280.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--device", default=None, help='Inference device, e.g. "cuda", "cuda:0", "0", or "cpu".')
    parser.add_argument("--mode", choices=["full_frame", "tiles"], default="full_frame")
    parser.add_argument("--tile-size", type=_parse_tile_size, default=960, help="Tile size as N or W,H.")
    parser.add_argument("--overlap", type=float, default=0.20, help="Tile overlap ratio or pixels if >= 1.")
    parser.add_argument("--enhance-low-light", action="store_true", help="Apply CLAHE+gamma to dark frames.")
    parser.add_argument("--draw-results", action="store_true", help="Draw boxes/keypoints on output video.")
    parser.add_argument("--feature-version", choices=["v1", "v2"], default="v1", help="Feature schema for NPZ/CSV outputs.")
    parser.add_argument("--use-tracking", action="store_true", help="Use Ultralytics/ByteTrack IDs in full_frame mode.")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config when --use-tracking is set.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = process_video(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        features_csv=args.csv_output or args.features_csv,
        npz_output=args.npz_output,
        json_output=args.json_output,
        video_output=args.video_output,
        target_fps=args.target_fps,
        imgsz=args.imgsz,
        conf=args.conf,
        mode=args.mode,
        tile_size=args.tile_size,
        overlap=args.overlap,
        enhance_low_light=args.enhance_low_light,
        draw_results=args.draw_results,
        device=args.device,
        feature_version=args.feature_version,
        use_tracking=args.use_tracking,
        tracker=args.tracker,
        save_csv=args.save_csv,
        save_npz=args.save_npz,
        save_json=args.save_json,
        save_video=args.save_video,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


# Example:
# python preprocess_cctv_pose.py --input video.mp4 --output out.mp4 --model yolo11s-pose.pt --target-fps 10 --imgsz 1280 --mode full_frame --save-csv --save-npz --save-json --save-video --draw-results
