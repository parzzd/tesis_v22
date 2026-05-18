from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from preprocess_cctv_pose import (
    KEYPOINT_CONF_THRESHOLD,
    VIDEO_EXTENSIONS,
    _draw_detections,
    _valid_keypoint_mask,
    get_video_info,
    run_yolo_pose_on_frame,
    sample_frames_by_time,
)


COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

KEYPOINT_GROUPS = {
    "head": [0, 1, 2, 3, 4],
    "torso": [5, 6, 11, 12],
    "arms_hands": [7, 8, 9, 10],
    "legs": [13, 14, 15, 16],
}


@dataclass(frozen=True)
class VideoItem:
    path: Path
    dataset: str
    subset: str
    label: str


def safe_id(video: VideoItem) -> str:
    stem = video.path.stem.replace(" ", "_")
    return f"{video.dataset}_{video.subset}_{video.label}_{stem}".replace("\\", "_").replace("/", "_")


def infer_video_item(path: Path) -> VideoItem:
    parts = path.parts
    dataset = "custom"
    subset = "custom"
    label = "unknown"
    for known in ("RWF-2000", "VioPeru-main"):
        if known in parts:
            dataset = known
            idx = parts.index(known)
            subset = parts[idx + 1] if idx + 1 < len(parts) else "unknown"
            label = parts[idx + 2] if idx + 2 < len(parts) and parts[idx + 2] in {"Fight", "NonFight"} else "unknown"
            if subset == "false_positives_validation":
                label = "NonFight"
            break
    return VideoItem(path=path, dataset=dataset, subset=subset, label=label)


def discover_dataset_videos(
    roots: list[Path],
    subsets: list[str],
    max_videos_per_class: int,
) -> list[VideoItem]:
    videos: list[VideoItem] = []
    for root in roots:
        if not root.exists():
            continue
        dataset = root.name
        for subset in subsets:
            subset_dir = root / subset
            if not subset_dir.exists():
                continue
            class_dirs = [p for p in subset_dir.iterdir() if p.is_dir() and p.name in {"Fight", "NonFight"}]
            if class_dirs:
                for class_dir in sorted(class_dirs):
                    files = sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
                    for path in files[:max_videos_per_class]:
                        videos.append(VideoItem(path=path, dataset=dataset, subset=subset, label=class_dir.name))
            else:
                files = sorted(p for p in subset_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
                for path in files[:max_videos_per_class]:
                    videos.append(VideoItem(path=path, dataset=dataset, subset=subset, label="NonFight"))
    return videos


def bbox_area_norm(det: dict[str, Any], width: int, height: int) -> float:
    x1, y1, x2, y2 = np.asarray(det.get("bbox_xyxy", [0, 0, 0, 0]), dtype=np.float32)
    area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    return area / max(float(width * height), 1.0)


def visible_group_rate(valid_mask: np.ndarray, indices: list[int]) -> float:
    if valid_mask.size == 0:
        return 0.0
    usable = [idx for idx in indices if idx < len(valid_mask)]
    if not usable:
        return 0.0
    return float(np.mean(valid_mask[usable]))


def summarize_frame(
    detections: list[dict[str, Any]],
    width: int,
    height: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    person_rows: list[dict[str, Any]] = []
    keypoint_updates: list[dict[str, Any]] = []
    visible_counts: list[int] = []
    score_values: list[float] = []
    area_values = [bbox_area_norm(det, width, height) for det in detections]
    group_values = {name: [] for name in KEYPOINT_GROUPS}

    for person_id, det in enumerate(detections):
        keypoints = np.asarray(det.get("keypoints_xy", []), dtype=np.float32)
        scores = np.asarray(det.get("keypoint_scores", []), dtype=np.float32)
        valid = _valid_keypoint_mask(keypoints, scores)
        visible_count = int(valid.sum()) if valid.size else 0
        visible_counts.append(visible_count)

        valid_scores = scores[valid] if valid.size and scores.size else np.asarray([], dtype=np.float32)
        if len(valid_scores):
            score_values.extend(valid_scores.astype(float).tolist())

        groups = {group: visible_group_rate(valid, indices) for group, indices in KEYPOINT_GROUPS.items()}
        for group, value in groups.items():
            group_values[group].append(value)

        person_rows.append(
            {
                "person_id": person_id,
                "track_id": det.get("track_id") if det.get("track_id") is not None else "",
                "bbox_area_norm": area_values[person_id],
                "detection_conf": float(det.get("confidence", 0.0)),
                "visible_keypoints": visible_count,
                "mean_keypoint_score": float(valid_scores.mean()) if len(valid_scores) else 0.0,
                **{f"{group}_visible_rate": value for group, value in groups.items()},
            }
        )

        for kp_idx, kp_name in enumerate(COCO_KEYPOINT_NAMES):
            visible = bool(kp_idx < len(valid) and valid[kp_idx])
            score = float(scores[kp_idx]) if kp_idx < len(scores) else 0.0
            keypoint_updates.append({"keypoint": kp_name, "visible": visible, "score": score})

    persons_with_pose = int(sum(1 for count in visible_counts if count > 0))
    mean_visible = float(np.mean(visible_counts)) if visible_counts else 0.0
    mean_score = float(np.mean(score_values)) if score_values else 0.0
    frame_quality = "good"
    if not detections:
        frame_quality = "no_person"
    elif persons_with_pose == 0:
        frame_quality = "no_keypoints"
    elif mean_visible < 6 or mean_score < 0.35:
        frame_quality = "weak_pose"

    frame_summary = {
        "person_count": len(detections),
        "persons_with_pose": persons_with_pose,
        "mean_visible_keypoints": mean_visible,
        "max_visible_keypoints": int(max(visible_counts)) if visible_counts else 0,
        "mean_keypoint_score": mean_score,
        "min_bbox_area_norm": float(np.min(area_values)) if area_values else 0.0,
        "mean_bbox_area_norm": float(np.mean(area_values)) if area_values else 0.0,
        "max_bbox_area_norm": float(np.max(area_values)) if area_values else 0.0,
        "small_person_count": int(sum(1 for area in area_values if area < 0.025)),
        "quality_flag": frame_quality,
        **{f"{group}_visible_rate": float(np.mean(values)) if values else 0.0 for group, values in group_values.items()},
    }
    return frame_summary, person_rows, keypoint_updates


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def draw_quality_overlay(frame: np.ndarray, detections: list[dict[str, Any]], label: str) -> np.ndarray:
    drawn = _draw_detections(frame, detections)
    height, width = drawn.shape[:2]
    cv2.rectangle(drawn, (0, 0), (min(width, 980), 30), (15, 23, 42), -1)
    cv2.putText(drawn, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (248, 250, 252), 1, cv2.LINE_AA)
    return drawn


def analyze_videos(
    videos: list[VideoItem],
    model_path: Path,
    output_dir: Path,
    target_fps: float,
    max_frames: int,
    imgsz: int,
    conf: float,
    device: str | None,
    save_frames: int,
    save_videos: bool,
) -> dict[str, Any]:
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames_annotated"
    frames_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos_annotated"
    if save_videos:
        videos_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    frame_rows: list[dict[str, Any]] = []
    person_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    keypoint_stats: dict[tuple[str, str], dict[str, float]] = {}

    for video in videos:
        video_id = safe_id(video)
        info = get_video_info(video.path)
        sampled = 0
        saved = 0
        frame_quality_counts = {"good": 0, "weak_pose": 0, "no_person": 0, "no_keypoints": 0}
        video_frame_indices: list[int] = []
        video_people_counts: list[int] = []
        video_visible_means: list[float] = []
        video_score_means: list[float] = []
        video_pose_frames = 0
        video_total_persons = 0
        writer: cv2.VideoWriter | None = None
        annotated_video_path = videos_dir / f"{video_id}.mp4"

        try:
            for frame_idx, timestamp, frame in sample_frames_by_time(video.path, target_fps):
                if max_frames > 0 and sampled >= max_frames:
                    break
                sampled += 1
                height, width = frame.shape[:2]
                detections = run_yolo_pose_on_frame(
                    model=model,
                    frame=frame,
                    imgsz=imgsz,
                    conf=conf,
                    device=device,
                )
                frame_summary, frame_person_rows, keypoint_updates = summarize_frame(detections, width, height)
                frame_quality_counts[frame_summary["quality_flag"]] += 1
                video_frame_indices.append(frame_idx)
                video_people_counts.append(int(frame_summary["person_count"]))
                video_visible_means.append(float(frame_summary["mean_visible_keypoints"]))
                video_score_means.append(float(frame_summary["mean_keypoint_score"]))
                video_total_persons += int(frame_summary["person_count"])
                if int(frame_summary["persons_with_pose"]) > 0:
                    video_pose_frames += 1

                frame_rows.append(
                    {
                        "video_id": video_id,
                        "dataset": video.dataset,
                        "subset": video.subset,
                        "label": video.label,
                        "video_path": str(video.path),
                        "frame_idx": frame_idx,
                        "timestamp": timestamp,
                        "width": width,
                        "height": height,
                        **frame_summary,
                    }
                )
                label = (
                    f"{video.dataset}/{video.subset}/{video.label} frame={frame_idx} "
                    f"persons={frame_summary['person_count']} visible={frame_summary['mean_visible_keypoints']:.1f} "
                    f"score={frame_summary['mean_keypoint_score']:.2f} {frame_summary['quality_flag']}"
                )
                annotated = None

                for row in frame_person_rows:
                    person_rows.append(
                        {
                            "video_id": video_id,
                            "dataset": video.dataset,
                            "subset": video.subset,
                            "label": video.label,
                            "video_path": str(video.path),
                            "frame_idx": frame_idx,
                            "timestamp": timestamp,
                            **row,
                        }
                    )
                for update in keypoint_updates:
                    key = (video_id, str(update["keypoint"]))
                    stats = keypoint_stats.setdefault(key, {"visible": 0.0, "total": 0.0, "score_sum": 0.0})
                    stats["total"] += 1.0
                    if update["visible"]:
                        stats["visible"] += 1.0
                        stats["score_sum"] += float(update["score"])

                if save_videos:
                    if annotated is None:
                        annotated = draw_quality_overlay(frame, detections, label)
                    if writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(annotated_video_path), fourcc, float(target_fps), (width, height))
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not create annotated video: {annotated_video_path}")
                    writer.write(annotated)

                should_save_regular = save_frames > 0 and saved < save_frames
                should_save_low_quality = frame_summary["quality_flag"] in {"weak_pose", "no_keypoints"} and saved < save_frames + 2
                if should_save_regular or should_save_low_quality:
                    if annotated is None:
                        annotated = draw_quality_overlay(frame, detections, label)
                    cv2.imwrite(str(frames_dir / f"{video_id}_frame_{frame_idx:06d}.jpg"), annotated)
                    saved += 1
        finally:
            if writer is not None:
                writer.release()

        frame_count = max(sampled, 1)
        video_rows.append(
            {
                "video_id": video_id,
                "dataset": video.dataset,
                "subset": video.subset,
                "label": video.label,
                "video_path": str(video.path),
                "source_fps": float(info["fps"]),
                "source_width": int(info["width"]),
                "source_height": int(info["height"]),
                "source_duration_seconds": float(info["duration_seconds"]),
                "sampled_frames": sampled,
                "pose_frame_rate": video_pose_frames / frame_count,
                "no_person_frame_rate": frame_quality_counts["no_person"] / frame_count,
                "weak_or_no_keypoints_frame_rate": (frame_quality_counts["weak_pose"] + frame_quality_counts["no_keypoints"]) / frame_count,
                "mean_person_count": float(np.mean(video_people_counts)) if video_people_counts else 0.0,
                "max_person_count": int(max(video_people_counts)) if video_people_counts else 0,
                "mean_visible_keypoints_per_frame": float(np.mean(video_visible_means)) if video_visible_means else 0.0,
                "mean_keypoint_score": float(np.mean([v for v in video_score_means if v > 0])) if any(v > 0 for v in video_score_means) else 0.0,
                "total_detected_persons": video_total_persons,
                **{f"{key}_frames": value for key, value in frame_quality_counts.items()},
            }
        )

    keypoint_rows: list[dict[str, Any]] = []
    for (video_id, kp_name), stats in sorted(keypoint_stats.items()):
        visible = int(stats["visible"])
        total = int(stats["total"])
        keypoint_rows.append(
            {
                "video_id": video_id,
                "keypoint": kp_name,
                "visible_count": visible,
                "total_person_instances": total,
                "visible_rate": visible / max(total, 1),
                "mean_visible_score": stats["score_sum"] / max(visible, 1),
            }
        )

    write_csv(
        output_dir / "frame_keypoint_quality.csv",
        frame_rows,
        [
            "video_id",
            "dataset",
            "subset",
            "label",
            "video_path",
            "frame_idx",
            "timestamp",
            "width",
            "height",
            "person_count",
            "persons_with_pose",
            "mean_visible_keypoints",
            "max_visible_keypoints",
            "mean_keypoint_score",
            "min_bbox_area_norm",
            "mean_bbox_area_norm",
            "max_bbox_area_norm",
            "small_person_count",
            "head_visible_rate",
            "torso_visible_rate",
            "arms_hands_visible_rate",
            "legs_visible_rate",
            "quality_flag",
        ],
    )
    write_csv(
        output_dir / "person_keypoint_quality.csv",
        person_rows,
        [
            "video_id",
            "dataset",
            "subset",
            "label",
            "video_path",
            "frame_idx",
            "timestamp",
            "person_id",
            "track_id",
            "bbox_area_norm",
            "detection_conf",
            "visible_keypoints",
            "mean_keypoint_score",
            "head_visible_rate",
            "torso_visible_rate",
            "arms_hands_visible_rate",
            "legs_visible_rate",
        ],
    )
    write_csv(output_dir / "video_keypoint_summary.csv", video_rows, list(video_rows[0].keys()) if video_rows else [])
    write_csv(
        output_dir / "keypoint_visibility_summary.csv",
        keypoint_rows,
        ["video_id", "keypoint", "visible_count", "total_person_instances", "visible_rate", "mean_visible_score"],
    )

    report = build_report(videos, video_rows, keypoint_rows, output_dir, target_fps, imgsz, conf)
    (output_dir / "keypoint_analysis_report.md").write_text(report, encoding="utf-8")
    metadata = {
        "model_path": str(model_path),
        "target_fps": target_fps,
        "imgsz": imgsz,
        "conf": conf,
        "device": device,
        "video_count": len(videos),
        "outputs": {
            "frame_csv": "frame_keypoint_quality.csv",
            "person_csv": "person_keypoint_quality.csv",
            "video_csv": "video_keypoint_summary.csv",
            "keypoint_csv": "keypoint_visibility_summary.csv",
            "report": "keypoint_analysis_report.md",
            "annotated_frames_dir": "frames_annotated",
            "annotated_videos_dir": "videos_annotated" if save_videos else None,
        },
    }
    (output_dir / "keypoint_analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def build_report(
    videos: list[VideoItem],
    video_rows: list[dict[str, Any]],
    keypoint_rows: list[dict[str, Any]],
    output_dir: Path,
    target_fps: float,
    imgsz: int,
    conf: float,
) -> str:
    total_frames = sum(int(row["sampled_frames"]) for row in video_rows)
    total_people = sum(int(row["total_detected_persons"]) for row in video_rows)
    pose_frame_rate = float(np.mean([row["pose_frame_rate"] for row in video_rows])) if video_rows else 0.0
    weak_rate = float(np.mean([row["weak_or_no_keypoints_frame_rate"] for row in video_rows])) if video_rows else 0.0
    visible_mean = float(np.mean([row["mean_visible_keypoints_per_frame"] for row in video_rows])) if video_rows else 0.0

    kp_global: dict[str, dict[str, float]] = {}
    for row in keypoint_rows:
        stats = kp_global.setdefault(row["keypoint"], {"visible": 0.0, "total": 0.0, "score_sum": 0.0})
        stats["visible"] += float(row["visible_count"])
        stats["total"] += float(row["total_person_instances"])
        stats["score_sum"] += float(row["mean_visible_score"]) * float(row["visible_count"])

    kp_lines = []
    for name in COCO_KEYPOINT_NAMES:
        stats = kp_global.get(name, {"visible": 0.0, "total": 0.0, "score_sum": 0.0})
        visible = stats["visible"]
        total = stats["total"]
        score = stats["score_sum"] / max(visible, 1.0)
        kp_lines.append(f"| {name} | {visible / max(total, 1.0):.3f} | {score:.3f} | {int(visible)} / {int(total)} |")

    video_lines = []
    for row in video_rows:
        video_lines.append(
            "| {video_id} | {dataset} | {label} | {sampled_frames} | {pose_frame_rate:.3f} | "
            "{no_person_frame_rate:.3f} | {weak_or_no_keypoints_frame_rate:.3f} | {mean_person_count:.2f} | "
            "{mean_visible_keypoints_per_frame:.2f} | {mean_keypoint_score:.3f} |".format(**row)
        )

    caution_rows = [
        row
        for row in video_rows
        if float(row["weak_or_no_keypoints_frame_rate"]) > 0.35
        or (float(row["mean_person_count"]) >= 0.50 and float(row["pose_frame_rate"]) < 0.60)
    ]
    caution_text = (
        "No se encontraron videos criticos en esta muestra."
        if not caution_rows
        else "\n".join(
            f"- `{row['video_id']}`: pose_frame_rate={row['pose_frame_rate']:.3f}, "
            f"weak_or_no_keypoints={row['weak_or_no_keypoints_frame_rate']:.3f}"
            for row in caution_rows
        )
    )

    return "\n".join(
        [
            "# Analisis de keypoints YOLO11s-pose",
            "",
            "Este reporte valida si YOLO11s-pose esta leyendo personas y keypoints antes de alimentar LGBM/LSTM/Stacker.",
            "",
            "## Configuracion",
            "",
            f"- Videos analizados: `{len(videos)}`",
            f"- Frames muestreados: `{total_frames}`",
            f"- Personas detectadas: `{total_people}`",
            f"- FPS efectivo: `{target_fps}`",
            f"- imgsz: `{imgsz}`",
            f"- conf: `{conf}`",
            f"- Umbral de keypoint visible: `{KEYPOINT_CONF_THRESHOLD}`",
            "",
            "## Resumen global",
            "",
            f"- Promedio de frames con pose: `{pose_frame_rate:.3f}`",
            f"- Promedio de frames debiles/sin keypoints: `{weak_rate:.3f}`",
            f"- Keypoints visibles promedio por frame: `{visible_mean:.2f}`",
            "",
            "Interpretacion rapida: si hay personas y `pose_frame_rate` baja de `0.60`, o si los frames debiles superan `0.35`, el video puede estar aportando features poco confiables para el modelo. Si `no_person_frame_rate` es alto en NonFight, puede ser normal y no necesariamente indica fallo.",
            "",
            "## Resumen por video",
            "",
            "| video_id | dataset | label | frames | pose_frame_rate | no_person | weak/no_kp | mean_people | mean_visible_kp | mean_kp_score |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *video_lines,
            "",
            "## Visibilidad global por keypoint",
            "",
            "| keypoint | visible_rate | mean_visible_score | visible / total |",
            "| --- | ---: | ---: | ---: |",
            *kp_lines,
            "",
            "## Videos a revisar visualmente",
            "",
            caution_text,
            "",
            "## Archivos generados",
            "",
            f"- `{output_dir / 'frame_keypoint_quality.csv'}`",
            f"- `{output_dir / 'person_keypoint_quality.csv'}`",
            f"- `{output_dir / 'video_keypoint_summary.csv'}`",
            f"- `{output_dir / 'keypoint_visibility_summary.csv'}`",
            f"- `{output_dir / 'frames_annotated'}`",
            f"- `{output_dir / 'videos_annotated'}`",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze YOLO11s-pose keypoint quality on sample CCTV videos.")
    parser.add_argument("--model", default="yolo11s-pose.pt", help="YOLO pose model path.")
    parser.add_argument("--output-dir", default="métricas/keypoints_yolo11s", help="Directory for reports.")
    parser.add_argument("--videos", nargs="*", default=None, help="Explicit video paths. If omitted, sample datasets are used.")
    parser.add_argument("--dataset-roots", nargs="*", default=["RWF-2000", "VioPeru-main"], help="Dataset roots to sample.")
    parser.add_argument("--subsets", nargs="*", default=["val"], help="Dataset subsets to sample, e.g. train val.")
    parser.add_argument("--max-videos-per-class", type=int, default=1, help="Number of videos per dataset/subset/class.")
    parser.add_argument("--target-fps", type=float, default=5.0, help="Effective FPS for analysis.")
    parser.add_argument("--max-frames", type=int, default=40, help="Max sampled frames per video. Use 0 for full video.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--device", default=None, help='YOLO device, e.g. "cuda", "cpu", or "0". Defaults to ultralytics auto.')
    parser.add_argument("--save-frames", type=int, default=4, help="Annotated frames to save per video.")
    parser.add_argument("--save-videos", action="store_true", help="Save annotated MP4 videos with keypoints overlaid.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.videos:
        videos = [infer_video_item(Path(path)) for path in args.videos]
    else:
        videos = discover_dataset_videos(
            roots=[Path(root) for root in args.dataset_roots],
            subsets=list(args.subsets),
            max_videos_per_class=max(int(args.max_videos_per_class), 1),
        )

    if not videos:
        raise SystemExit("No videos found for keypoint analysis.")

    metadata = analyze_videos(
        videos=videos,
        model_path=Path(args.model),
        output_dir=Path(args.output_dir),
        target_fps=float(args.target_fps),
        max_frames=int(args.max_frames),
        imgsz=int(args.imgsz),
        conf=float(args.conf),
        device=args.device,
        save_frames=int(args.save_frames),
        save_videos=bool(args.save_videos),
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
