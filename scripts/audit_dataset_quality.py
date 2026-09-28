import argparse
import statistics
import sys
from pathlib import Path

import cv2

from dataset_common import ROOT_DIR, get_video_metadata, read_csv, write_csv

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pose.landmarks import Landmarks
from src.pose.pose_detector import PoseDetector


QUALITY_FIELDS = [
    "new_filename",
    "exercise",
    "form_label",
    "camera_view",
    "participant_id",
    "readable",
    "duration_seconds",
    "fps",
    "width",
    "height",
    "frame_count",
    "sampled_frames",
    "pose_detection_ratio",
    "important_landmark_visibility_mean",
    "movement_signal_range",
    "flags",
    "processed_path",
]


IMPORTANT_LANDMARKS = {
    "squat": [
        Landmarks.LEFT_HIP,
        Landmarks.RIGHT_HIP,
        Landmarks.LEFT_KNEE,
        Landmarks.RIGHT_KNEE,
        Landmarks.LEFT_ANKLE,
        Landmarks.RIGHT_ANKLE,
    ],
    "bicep_curl": [
        Landmarks.LEFT_SHOULDER,
        Landmarks.RIGHT_SHOULDER,
        Landmarks.LEFT_ELBOW,
        Landmarks.RIGHT_ELBOW,
        Landmarks.LEFT_WRIST,
        Landmarks.RIGHT_WRIST,
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Audit video quality and pose detectability.")
    parser.add_argument(
        "--manifest",
        default=str(ROOT_DIR / "dataset" / "processed" / "manifests" / "raw_videos_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT_DIR / "dataset" / "processed" / "results" / "video_quality_report.csv"),
    )
    parser.add_argument(
        "--summary",
        default=str(ROOT_DIR / "docs" / "dataset_quality_report.md"),
    )
    parser.add_argument("--sample-step", type=int, default=10)
    return parser.parse_args()


def movement_signal(detector, landmarks, exercise):
    if exercise == "squat":
        left = detector.calculate_angle(
            landmarks,
            Landmarks.LEFT_HIP,
            Landmarks.LEFT_KNEE,
            Landmarks.LEFT_ANKLE,
        )
        right = detector.calculate_angle(
            landmarks,
            Landmarks.RIGHT_HIP,
            Landmarks.RIGHT_KNEE,
            Landmarks.RIGHT_ANKLE,
        )
        values = [value for value in [left, right] if value is not None]
        return sum(values) / len(values) if values else None

    if exercise == "bicep_curl":
        left = detector.calculate_angle(
            landmarks,
            Landmarks.LEFT_SHOULDER,
            Landmarks.LEFT_ELBOW,
            Landmarks.LEFT_WRIST,
        )
        right = detector.calculate_angle(
            landmarks,
            Landmarks.RIGHT_SHOULDER,
            Landmarks.RIGHT_ELBOW,
            Landmarks.RIGHT_WRIST,
        )
        values = [value for value in [left, right] if value is not None]
        return sum(values) / len(values) if values else None

    return None


def audit_video(row, sample_step):
    path = Path(row["processed_path"])
    metadata = get_video_metadata(path)
    flags = []

    if not metadata["readable"]:
        flags.append("unreadable")
        return {
            **{field: row.get(field, "") for field in ["new_filename", "exercise", "form_label", "camera_view", "participant_id", "processed_path"]},
            **metadata,
            "sampled_frames": 0,
            "pose_detection_ratio": 0,
            "important_landmark_visibility_mean": 0,
            "movement_signal_range": 0,
            "flags": "; ".join(flags),
        }

    if metadata["duration_seconds"] < 5:
        flags.append("very_short")
    if metadata["fps"] < 10:
        flags.append("low_fps")
    if metadata["width"] < 360 or metadata["height"] < 240:
        flags.append("low_resolution")

    detector = PoseDetector()
    capture = cv2.VideoCapture(str(path))
    frame_index = 0
    sampled = 0
    detected = 0
    visibilities = []
    signals = []

    important = IMPORTANT_LANDMARKS.get(row["exercise"], [])

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % max(sample_step, 1) != 0:
            continue

        sampled += 1
        detector.find_pose(frame, draw=False)
        landmarks = detector.get_landmarks(frame)
        if not landmarks:
            continue

        detected += 1
        visible_values = [
            landmarks[idx]["visibility"]
            for idx in important
            if idx in landmarks
        ]
        if visible_values:
            visibilities.append(sum(visible_values) / len(visible_values))

        signal = movement_signal(detector, landmarks, row["exercise"])
        if signal is not None:
            signals.append(signal)

    capture.release()

    pose_ratio = detected / sampled if sampled else 0
    mean_visibility = statistics.mean(visibilities) if visibilities else 0
    signal_range = max(signals) - min(signals) if len(signals) >= 2 else 0

    if pose_ratio < 0.5:
        flags.append("low_pose_detection")
    if mean_visibility < 0.5:
        flags.append("low_landmark_visibility")
    if signal_range < 15:
        flags.append("possible_zero_or_low_movement")

    return {
        "new_filename": row["new_filename"],
        "exercise": row["exercise"],
        "form_label": row["form_label"],
        "camera_view": row["camera_view"],
        "participant_id": row["participant_id"],
        "readable": metadata["readable"],
        "duration_seconds": metadata["duration_seconds"],
        "fps": metadata["fps"],
        "width": metadata["width"],
        "height": metadata["height"],
        "frame_count": metadata["frame_count"],
        "sampled_frames": sampled,
        "pose_detection_ratio": round(pose_ratio, 3),
        "important_landmark_visibility_mean": round(mean_visibility, 3),
        "movement_signal_range": round(signal_range, 3),
        "flags": "; ".join(flags),
        "processed_path": row["processed_path"],
    }


def write_markdown_summary(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    durations = [float(row["duration_seconds"]) for row in rows if row["readable"]]
    flagged = [row for row in rows if row["flags"]]

    lines = [
        "# Dataset Quality Report",
        "",
        "This report is generated automatically from `scripts/audit_dataset_quality.py`.",
        "Videos are flagged for human review only; no source videos are deleted or modified.",
        "",
        f"- Videos audited: {len(rows)}",
        f"- Flagged for review: {len(flagged)}",
    ]

    if durations:
        lines.extend([
            f"- Average duration: {round(statistics.mean(durations), 2)} seconds",
            f"- Minimum duration: {round(min(durations), 2)} seconds",
            f"- Maximum duration: {round(max(durations), 2)} seconds",
        ])

    lines.extend(["", "## Flagged Videos", ""])
    if not flagged:
        lines.append("No videos were flagged by the automatic checks.")
    else:
        lines.append("| Video | Exercise | Label | View | Flags |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in flagged:
            lines.append(
                f"| {row['new_filename']} | {row['exercise']} | {row['form_label']} | "
                f"{row['camera_view']} | {row['flags']} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    manifest_rows = read_csv(args.manifest)
    if not manifest_rows:
        print("No manifest rows found. Run scripts/build_dataset_manifest.py first.")
        return 1

    rows = [audit_video(row, args.sample_step) for row in manifest_rows]
    write_csv(args.output, rows, QUALITY_FIELDS)
    write_markdown_summary(args.summary, rows)
    print(f"Wrote quality CSV to {args.output}")
    print(f"Wrote quality summary to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
