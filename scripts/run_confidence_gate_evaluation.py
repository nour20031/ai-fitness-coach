import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2

from dataset_common import ROOT_DIR, read_csv, write_csv

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pose.bicep_curl_analyzer import BicepCurlAnalyzer
from src.pose.confidence_validator import ConfidenceValidator
from src.pose.pose_detector import PoseDetector
from src.pose.squat_analyzer import SquatAnalyzer


FIELDS = [
    "source_video_id",
    "participant_id",
    "exercise",
    "form_label",
    "camera_view",
    "expected_reps",
    "detected_reps",
    "confidence_status",
    "usable_frame_rate",
    "required_joint_availability",
    "angle_calculability_rate",
    "dropout_rate",
    "movement_signal_range",
    "reason_codes",
    "user_message",
    "source_path",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 5 confidence-gate evaluation.")
    parser.add_argument(
        "--manifest",
        default=str(ROOT_DIR / "dataset" / "processed" / "evaluation" / "pose_benchmark_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "dataset" / "processed" / "evaluation" / "results"),
    )
    parser.add_argument("--frame-step", type=int, default=1)
    return parser.parse_args()


def create_analyzer(exercise):
    if exercise == "squat":
        return SquatAnalyzer(verbose=False)
    if exercise == "bicep_curl":
        return BicepCurlAnalyzer(require_ready=False, verbose=False)
    raise ValueError(f"Unsupported exercise: {exercise}")


def process_video(row, frame_step):
    detector = PoseDetector()
    analyzer = create_analyzer(row["exercise"])
    validator = ConfidenceValidator(row["exercise"])
    capture = cv2.VideoCapture(row["source_path"])

    if not capture.isOpened():
        return {
            "source_video_id": row["source_video_id"],
            "participant_id": row["participant_id"],
            "exercise": row["exercise"],
            "form_label": row["form_label"],
            "camera_view": row["camera_view"],
            "expected_reps": row["expected_reps"],
            "detected_reps": 0,
            "confidence_status": "LOW",
            "usable_frame_rate": 0,
            "required_joint_availability": 0,
            "angle_calculability_rate": 0,
            "dropout_rate": 0,
            "movement_signal_range": 0,
            "reason_codes": "PROCESSING_ERROR",
            "user_message": "Analysis confidence is low because the video could not be opened.",
            "source_path": row["source_path"],
        }

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        current_frame = frame_index
        frame_index += 1
        if current_frame % max(frame_step, 1) != 0:
            continue

        detector.find_pose(frame, draw=False)
        landmarks = detector.get_landmarks(frame)
        validator.update(detector, landmarks)
        if landmarks:
            analyzer.analyze(detector, landmarks, frame)

    capture.release()

    if hasattr(analyzer, "finalize"):
        analyzer.finalize()

    summary = analyzer.get_session_summary()
    confidence = validator.result(summary.get("total_reps", 0))
    return {
        "source_video_id": row["source_video_id"],
        "participant_id": row["participant_id"],
        "exercise": row["exercise"],
        "form_label": row["form_label"],
        "camera_view": row["camera_view"],
        "expected_reps": row["expected_reps"],
        "detected_reps": summary.get("total_reps", 0),
        "confidence_status": confidence["confidence_status"],
        "usable_frame_rate": confidence["usable_frame_rate"],
        "required_joint_availability": confidence["required_joint_availability"],
        "angle_calculability_rate": confidence["angle_calculability_rate"],
        "dropout_rate": confidence["dropout_rate"],
        "movement_signal_range": confidence["movement_signal_range"],
        "reason_codes": ";".join(confidence["reason_codes"]),
        "user_message": confidence["user_message"],
        "source_path": row["source_path"],
    }


def markdown_table(rows, fields):
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    return "\n".join(lines)


def write_report(output_dir, rows):
    status_counts = Counter(row["confidence_status"] for row in rows)
    reason_counts = Counter()
    for row in rows:
        for reason in row["reason_codes"].split(";"):
            if reason:
                reason_counts[reason] += 1

    difficult_ids = {"vid01", "vid03", "vid09", "vid10", "vid15", "vid23"}
    difficult_rows = [row for row in rows if row["source_video_id"] in difficult_ids]
    blocked_rows = [row for row in rows if row["confidence_status"] == "LOW"]

    lines = [
        "# Phase 5 Confidence Gate Results",
        "",
        "The confidence gate runs after MediaPipe landmark extraction and before final coaching feedback is shown. Low-confidence videos are blocked from confident feedback and receive re-record guidance.",
        "",
        "## Summary",
        "",
        f"- Videos evaluated: {len(rows)}",
        f"- High confidence: {status_counts.get('HIGH', 0)}",
        f"- Low confidence: {status_counts.get('LOW', 0)}",
        "",
        "## Reason Codes",
        "",
    ]
    if reason_counts:
        lines.append(markdown_table(
            [{"reason_code": key, "count": value} for key, value in sorted(reason_counts.items())],
            ["reason_code", "count"],
        ))
        lines.extend(["", "Table 1. Low-confidence reason-code counts."])
    else:
        lines.append("No low-confidence reason codes were recorded.")

    lines.extend([
        "",
        "## Difficult-Video Check",
        "",
        markdown_table(
            difficult_rows,
            [
                "source_video_id",
                "exercise",
                "form_label",
                "camera_view",
                "detected_reps",
                "confidence_status",
                "movement_signal_range",
                "reason_codes",
            ],
        ),
        "",
        "Table 2. Confidence-gate behaviour on known difficult Phase 4 videos.",
        "",
        "## Blocked Videos",
        "",
    ])
    if blocked_rows:
        lines.append(markdown_table(
            blocked_rows,
            [
                "source_video_id",
                "exercise",
                "form_label",
                "camera_view",
                "detected_reps",
                "confidence_status",
                "reason_codes",
                "user_message",
            ],
        ))
        lines.extend(["", "Table 3. Videos blocked from confident feedback."])
    else:
        lines.append("No videos were blocked.")

    (Path(output_dir) / "confidence_gate_results.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    rows = read_csv(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for index, row in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] {row['source_video_id']} {row['exercise']} {row['camera_view']}")
        results.append(process_video(row, args.frame_step))

    write_csv(output_dir / "confidence_gate_results.csv", results, FIELDS)
    write_report(output_dir, results)
    print(f"Wrote Phase 5 confidence-gate results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
