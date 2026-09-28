import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

from dataset_common import ROOT_DIR, read_csv, write_csv

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pose.bicep_curl_analyzer import BicepCurlAnalyzer
from src.pose.pose_detector import PoseDetector
from src.pose.squat_analyzer import SquatAnalyzer


RESULT_FIELDS = [
    "source_video_id",
    "participant_id",
    "exercise",
    "form_label",
    "camera_view",
    "expected_reps",
    "detected_reps",
    "absolute_rep_count_error",
    "predicted_form_label",
    "form_label_match",
    "pose_detection_coverage",
    "processed_frames",
    "total_frames",
    "fps",
    "duration",
    "failure_categories",
    "likely_failure_source",
    "issue_counts",
    "rep_history",
    "thresholds",
    "source_path",
    "notes",
]

SUMMARY_FIELDS = [
    "group_type",
    "group_value",
    "video_count",
    "expected_reps",
    "detected_reps",
    "rep_count_mae",
    "exact_count_rate",
    "within_plus_minus_1_rate",
    "form_label_match_rate",
    "mean_pose_detection_coverage",
    "failure_rate",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Phase 4 final movement-analysis evaluation with MediaPipe Pose."
    )
    parser.add_argument(
        "--manifest",
        default=str(ROOT_DIR / "dataset" / "processed" / "evaluation" / "pose_benchmark_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "dataset" / "processed" / "evaluation" / "results"),
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Analyze every Nth frame. Phase 4 defaults to every frame, matching the app pipeline.",
    )
    return parser.parse_args()


def create_analyzer(exercise):
    if exercise == "squat":
        return SquatAnalyzer(verbose=False)
    if exercise == "bicep_curl":
        return BicepCurlAnalyzer(require_ready=False, verbose=False)
    raise ValueError(f"Unsupported exercise: {exercise}")


def truthy(value):
    return str(value).lower() == "true"


def get_failure_categories(row):
    failures = []
    pose_coverage = float(row["pose_detection_coverage"])
    absolute_error = int(row["absolute_rep_count_error"])
    detected_reps = int(row["detected_reps"])

    if pose_coverage == 0:
        failures.append("POSE_NOT_DETECTED")
    elif pose_coverage < 0.70:
        failures.append("LOW_POSE_COVERAGE")
    if detected_reps == 0:
        failures.append("NO_REPS_DETECTED")
    if absolute_error > 1:
        failures.append("REP_COUNT_ERROR")
    if not truthy(row["form_label_match"]):
        failures.append("FORM_LABEL_MISMATCH")
    return failures


def likely_failure_source(row, failures):
    if not failures:
        return ""
    if "POSE_NOT_DETECTED" in failures or "LOW_POSE_COVERAGE" in failures:
        return "pose_landmarks_or_joint_visibility"
    if "NO_REPS_DETECTED" in failures:
        return "rep_state_logic_or_insufficient_visible_movement"
    if "REP_COUNT_ERROR" in failures and row["exercise"] == "bicep_curl":
        return "curl_rep_state_logic_or_partial_arm_visibility"
    if "REP_COUNT_ERROR" in failures and row["exercise"] == "squat":
        return "squat_rep_state_logic_or_camera_angle"
    if "FORM_LABEL_MISMATCH" in failures:
        return "form_threshold_or_movement_range"
    return "manual_review_required"


def process_video(row, frame_step):
    analyzer = create_analyzer(row["exercise"])
    detector = PoseDetector()
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
            "absolute_rep_count_error": int(row["expected_reps"]),
            "predicted_form_label": "unknown",
            "form_label_match": False,
            "pose_detection_coverage": 0,
            "processed_frames": 0,
            "total_frames": row["frame_count"],
            "fps": row["fps"],
            "duration": row["duration"],
            "failure_categories": "PROCESSING_ERROR",
            "likely_failure_source": "video_could_not_be_opened",
            "issue_counts": "{}",
            "rep_history": "[]",
            "thresholds": "{}",
            "source_path": row["source_path"],
            "notes": "could_not_open_video",
        }

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or int(row.get("frame_count") or 0)
    processed_frames = 0
    pose_frames = 0
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        current_frame = frame_index
        frame_index += 1
        if current_frame % max(frame_step, 1) != 0:
            continue

        processed_frames += 1
        detector.find_pose(frame, draw=False)
        landmarks = detector.get_landmarks(frame)
        if landmarks:
            pose_frames += 1
            analyzer.analyze(detector, landmarks, frame)

    capture.release()

    if hasattr(analyzer, "finalize"):
        analyzer.finalize()

    summary = analyzer.get_session_summary()
    detected_reps = int(summary.get("total_reps", 0))
    expected_reps = int(row["expected_reps"])
    predicted_label = summary.get("predicted_form_label", "unknown")
    result = {
        "source_video_id": row["source_video_id"],
        "participant_id": row["participant_id"],
        "exercise": row["exercise"],
        "form_label": row["form_label"],
        "camera_view": row["camera_view"],
        "expected_reps": expected_reps,
        "detected_reps": detected_reps,
        "absolute_rep_count_error": abs(detected_reps - expected_reps),
        "predicted_form_label": predicted_label,
        "form_label_match": predicted_label == row["form_label"],
        "pose_detection_coverage": round(pose_frames / processed_frames, 4) if processed_frames else 0,
        "processed_frames": processed_frames,
        "total_frames": total_frames,
        "fps": row["fps"],
        "duration": row["duration"],
        "issue_counts": json.dumps(summary.get("issue_counts", {}), sort_keys=True),
        "rep_history": json.dumps(summary.get("rep_history", [])),
        "thresholds": json.dumps(summary.get("thresholds", {}), sort_keys=True),
        "source_path": row["source_path"],
        "notes": summary.get("form_label_reason", ""),
    }
    failures = get_failure_categories(result)
    result["failure_categories"] = ";".join(failures)
    result["likely_failure_source"] = likely_failure_source(result, failures)
    return result


def mean(values):
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0


def summary_row(group_type, group_value, rows):
    expected = sum(int(row["expected_reps"]) for row in rows)
    detected = sum(int(row["detected_reps"]) for row in rows)
    errors = [int(row["absolute_rep_count_error"]) for row in rows]
    return {
        "group_type": group_type,
        "group_value": group_value,
        "video_count": len(rows),
        "expected_reps": expected,
        "detected_reps": detected,
        "rep_count_mae": round(mean(errors), 4),
        "exact_count_rate": round(sum(error == 0 for error in errors) / len(rows), 4),
        "within_plus_minus_1_rate": round(sum(error <= 1 for error in errors) / len(rows), 4),
        "form_label_match_rate": round(
            sum(truthy(row["form_label_match"]) for row in rows) / len(rows),
            4,
        ),
        "mean_pose_detection_coverage": round(
            mean(row["pose_detection_coverage"] for row in rows),
            4,
        ),
        "failure_rate": round(
            sum(1 for row in rows if row["failure_categories"]) / len(rows),
            4,
        ),
    }


def group_summaries(rows):
    output = [summary_row("overall", "all", rows)]
    for key in ["exercise", "camera_view", "form_label", "participant_id"]:
        groups = defaultdict(list)
        for row in rows:
            groups[row[key]].append(row)
        for value, group in sorted(groups.items()):
            output.append(summary_row(key, value, group))

    exercise_view = defaultdict(list)
    exercise_label = defaultdict(list)
    for row in rows:
        exercise_view[f"{row['exercise']}/{row['camera_view']}"].append(row)
        exercise_label[f"{row['exercise']}/{row['form_label']}"].append(row)
    for value, group in sorted(exercise_view.items()):
        output.append(summary_row("exercise_camera_view", value, group))
    for value, group in sorted(exercise_label.items()):
        output.append(summary_row("exercise_form_label", value, group))
    return output


def markdown_table(rows, fields):
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    return "\n".join(lines)


def write_markdown_report(output_dir, rows, summaries):
    overall = next(row for row in summaries if row["group_type"] == "overall")
    exercise_rows = [row for row in summaries if row["group_type"] == "exercise"]
    view_rows = [row for row in summaries if row["group_type"] == "camera_view"]
    label_rows = [row for row in summaries if row["group_type"] == "form_label"]
    failure_rows = [row for row in rows if row["failure_categories"]]

    lines = [
        "# Phase 4 Movement Analysis Results",
        "",
        "MediaPipe Pose / BlazePose is fixed as the pose model. These results evaluate the explainable rule-based movement-analysis layer for squat and bicep curl on the frozen 23 original source videos.",
        "",
        "## Overall Results",
        "",
        markdown_table(
            [overall],
            [
                "video_count",
                "expected_reps",
                "detected_reps",
                "rep_count_mae",
                "exact_count_rate",
                "within_plus_minus_1_rate",
                "form_label_match_rate",
                "failure_rate",
            ],
        ),
        "",
        "Table 1. Overall movement-analysis performance on the frozen source-video benchmark.",
        "",
        "## Group Results",
        "",
        markdown_table(
            exercise_rows,
            ["group_value", "video_count", "rep_count_mae", "exact_count_rate", "within_plus_minus_1_rate", "form_label_match_rate"],
        ),
        "",
        "Table 2. Movement-analysis performance by exercise.",
        "",
        markdown_table(
            view_rows,
            ["group_value", "video_count", "rep_count_mae", "exact_count_rate", "within_plus_minus_1_rate", "form_label_match_rate"],
        ),
        "",
        "Table 3. Movement-analysis performance by camera view.",
        "",
        markdown_table(
            label_rows,
            ["group_value", "video_count", "rep_count_mae", "exact_count_rate", "within_plus_minus_1_rate", "form_label_match_rate"],
        ),
        "",
        "Table 4. Movement-analysis performance by verified form label.",
        "",
        "## Failure Cases",
        "",
    ]
    if failure_rows:
        lines.append(markdown_table(
            failure_rows,
            [
                "source_video_id",
                "exercise",
                "form_label",
                "camera_view",
                "expected_reps",
                "detected_reps",
                "absolute_rep_count_error",
                "predicted_form_label",
                "failure_categories",
                "likely_failure_source",
            ],
        ))
        lines.extend([
            "",
            "Table 5. Videos requiring review after the final Phase 4 movement-analysis evaluation.",
        ])
    else:
        lines.append("No failure cases were recorded.")

    (Path(output_dir) / "movement_analysis_results.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    rows = read_csv(args.manifest)
    if not rows:
        print(f"No rows found in {args.manifest}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for index, row in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] {row['source_video_id']} {row['exercise']} {row['form_label']} {row['camera_view']}")
        results.append(process_video(row, args.frame_step))

    summaries = group_summaries(results)
    write_csv(output_dir / "movement_analysis_results.csv", results, RESULT_FIELDS)
    write_csv(output_dir / "movement_analysis_group_summary.csv", summaries, SUMMARY_FIELDS)
    write_markdown_report(output_dir, results, summaries)

    print(f"Wrote Phase 4 movement-analysis results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
