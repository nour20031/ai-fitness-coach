import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from dataset_common import ROOT_DIR, read_csv, write_csv

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.orchestration import FitnessCoachOrchestrator


MANIFEST_PATH = ROOT_DIR / "dataset" / "processed" / "evaluation" / "pose_benchmark_manifest.csv"
OUTPUT_DIR = ROOT_DIR / "dataset" / "processed" / "evaluation" / "robustness"
VIDEO_OUTPUT_DIR = ROOT_DIR / "dataset" / "processed" / "robustness" / "videos"

CONDITIONS = [
    "baseline",
    "darker",
    "brighter",
    "rotation",
    "crop",
    "partial_body",
    "poor_framing",
    "fast_movement",
    "occlusion",
]

TRANSFORM_PARAMETERS = {
    "baseline": "original source video; no transform",
    "darker": "cv2.convertScaleAbs(alpha=0.60, beta=-10)",
    "brighter": "cv2.convertScaleAbs(alpha=1.18, beta=25)",
    "rotation": "5 degree centre rotation, borderMode=BORDER_REPLICATE; reused Phase 3 preliminary rotation level",
    "crop": "centre crop 5 percent margins on all sides, resized back to original dimensions; reused Phase 3 crop/zoom level",
    "partial_body": "exercise-specific visibility stress: squat masks bottom 22 percent; bicep_curl masks right 26 percent",
    "poor_framing": "translate frame +16 percent width and +8 percent height with black border",
    "fast_movement": "temporal compression proxy: keep every second frame at original fps, approximately half duration",
    "occlusion": "deterministic black rectangle over relevant joints: squat central lower body; bicep_curl upper/side arm area",
}

RESULT_FIELDS = [
    "robustness_id",
    "source_video_id",
    "participant_id",
    "exercise",
    "verified_label",
    "camera_view",
    "expected_reps",
    "condition",
    "source_path",
    "analysis_path",
    "is_derived",
    "processed_frames",
    "total_frames",
    "pose_detection_coverage",
    "required_joint_availability",
    "angle_calculability",
    "tracking_dropout",
    "usable_frame_rate",
    "movement_signal_range",
    "expected_reps_result",
    "detected_reps",
    "absolute_rep_count_error",
    "exact_count",
    "within_plus_minus_1",
    "expected_form_label",
    "predicted_form_label",
    "form_label_match",
    "confidence_status",
    "confidence_reasons",
    "coaching_blocked",
    "failure_categories",
    "processing_time_seconds",
    "notes",
]

MANIFEST_FIELDS = [
    "robustness_id",
    "source_video_id",
    "participant_id",
    "exercise",
    "verified_label",
    "camera_view",
    "expected_reps",
    "condition",
    "transform_parameters",
    "source_path",
    "derived_path",
    "is_derived",
    "notes",
]

SUMMARY_FIELDS = [
    "condition",
    "video_count",
    "mean_rep_count_mae",
    "median_rep_count_absolute_error",
    "exact_count_rate",
    "within_plus_minus_1_rate",
    "form_label_match_rate",
    "high_confidence_rate",
    "low_confidence_rate",
    "coaching_block_rate",
    "mean_pose_detection_coverage",
    "mean_required_joint_availability",
    "mean_angle_calculability",
    "mean_tracking_dropout",
    "mean_movement_signal_range",
    "delta_rep_mae_vs_baseline",
    "delta_exact_count_rate_vs_baseline",
    "delta_within_plus_minus_1_rate_vs_baseline",
    "delta_form_label_match_vs_baseline",
    "delta_pose_coverage_vs_baseline",
    "delta_joint_availability_vs_baseline",
    "delta_low_confidence_rate_vs_baseline",
]

PAIRED_FIELDS = [
    "source_video_id",
    "exercise",
    "verified_label",
    "camera_view",
    "condition",
    "baseline_detected_reps",
    "condition_detected_reps",
    "detected_rep_difference",
    "baseline_abs_error",
    "condition_abs_error",
    "abs_error_difference",
    "baseline_form_match",
    "condition_form_match",
    "baseline_confidence",
    "condition_confidence",
    "baseline_pose_coverage",
    "condition_pose_coverage",
    "pose_coverage_difference",
    "baseline_joint_availability",
    "condition_joint_availability",
    "joint_availability_difference",
    "paired_change",
]

GROUP_FIELDS = [
    "group_type",
    "group_value",
    "condition",
    "video_count",
    "mean_rep_count_mae",
    "exact_count_rate",
    "within_plus_minus_1_rate",
    "form_label_match_rate",
    "high_confidence_rate",
    "low_confidence_rate",
    "mean_pose_detection_coverage",
    "mean_required_joint_availability",
    "mean_angle_calculability",
]

FAILURE_FIELDS = [
    "robustness_id",
    "source_video_id",
    "exercise",
    "verified_label",
    "camera_view",
    "condition",
    "baseline_detected_reps",
    "condition_detected_reps",
    "baseline_abs_error",
    "condition_abs_error",
    "predicted_form_label",
    "confidence_status",
    "pose_detection_coverage",
    "required_joint_availability",
    "angle_calculability",
    "movement_signal_range",
    "failure_categories",
    "notes",
]

STAT_FIELDS = [
    "condition",
    "video_count",
    "mean_abs_error_difference",
    "median_abs_error_difference",
    "improved_count",
    "unchanged_count",
    "worsened_count",
    "wilcoxon_statistic",
    "wilcoxon_p_value",
    "statistical_notes",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 12 robustness and failure tests.")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--video-output-dir", default=str(VIDEO_OUTPUT_DIR))
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--only-create-variants", action="store_true")
    parser.add_argument("--force-recreate-variants", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel analysis workers for the frozen visual pipeline. Results remain identical in schema and are checkpointed by the parent process.",
    )
    return parser.parse_args()


def safe_condition_id(text):
    return text.replace("bicep_curl", "curl")


def derived_path(video_output_dir, row, condition):
    source_id = row["source_video_id"]
    stem = f"{source_id}_{row['exercise']}_{row['form_label']}_{row['camera_view']}_{condition}.mp4"
    return Path(video_output_dir) / condition / stem


def transform_frame(frame, condition, exercise):
    if condition == "baseline" or condition == "fast_movement":
        return frame
    if condition == "darker":
        return cv2.convertScaleAbs(frame, alpha=0.60, beta=-10)
    if condition == "brighter":
        return cv2.convertScaleAbs(frame, alpha=1.18, beta=25)
    if condition == "rotation":
        height, width = frame.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 5, 1.0)
        return cv2.warpAffine(frame, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    if condition == "crop":
        height, width = frame.shape[:2]
        margin_x = int(width * 0.05)
        margin_y = int(height * 0.05)
        cropped = frame[margin_y:height - margin_y, margin_x:width - margin_x]
        return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    if condition == "partial_body":
        output = frame.copy()
        height, width = output.shape[:2]
        if exercise == "squat":
            y1 = int(height * 0.78)
            output[y1:height, 0:width] = 0
        else:
            x1 = int(width * 0.74)
            output[0:height, x1:width] = 0
        return output
    if condition == "poor_framing":
        height, width = frame.shape[:2]
        matrix = np.float32([[1, 0, int(width * 0.16)], [0, 1, int(height * 0.08)]])
        return cv2.warpAffine(frame, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    if condition == "occlusion":
        output = frame.copy()
        height, width = output.shape[:2]
        if exercise == "squat":
            x1, x2 = int(width * 0.35), int(width * 0.65)
            y1, y2 = int(height * 0.45), int(height * 0.82)
        else:
            x1, x2 = int(width * 0.42), int(width * 0.72)
            y1, y2 = int(height * 0.20), int(height * 0.62)
        output[y1:y2, x1:x2] = 0
        return output
    raise ValueError(f"Unknown robustness condition: {condition}")


def create_variant(source_path, destination_path, condition, exercise, force=False):
    if condition == "baseline":
        return source_path
    if destination_path.exists() and destination_path.stat().st_size > 0 and not force:
        return destination_path

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {source_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if condition == "fast_movement" and frame_index % 2 == 1:
            frame_index += 1
            continue
        writer.write(transform_frame(frame, condition, exercise))
        frame_index += 1

    capture.release()
    writer.release()
    return destination_path


def build_robustness_manifest(rows, video_output_dir, force_recreate=False):
    manifest_rows = []
    for row in rows:
        source_path = Path(row["source_path"])
        for condition in CONDITIONS:
            robustness_id = f"{row['source_video_id']}_{condition}"
            if condition == "baseline":
                derived = source_path
                is_derived = "False"
            else:
                derived = derived_path(video_output_dir, row, condition)
                create_variant(source_path, derived, condition, row["exercise"], force=force_recreate)
                is_derived = "True"
            manifest_rows.append({
                "robustness_id": robustness_id,
                "source_video_id": row["source_video_id"],
                "participant_id": row["participant_id"],
                "exercise": row["exercise"],
                "verified_label": row["form_label"],
                "camera_view": row["camera_view"],
                "expected_reps": row["expected_reps"],
                "condition": condition,
                "transform_parameters": TRANSFORM_PARAMETERS[condition],
                "source_path": row["source_path"],
                "derived_path": str(derived),
                "is_derived": is_derived,
                "notes": "Derived robustness variants are paired stress-test inputs, not new participants or independent samples.",
            })
    return manifest_rows


def truthy(value):
    return str(value).lower() == "true"


def parse_existing_results(path):
    if not Path(path).exists():
        return {}, []
    rows = read_csv(path)
    return {row["robustness_id"]: row for row in rows}, rows


def failure_categories(structured, confidence, expected_reps, detected_reps, expected_label, predicted_label):
    categories = []
    metrics = structured.confidence_metrics
    if float(metrics.get("pose_detected_frames", 0) or 0) == 0:
        categories.append("POSE_NOT_DETECTED")
    if float(metrics.get("required_joint_availability", 0) or 0) < 0.65:
        categories.append("LOW_REQUIRED_JOINT_AVAILABILITY")
    if float(metrics.get("angle_calculability_rate", 0) or 0) < 0.65:
        categories.append("LOW_ANGLE_CALCULABILITY")
    if float(metrics.get("dropout_rate", 0) or 0) > 0.25:
        categories.append("HIGH_TRACKING_DROPOUT")
    if detected_reps == 0:
        categories.append("NO_REPS_DETECTED")
    if abs(detected_reps - expected_reps) > 1:
        categories.append("REP_COUNT_ERROR_GT_1")
    if predicted_label != expected_label:
        categories.append("FORM_MISMATCH")
    if structured.confidence_status == "LOW":
        categories.append("LOW_CONFIDENCE")
        categories.append("COACHING_BLOCKED")
    return categories


def analyze_manifest_row(orchestrator, row):
    start = time.perf_counter()
    result = orchestrator.analyze_video(
        row["derived_path"],
        row["exercise"],
        preview_interval=0,
        require_ready=False,
    )
    elapsed = round(time.perf_counter() - start, 4)
    expected_reps = int(row["expected_reps"])
    if result.get("status") != "success":
        return {
            "robustness_id": row["robustness_id"],
            "source_video_id": row["source_video_id"],
            "participant_id": row["participant_id"],
            "exercise": row["exercise"],
            "verified_label": row["verified_label"],
            "camera_view": row["camera_view"],
            "expected_reps": expected_reps,
            "condition": row["condition"],
            "source_path": row["source_path"],
            "analysis_path": row["derived_path"],
            "is_derived": row["is_derived"],
            "processed_frames": 0,
            "total_frames": "",
            "pose_detection_coverage": 0,
            "required_joint_availability": 0,
            "angle_calculability": 0,
            "tracking_dropout": 0,
            "usable_frame_rate": 0,
            "movement_signal_range": 0,
            "expected_reps_result": expected_reps,
            "detected_reps": 0,
            "absolute_rep_count_error": expected_reps,
            "exact_count": False,
            "within_plus_minus_1": False,
            "expected_form_label": row["verified_label"],
            "predicted_form_label": "unknown",
            "form_label_match": False,
            "confidence_status": "LOW",
            "confidence_reasons": "PROCESSING_ERROR",
            "coaching_blocked": True,
            "failure_categories": "PROCESSING_ERROR",
            "processing_time_seconds": elapsed,
            "notes": result.get("error", "processing_error"),
        }

    structured = result["structured_result"]
    metrics = structured.confidence_metrics
    detected_reps = int(structured.rep_count)
    abs_error = abs(detected_reps - expected_reps)
    predicted_label = structured.form_label
    categories = failure_categories(
        structured,
        result.get("confidence", {}),
        expected_reps,
        detected_reps,
        row["verified_label"],
        predicted_label,
    )
    return {
        "robustness_id": row["robustness_id"],
        "source_video_id": row["source_video_id"],
        "participant_id": row["participant_id"],
        "exercise": row["exercise"],
        "verified_label": row["verified_label"],
        "camera_view": row["camera_view"],
        "expected_reps": expected_reps,
        "condition": row["condition"],
        "source_path": row["source_path"],
        "analysis_path": row["derived_path"],
        "is_derived": row["is_derived"],
        "processed_frames": metrics.get("processed_frames", ""),
        "total_frames": metrics.get("processed_frames", ""),
        "pose_detection_coverage": metrics.get("usable_frame_rate", 0),
        "required_joint_availability": metrics.get("required_joint_availability", 0),
        "angle_calculability": metrics.get("angle_calculability_rate", 0),
        "tracking_dropout": metrics.get("dropout_rate", 0),
        "usable_frame_rate": metrics.get("usable_frame_rate", 0),
        "movement_signal_range": metrics.get("movement_signal_range", 0),
        "expected_reps_result": expected_reps,
        "detected_reps": detected_reps,
        "absolute_rep_count_error": abs_error,
        "exact_count": abs_error == 0,
        "within_plus_minus_1": abs_error <= 1,
        "expected_form_label": row["verified_label"],
        "predicted_form_label": predicted_label,
        "form_label_match": predicted_label == row["verified_label"],
        "confidence_status": structured.confidence_status,
        "confidence_reasons": ";".join(structured.confidence_reasons),
        "coaching_blocked": structured.confidence_status == "LOW",
        "failure_categories": ";".join(categories),
        "processing_time_seconds": result.get("processing_time_seconds", elapsed),
        "notes": "",
    }


def analyze_manifest_row_worker(row):
    orchestrator = FitnessCoachOrchestrator()
    return analyze_manifest_row(orchestrator, row)


def mean(values):
    values = [float(value) for value in values]
    return round(sum(values) / len(values), 4) if values else 0


def median(values):
    values = sorted(float(value) for value in values)
    if not values:
        return 0
    midpoint = len(values) // 2
    if len(values) % 2:
        return round(values[midpoint], 4)
    return round((values[midpoint - 1] + values[midpoint]) / 2, 4)


def rate(rows, predicate):
    return round(sum(1 for row in rows if predicate(row)) / len(rows), 4) if rows else 0


def condition_metrics(rows):
    return {
        "video_count": len(rows),
        "mean_rep_count_mae": mean(row["absolute_rep_count_error"] for row in rows),
        "median_rep_count_absolute_error": median(row["absolute_rep_count_error"] for row in rows),
        "exact_count_rate": rate(rows, lambda row: truthy(row["exact_count"])),
        "within_plus_minus_1_rate": rate(rows, lambda row: truthy(row["within_plus_minus_1"])),
        "form_label_match_rate": rate(rows, lambda row: truthy(row["form_label_match"])),
        "high_confidence_rate": rate(rows, lambda row: row["confidence_status"] == "HIGH"),
        "low_confidence_rate": rate(rows, lambda row: row["confidence_status"] == "LOW"),
        "coaching_block_rate": rate(rows, lambda row: truthy(row["coaching_blocked"])),
        "mean_pose_detection_coverage": mean(row["pose_detection_coverage"] for row in rows),
        "mean_required_joint_availability": mean(row["required_joint_availability"] for row in rows),
        "mean_angle_calculability": mean(row["angle_calculability"] for row in rows),
        "mean_tracking_dropout": mean(row["tracking_dropout"] for row in rows),
        "mean_movement_signal_range": mean(row["movement_signal_range"] for row in rows),
    }


def summarize_conditions(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    baseline = condition_metrics(groups["baseline"])
    output = []
    for condition in CONDITIONS:
        metrics = condition_metrics(groups[condition])
        metrics.update({
            "condition": condition,
            "delta_rep_mae_vs_baseline": round(metrics["mean_rep_count_mae"] - baseline["mean_rep_count_mae"], 4),
            "delta_exact_count_rate_vs_baseline": round(metrics["exact_count_rate"] - baseline["exact_count_rate"], 4),
            "delta_within_plus_minus_1_rate_vs_baseline": round(metrics["within_plus_minus_1_rate"] - baseline["within_plus_minus_1_rate"], 4),
            "delta_form_label_match_vs_baseline": round(metrics["form_label_match_rate"] - baseline["form_label_match_rate"], 4),
            "delta_pose_coverage_vs_baseline": round(metrics["mean_pose_detection_coverage"] - baseline["mean_pose_detection_coverage"], 4),
            "delta_joint_availability_vs_baseline": round(metrics["mean_required_joint_availability"] - baseline["mean_required_joint_availability"], 4),
            "delta_low_confidence_rate_vs_baseline": round(metrics["low_confidence_rate"] - baseline["low_confidence_rate"], 4),
        })
        output.append(metrics)
    return output


def summarize_group(rows, group_type, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row[key], row["condition"])].append(row)
    output = []
    for (value, condition), group_rows in sorted(grouped.items()):
        metrics = condition_metrics(group_rows)
        output.append({
            "group_type": group_type,
            "group_value": value,
            "condition": condition,
            "video_count": metrics["video_count"],
            "mean_rep_count_mae": metrics["mean_rep_count_mae"],
            "exact_count_rate": metrics["exact_count_rate"],
            "within_plus_minus_1_rate": metrics["within_plus_minus_1_rate"],
            "form_label_match_rate": metrics["form_label_match_rate"],
            "high_confidence_rate": metrics["high_confidence_rate"],
            "low_confidence_rate": metrics["low_confidence_rate"],
            "mean_pose_detection_coverage": metrics["mean_pose_detection_coverage"],
            "mean_required_joint_availability": metrics["mean_required_joint_availability"],
            "mean_angle_calculability": metrics["mean_angle_calculability"],
        })
    return output


def paired_differences(rows):
    by_key = {(row["source_video_id"], row["condition"]): row for row in rows}
    output = []
    for row in rows:
        if row["condition"] == "baseline":
            continue
        baseline = by_key.get((row["source_video_id"], "baseline"))
        if not baseline:
            continue
        abs_diff = int(row["absolute_rep_count_error"]) - int(baseline["absolute_rep_count_error"])
        if abs_diff < 0:
            paired_change = "improved"
        elif abs_diff > 0:
            paired_change = "worsened"
        else:
            paired_change = "unchanged"
        output.append({
            "source_video_id": row["source_video_id"],
            "exercise": row["exercise"],
            "verified_label": row["verified_label"],
            "camera_view": row["camera_view"],
            "condition": row["condition"],
            "baseline_detected_reps": baseline["detected_reps"],
            "condition_detected_reps": row["detected_reps"],
            "detected_rep_difference": int(row["detected_reps"]) - int(baseline["detected_reps"]),
            "baseline_abs_error": baseline["absolute_rep_count_error"],
            "condition_abs_error": row["absolute_rep_count_error"],
            "abs_error_difference": abs_diff,
            "baseline_form_match": baseline["form_label_match"],
            "condition_form_match": row["form_label_match"],
            "baseline_confidence": baseline["confidence_status"],
            "condition_confidence": row["confidence_status"],
            "baseline_pose_coverage": baseline["pose_detection_coverage"],
            "condition_pose_coverage": row["pose_detection_coverage"],
            "pose_coverage_difference": round(float(row["pose_detection_coverage"]) - float(baseline["pose_detection_coverage"]), 4),
            "baseline_joint_availability": baseline["required_joint_availability"],
            "condition_joint_availability": row["required_joint_availability"],
            "joint_availability_difference": round(float(row["required_joint_availability"]) - float(baseline["required_joint_availability"]), 4),
            "paired_change": paired_change,
        })
    return output


def robustness_statistics(paired_rows):
    groups = defaultdict(list)
    for row in paired_rows:
        groups[row["condition"]].append(row)
    output = []
    for condition in [item for item in CONDITIONS if item != "baseline"]:
        group = groups[condition]
        diffs = [float(row["abs_error_difference"]) for row in group]
        improved = sum(1 for value in diffs if value < 0)
        unchanged = sum(1 for value in diffs if value == 0)
        worsened = sum(1 for value in diffs if value > 0)
        stat = ""
        p_value = ""
        notes = ""
        try:
            from scipy.stats import wilcoxon

            non_zero = [value for value in diffs if value != 0]
            if len(non_zero) >= 5:
                result = wilcoxon(non_zero)
                stat = round(float(result.statistic), 4)
                p_value = round(float(result.pvalue), 6)
            else:
                notes = "Wilcoxon not run because fewer than five non-zero paired differences were available."
        except Exception as exc:
            notes = f"Wilcoxon not run: {exc}"
        output.append({
            "condition": condition,
            "video_count": len(group),
            "mean_abs_error_difference": mean(diffs),
            "median_abs_error_difference": median(diffs),
            "improved_count": improved,
            "unchanged_count": unchanged,
            "worsened_count": worsened,
            "wilcoxon_statistic": stat,
            "wilcoxon_p_value": p_value,
            "statistical_notes": notes,
        })
    return output


def failure_case_rows(rows, paired_rows):
    paired_lookup = {(row["source_video_id"], row["condition"]): row for row in paired_rows}
    candidates = []
    for row in rows:
        if row["condition"] == "baseline":
            continue
        categories = row["failure_categories"]
        pair = paired_lookup.get((row["source_video_id"], row["condition"]), {})
        if categories or int(pair.get("abs_error_difference", 0) or 0) > 0:
            candidates.append((row, pair))
    candidates = sorted(
        candidates,
        key=lambda item: (
            int(item[0]["absolute_rep_count_error"]),
            abs(float(item[1].get("joint_availability_difference", 0) or 0)),
        ),
        reverse=True,
    )
    output = []
    seen = set()
    for row, pair in candidates:
        key = (row["condition"], row["exercise"], row["failure_categories"])
        if key in seen and len(output) >= 20:
            continue
        seen.add(key)
        output.append({
            "robustness_id": row["robustness_id"],
            "source_video_id": row["source_video_id"],
            "exercise": row["exercise"],
            "verified_label": row["verified_label"],
            "camera_view": row["camera_view"],
            "condition": row["condition"],
            "baseline_detected_reps": pair.get("baseline_detected_reps", ""),
            "condition_detected_reps": row["detected_reps"],
            "baseline_abs_error": pair.get("baseline_abs_error", ""),
            "condition_abs_error": row["absolute_rep_count_error"],
            "predicted_form_label": row["predicted_form_label"],
            "confidence_status": row["confidence_status"],
            "pose_detection_coverage": row["pose_detection_coverage"],
            "required_joint_availability": row["required_joint_availability"],
            "angle_calculability": row["angle_calculability"],
            "movement_signal_range": row["movement_signal_range"],
            "failure_categories": row["failure_categories"],
            "notes": "Representative derived robustness case; not an independent participant sample.",
        })
        if len(output) >= 30:
            break
    return output


def markdown_table(rows, fields):
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def write_protocol(output_dir):
    lines = [
        "# Phase 12 Robustness Test Protocol",
        "",
        "Phase 12 tests how the frozen final visual pipeline behaves when exercise-video input is degraded. It does not tune models, thresholds, labels, or app behaviour.",
        "",
        "## Paired Design",
        "",
        "The same 23 original source videos are analysed in baseline form and under deterministic derived robustness conditions. Each robustness variant inherits the source video ID, anonymous participant ID, exercise, verified label, camera view, and expected repetition count from its original recording.",
        "",
        "Robustness variants are derived from existing recordings and do not represent additional participants or independent observations.",
        "",
        "## Conditions And Parameters",
        "",
        "| Condition | Parameters |",
        "| --- | --- |",
    ]
    for condition in CONDITIONS:
        lines.append(f"| {condition} | {TRANSFORM_PARAMETERS[condition]} |")
    lines.extend([
        "",
        "## Metrics",
        "",
        "- Pose/data quality: processed frames, pose detection coverage, required-joint availability, angle calculability, tracking dropout, usable frame rate, and movement signal range.",
        "- Movement analysis: expected reps, detected reps, absolute rep-count error, exact-count rate, within +/-1 rate, expected label, predicted label, and form-label match.",
        "- Confidence behaviour: HIGH/LOW status, confidence reason codes, and whether coaching would be blocked.",
        "- Failure categories: POSE_NOT_DETECTED, LOW_REQUIRED_JOINT_AVAILABILITY, LOW_ANGLE_CALCULABILITY, HIGH_TRACKING_DROPOUT, NO_REPS_DETECTED, REP_COUNT_ERROR_GT_1, FORM_MISMATCH, LOW_CONFIDENCE, COACHING_BLOCKED.",
        "",
        "## Synthetic Condition Limitation",
        "",
        "Controlled transformations are proxies. A digitally rotated video is not identical to a physically tilted camera, temporal compression is not identical to a participant naturally moving faster, and artificial occlusion is not equivalent to every real-world obstruction. These variants are paired stress-test inputs, not new dataset samples.",
    ])
    text = "\n".join(lines) + "\n"
    (Path(output_dir) / "ROBUSTNESS_TEST_PROTOCOL.md").write_text(text, encoding="utf-8")
    (ROOT_DIR / "docs" / "ROBUSTNESS_TEST_PROTOCOL.md").write_text(text, encoding="utf-8")


def write_results_doc(output_dir, summaries, exercise_summary, view_summary, failures, stats):
    baseline = next(row for row in summaries if row["condition"] == "baseline")
    worst = max(
        [row for row in summaries if row["condition"] != "baseline"],
        key=lambda row: (float(row["delta_rep_mae_vs_baseline"]), -float(row["delta_form_label_match_vs_baseline"])),
    )
    lines = [
        "# Phase 12 Robustness Test Results",
        "",
        "These results use controlled derived variants of the frozen 23-video benchmark. Derived robustness videos are paired stress-test inputs and are not counted as additional participants or independent dataset samples.",
        "",
        "## Baseline",
        "",
        markdown_table([baseline], [
            "condition",
            "video_count",
            "mean_rep_count_mae",
            "exact_count_rate",
            "within_plus_minus_1_rate",
            "form_label_match_rate",
            "high_confidence_rate",
            "low_confidence_rate",
            "mean_pose_detection_coverage",
            "mean_required_joint_availability",
        ]),
        "",
        "## Condition Summary",
        "",
        markdown_table(summaries, [
            "condition",
            "video_count",
            "mean_rep_count_mae",
            "delta_rep_mae_vs_baseline",
            "exact_count_rate",
            "delta_exact_count_rate_vs_baseline",
            "within_plus_minus_1_rate",
            "form_label_match_rate",
            "high_confidence_rate",
            "low_confidence_rate",
            "coaching_block_rate",
            "mean_pose_detection_coverage",
            "mean_required_joint_availability",
            "mean_angle_calculability",
        ]),
        "",
        f"Largest degradation by rep-count MAE delta: `{worst['condition']}`.",
        "",
        "## Exercise Summary",
        "",
        markdown_table(exercise_summary, [
            "group_value",
            "condition",
            "video_count",
            "mean_rep_count_mae",
            "within_plus_minus_1_rate",
            "form_label_match_rate",
            "high_confidence_rate",
            "mean_required_joint_availability",
        ]),
        "",
        "## View Summary",
        "",
        markdown_table(view_summary, [
            "group_value",
            "condition",
            "video_count",
            "mean_rep_count_mae",
            "within_plus_minus_1_rate",
            "form_label_match_rate",
            "high_confidence_rate",
            "mean_required_joint_availability",
        ]),
        "",
        "## Representative Failure Cases",
        "",
        markdown_table(failures[:12], [
            "source_video_id",
            "exercise",
            "verified_label",
            "camera_view",
            "condition",
            "baseline_detected_reps",
            "condition_detected_reps",
            "condition_abs_error",
            "predicted_form_label",
            "confidence_status",
            "failure_categories",
        ]) if failures else "No representative failures were selected.",
        "",
        "## Paired Statistics",
        "",
        markdown_table(stats, [
            "condition",
            "video_count",
            "mean_abs_error_difference",
            "median_abs_error_difference",
            "improved_count",
            "unchanged_count",
            "worsened_count",
            "wilcoxon_statistic",
            "wilcoxon_p_value",
            "statistical_notes",
        ]),
        "",
        "## Limitations",
        "",
        "- Synthetic transformations are controlled proxies, not independently recorded real-world conditions.",
        "- Robustness variants do not increase participant diversity and are not counted as new source videos.",
        "- Results show sensitivity of the frozen visual/movement/confidence pipeline; they do not perform the full combined AI error-propagation study reserved for Phase 13.",
        "- No models, movement thresholds, confidence thresholds, or labels were changed during Phase 12.",
    ]
    text = "\n".join(lines) + "\n"
    (Path(output_dir) / "ROBUSTNESS_TEST_RESULTS.md").write_text(text, encoding="utf-8")
    (ROOT_DIR / "docs" / "ROBUSTNESS_TEST_RESULTS.md").write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    video_output_dir = Path(args.video_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_csv(args.manifest)
    if args.max_videos:
        source_rows = source_rows[:args.max_videos]

    robustness_manifest = build_robustness_manifest(
        source_rows,
        video_output_dir,
        force_recreate=args.force_recreate_variants,
    )
    write_csv(output_dir / "robustness_manifest.csv", robustness_manifest, MANIFEST_FIELDS)
    write_protocol(output_dir)

    if args.only_create_variants:
        print(f"Wrote robustness manifest and variants for {len(robustness_manifest)} rows.")
        return 0

    result_path = output_dir / "robustness_video_results.csv"
    existing_by_id, result_rows = parse_existing_results(result_path)
    remaining = []
    for index, row in enumerate(robustness_manifest, start=1):
        if row["robustness_id"] in existing_by_id:
            print(f"[{index}/{len(robustness_manifest)}] skip {row['robustness_id']}", flush=True)
        else:
            remaining.append((index, row))

    if args.workers > 1 and remaining:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(analyze_manifest_row_worker, row): (index, row)
                for index, row in remaining
            }
            for future in as_completed(future_map):
                index, row = future_map[future]
                print(f"[{index}/{len(robustness_manifest)}] analysed {row['robustness_id']}", flush=True)
                result_rows.append(future.result())
                write_csv(result_path, result_rows, RESULT_FIELDS)
    else:
        orchestrator = FitnessCoachOrchestrator()
        for index, row in remaining:
            print(f"[{index}/{len(robustness_manifest)}] analyse {row['robustness_id']}", flush=True)
            result_rows.append(analyze_manifest_row(orchestrator, row))
            write_csv(result_path, result_rows, RESULT_FIELDS)

    condition_summary = summarize_conditions(result_rows)
    paired_rows = paired_differences(result_rows)
    exercise_summary = summarize_group(result_rows, "exercise", "exercise")
    view_summary = summarize_group(result_rows, "camera_view", "camera_view")
    stats = robustness_statistics(paired_rows)
    failures = failure_case_rows(result_rows, paired_rows)

    write_csv(output_dir / "robustness_condition_summary.csv", condition_summary, SUMMARY_FIELDS)
    write_csv(output_dir / "robustness_paired_differences.csv", paired_rows, PAIRED_FIELDS)
    write_csv(output_dir / "robustness_exercise_summary.csv", exercise_summary, GROUP_FIELDS)
    write_csv(output_dir / "robustness_view_summary.csv", view_summary, GROUP_FIELDS)
    write_csv(output_dir / "robustness_statistics.csv", stats, STAT_FIELDS)
    write_csv(output_dir / "robustness_failure_cases.csv", failures, FAILURE_FIELDS)
    write_results_doc(output_dir, condition_summary, exercise_summary, view_summary, failures, stats)

    print(f"Wrote Phase 12 robustness outputs to {output_dir}")
    print(f"Original videos: {len(source_rows)}")
    print(f"Derived robustness variants: {sum(1 for row in robustness_manifest if row['is_derived'] == 'True')}")
    print(f"Total analysed rows: {len(result_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
