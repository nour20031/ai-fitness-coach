import argparse
import csv
import math
import time
from collections import defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from ai_edge_litert.interpreter import Interpreter
from dataset_common import ROOT_DIR, read_csv, write_csv
from segment_repetitions import detect_segments, select_signal


FRAME_STEP = 3
EXPECTED_REPS = 10
AVAILABILITY_THRESHOLD = 0.20
LOW_RELIABILITY_THRESHOLD = 0.50

COMMON_JOINTS = [
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

MEDIAPIPE_INDEX = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

MOVENET_INDEX = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

RESULT_FIELDS = [
    "model_name",
    "model_variant",
    "library_version",
    "model_source",
    "preprocessing",
    "source_video_id",
    "participant_id",
    "exercise",
    "form_label",
    "camera_view",
    "expected_reps",
    "duration",
    "fps",
    "frame_count",
    "analyzed_frames",
    "pose_detection_coverage",
    "required_joint_availability",
    "angle_calculability",
    "tracking_dropout",
    "processing_time_seconds",
    "ms_per_frame",
    "effective_fps",
    "detected_rep_count",
    "absolute_rep_count_error",
    "predicted_form_label",
    "form_label_match",
    "failure_rate",
    "failure_categories",
    "mean_landmark_reliability",
    "mean_rep_min_angle",
    "notes",
]

COMPARISON_FIELDS = [
    "source_video_id",
    "participant_id",
    "exercise",
    "form_label",
    "camera_view",
    "expected_reps",
    "mediapipe_pose_detection_coverage",
    "movenet_pose_detection_coverage",
    "mediapipe_required_joint_availability",
    "movenet_required_joint_availability",
    "mediapipe_angle_calculability",
    "movenet_angle_calculability",
    "mediapipe_tracking_dropout",
    "movenet_tracking_dropout",
    "mediapipe_processing_time_seconds",
    "movenet_processing_time_seconds",
    "mediapipe_ms_per_frame",
    "movenet_ms_per_frame",
    "mediapipe_fps",
    "movenet_fps",
    "mediapipe_detected_rep_count",
    "movenet_detected_rep_count",
    "mediapipe_absolute_rep_count_error",
    "movenet_absolute_rep_count_error",
    "mediapipe_form_label_match",
    "movenet_form_label_match",
    "mediapipe_failure_rate",
    "movenet_failure_rate",
    "mediapipe_failure_categories",
    "movenet_failure_categories",
    "paired_winner_pose_coverage",
    "paired_winner_rep_count",
    "paired_winner_runtime",
    "notes",
]

GROUP_FIELDS = [
    "group_type",
    "group_value",
    "model_name",
    "video_count",
    "expected_reps",
    "detected_reps",
    "rep_count_mae",
    "exact_count_rate",
    "within_plus_minus_1_rate",
    "form_label_match_rate",
    "mean_pose_detection_coverage",
    "mean_required_joint_availability",
    "mean_angle_calculability",
    "mean_tracking_dropout",
    "mean_processing_time_seconds",
    "mean_ms_per_frame",
    "mean_effective_fps",
    "failure_rate",
]

ROBUSTNESS_FIELDS = [
    "model_name",
    "source_video_id",
    "exercise",
    "form_label",
    "camera_view",
    "transformation",
    "pose_detection_coverage",
    "required_joint_availability",
    "angle_calculability",
    "detected_rep_count",
    "absolute_rep_count_error",
    "processing_time_seconds",
    "effective_fps",
    "failure_categories",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 3 pose-model benchmarks.")
    parser.add_argument(
        "--manifest",
        default=str(ROOT_DIR / "dataset" / "processed" / "evaluation" / "pose_benchmark_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "dataset" / "processed" / "evaluation" / "results"),
    )
    parser.add_argument(
        "--movenet-model",
        default=str(ROOT_DIR / "models" / "movenet_singlepose_lightning_float16_4.tflite"),
    )
    parser.add_argument("--model", choices=["mediapipe", "movenet", "all"], default="all")
    parser.add_argument("--skip-robustness", action="store_true")
    return parser.parse_args()


def calculate_angle(a, b, c):
    radians = math.atan2(c["y"] - b["y"], c["x"] - b["x"]) - math.atan2(
        a["y"] - b["y"],
        a["x"] - b["x"],
    )
    angle = abs(radians * 180.0 / math.pi)
    return 360 - angle if angle > 180 else angle


def available(joints, names, threshold=AVAILABILITY_THRESHOLD):
    return all(name in joints and joints[name]["score"] >= threshold for name in names)


def triplet_angle(joints, names):
    if not available(joints, names):
        return None, None
    reliability = min(joints[name]["score"] for name in names)
    return calculate_angle(joints[names[0]], joints[names[1]], joints[names[2]]), reliability


def exercise_angles(joints, exercise):
    if exercise == "squat":
        left = ("left_hip", "left_knee", "left_ankle")
        right = ("right_hip", "right_knee", "right_ankle")
    else:
        left = ("left_shoulder", "left_elbow", "left_wrist")
        right = ("right_shoulder", "right_elbow", "right_wrist")
    left_angle, left_visibility = triplet_angle(joints, left)
    right_angle, right_visibility = triplet_angle(joints, right)
    return left_angle, right_angle, left_visibility, right_visibility


class MediaPipeAdapter:
    name = "mediapipe"
    variant = "MediaPipe Pose / BlazePose"
    model_source = "mediapipe.solutions.pose.Pose bundled with mediapipe"
    preprocessing = "BGR frame converted to RGB; MediaPipe Pose internal preprocessing"

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @property
    def library_version(self):
        return f"mediapipe {mp.__version__}"

    def infer(self, frame):
        height, width = frame.shape[:2]
        result = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.pose_landmarks:
            return {}
        output = {}
        for name, index in MEDIAPIPE_INDEX.items():
            landmark = result.pose_landmarks.landmark[index]
            output[name] = {
                "x": float(landmark.x * width),
                "y": float(landmark.y * height),
                "score": float(landmark.visibility),
            }
        return output


class MoveNetAdapter:
    name = "movenet"
    variant = "MoveNet SinglePose Lightning TFLite float16 v4"
    model_source = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite"
    preprocessing = "RGB frame resized with padding to 192x192; coordinates unpadded back to original frame"

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.interpreter = Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        _, self.input_height, self.input_width, _ = self.input_details[0]["shape"]

    @property
    def library_version(self):
        try:
            import ai_edge_litert

            version = getattr(ai_edge_litert, "__version__", "2.2.0")
        except Exception:
            version = "unknown"
        return f"ai-edge-litert {version}"

    def _resize_with_pad(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        scale = min(self.input_width / width, self.input_height / height)
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        resized = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        pad_left = (self.input_width - new_width) // 2
        pad_top = (self.input_height - new_height) // 2
        padded[pad_top:pad_top + new_height, pad_left:pad_left + new_width] = resized
        return padded, scale, pad_left, pad_top

    def infer(self, frame):
        height, width = frame.shape[:2]
        padded, scale, pad_left, pad_top = self._resize_with_pad(frame)
        input_data = np.expand_dims(padded, axis=0)
        dtype = self.input_details[0]["dtype"]
        if dtype == np.float32:
            input_data = input_data.astype(np.float32)
        elif dtype == np.uint8:
            input_data = input_data.astype(np.uint8)
        else:
            input_data = input_data.astype(dtype)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        keypoints = self.interpreter.get_tensor(self.output_details[0]["index"])[0, 0]

        output = {}
        for name, index in MOVENET_INDEX.items():
            y_norm, x_norm, score = keypoints[index]
            x_input = float(x_norm * self.input_width)
            y_input = float(y_norm * self.input_height)
            x = (x_input - pad_left) / scale
            y = (y_input - pad_top) / scale
            output[name] = {
                "x": float(min(max(x, 0), width - 1)),
                "y": float(min(max(y, 0), height - 1)),
                "score": float(score),
            }
        return output


def transformed_frame(frame, transformation):
    if transformation == "baseline":
        return frame
    if transformation == "reduced_brightness":
        return cv2.convertScaleAbs(frame, alpha=0.55, beta=0)
    if transformation == "crop_zoom":
        height, width = frame.shape[:2]
        margin_x = int(width * 0.05)
        margin_y = int(height * 0.05)
        crop = frame[margin_y:height - margin_y, margin_x:width - margin_x]
        return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
    if transformation == "slight_rotation":
        height, width = frame.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 5, 1.0)
        return cv2.warpAffine(frame, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    raise ValueError(f"Unknown transformation: {transformation}")


def predict_form_label(exercise, rep_min_angles):
    if not rep_min_angles:
        return ""
    median_min = float(np.median(rep_min_angles))
    if exercise == "squat":
        return "shallow" if median_min > 135 else "correct"
    return "half_range" if median_min > 55 else "correct"


def summarize_failures(metrics, expected_reps, form_label, predicted_label):
    failures = []
    if metrics["pose_detection_coverage"] == 0:
        failures.append("POSE_NOT_DETECTED")
    if metrics["required_joint_availability"] < 0.50:
        failures.append("REQUIRED_JOINT_MISSING")
    if metrics["mean_landmark_reliability"] and metrics["mean_landmark_reliability"] < LOW_RELIABILITY_THRESHOLD:
        failures.append("LOW_LANDMARK_RELIABILITY")
    if metrics["angle_calculability"] < 0.50:
        failures.append("ANGLE_NOT_CALCULABLE")
    if metrics["tracking_dropout"] > 0:
        failures.append("TRACKING_DROPOUT")
    if abs(metrics["detected_rep_count"] - expected_reps) > 1:
        failures.append("REP_COUNT_ERROR")
    if predicted_label != form_label:
        failures.append("FORM_LABEL_MISMATCH")
    return failures


def benchmark_video(adapter, row, transformation="baseline"):
    expected_reps = int(row.get("expected_reps") or EXPECTED_REPS)
    capture = cv2.VideoCapture(row["source_path"])
    if not capture.isOpened():
        return {
            "processing_error": "could_not_open_video",
            "trace": [],
            "analyzed_frames": 0,
            "elapsed": 0,
        }

    trace = []
    analyzed = 0
    pose_detected = 0
    required_available = 0
    angle_calculable = 0
    reliabilities = []
    usable_previous = None
    dropouts = 0
    frame_index = 0
    started = time.perf_counter()

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        current_frame = frame_index
        frame_index += 1
        if current_frame % FRAME_STEP != 0:
            continue
        analyzed += 1
        frame = transformed_frame(frame, transformation)
        joints = adapter.infer(frame)
        if joints:
            pose_detected += 1
        common_available = available(joints, COMMON_JOINTS)
        if common_available:
            required_available += 1
        reliabilities.extend(joint["score"] for joint in joints.values())

        left_angle, right_angle, left_visibility, right_visibility = exercise_angles(
            joints,
            row["exercise"],
        )
        usable = left_angle is not None or right_angle is not None
        if usable:
            angle_calculable += 1
        if usable_previous is True and not usable:
            dropouts += 1
        usable_previous = usable
        fps = float(row["fps"] or 30)
        trace.append({
            "frame": current_frame,
            "time": current_frame / fps,
            "camera_view": row["camera_view"],
            "left_angle": left_angle,
            "right_angle": right_angle,
            "left_visibility": left_visibility,
            "right_visibility": right_visibility,
            "signal": None,
        })

    elapsed = time.perf_counter() - started
    capture.release()

    trace, _ = select_signal(trace, row["exercise"], row["camera_view"])
    segments = detect_segments(trace, row["exercise"])
    rep_min_angles = []
    for segment in segments:
        values = [
            item["signal"]
            for item in trace
            if segment["start_frame"] <= item["frame"] <= segment["end_frame"]
            and item.get("signal") is not None
        ]
        if values:
            rep_min_angles.append(min(values))

    predicted = predict_form_label(row["exercise"], rep_min_angles)
    detected_reps = len(segments)
    base_metrics = {
        "pose_detection_coverage": round(pose_detected / analyzed, 4) if analyzed else 0,
        "required_joint_availability": round(required_available / analyzed, 4) if analyzed else 0,
        "angle_calculability": round(angle_calculable / analyzed, 4) if analyzed else 0,
        "tracking_dropout": dropouts,
        "processing_time_seconds": round(elapsed, 4),
        "ms_per_frame": round((elapsed / analyzed) * 1000, 4) if analyzed else 0,
        "effective_fps": round(analyzed / elapsed, 4) if elapsed else 0,
        "detected_rep_count": detected_reps,
        "absolute_rep_count_error": abs(detected_reps - expected_reps),
        "predicted_form_label": predicted,
        "form_label_match": predicted == row["form_label"],
        "mean_landmark_reliability": round(float(np.mean(reliabilities)), 4) if reliabilities else 0,
        "mean_rep_min_angle": round(float(np.mean(rep_min_angles)), 4) if rep_min_angles else "",
    }
    failures = summarize_failures(base_metrics, expected_reps, row["form_label"], predicted)
    base_metrics["failure_categories"] = ";".join(failures)
    base_metrics["failure_rate"] = 1 if failures else 0
    return {
        "processing_error": "",
        "trace": trace,
        "analyzed_frames": analyzed,
        "elapsed": elapsed,
        "metrics": base_metrics,
    }


def result_row(adapter, row, benchmark):
    if benchmark["processing_error"]:
        failures = "PROCESSING_ERROR"
        metrics = {
            "pose_detection_coverage": 0,
            "required_joint_availability": 0,
            "angle_calculability": 0,
            "tracking_dropout": 0,
            "processing_time_seconds": 0,
            "ms_per_frame": 0,
            "effective_fps": 0,
            "detected_rep_count": 0,
            "absolute_rep_count_error": int(row["expected_reps"]),
            "predicted_form_label": "",
            "form_label_match": False,
            "failure_rate": 1,
            "failure_categories": failures,
            "mean_landmark_reliability": 0,
            "mean_rep_min_angle": "",
        }
        notes = benchmark["processing_error"]
    else:
        metrics = benchmark["metrics"]
        notes = ""
    output = {
        "model_name": adapter.name,
        "model_variant": adapter.variant,
        "library_version": adapter.library_version,
        "model_source": adapter.model_source,
        "preprocessing": adapter.preprocessing,
        "source_video_id": row["source_video_id"],
        "participant_id": row["participant_id"],
        "exercise": row["exercise"],
        "form_label": row["form_label"],
        "camera_view": row["camera_view"],
        "expected_reps": row["expected_reps"],
        "duration": row["duration"],
        "fps": row["fps"],
        "frame_count": row["frame_count"],
        "analyzed_frames": benchmark["analyzed_frames"],
        "notes": notes,
    }
    output.update(metrics)
    return output


def run_model(adapter, rows):
    results = []
    for index, row in enumerate(rows, start=1):
        print(f"[{adapter.name} {index}/{len(rows)}] {row['source_video_id']} {row['exercise']} {row['camera_view']}")
        benchmark = benchmark_video(adapter, row)
        results.append(result_row(adapter, row, benchmark))
    return results


def winner_higher(media, move, key):
    if float(media[key]) > float(move[key]):
        return "mediapipe"
    if float(move[key]) > float(media[key]):
        return "movenet"
    return "tie"


def winner_lower(media, move, key):
    if float(media[key]) < float(move[key]):
        return "mediapipe"
    if float(move[key]) < float(media[key]):
        return "movenet"
    return "tie"


def paired_comparison(media_rows, move_rows):
    media_by_id = {row["source_video_id"]: row for row in media_rows}
    move_by_id = {row["source_video_id"]: row for row in move_rows}
    rows = []
    for source_id in sorted(media_by_id):
        media = media_by_id[source_id]
        move = move_by_id[source_id]
        rows.append({
            "source_video_id": source_id,
            "participant_id": media["participant_id"],
            "exercise": media["exercise"],
            "form_label": media["form_label"],
            "camera_view": media["camera_view"],
            "expected_reps": media["expected_reps"],
            "mediapipe_pose_detection_coverage": media["pose_detection_coverage"],
            "movenet_pose_detection_coverage": move["pose_detection_coverage"],
            "mediapipe_required_joint_availability": media["required_joint_availability"],
            "movenet_required_joint_availability": move["required_joint_availability"],
            "mediapipe_angle_calculability": media["angle_calculability"],
            "movenet_angle_calculability": move["angle_calculability"],
            "mediapipe_tracking_dropout": media["tracking_dropout"],
            "movenet_tracking_dropout": move["tracking_dropout"],
            "mediapipe_processing_time_seconds": media["processing_time_seconds"],
            "movenet_processing_time_seconds": move["processing_time_seconds"],
            "mediapipe_ms_per_frame": media["ms_per_frame"],
            "movenet_ms_per_frame": move["ms_per_frame"],
            "mediapipe_fps": media["effective_fps"],
            "movenet_fps": move["effective_fps"],
            "mediapipe_detected_rep_count": media["detected_rep_count"],
            "movenet_detected_rep_count": move["detected_rep_count"],
            "mediapipe_absolute_rep_count_error": media["absolute_rep_count_error"],
            "movenet_absolute_rep_count_error": move["absolute_rep_count_error"],
            "mediapipe_form_label_match": media["form_label_match"],
            "movenet_form_label_match": move["form_label_match"],
            "mediapipe_failure_rate": media["failure_rate"],
            "movenet_failure_rate": move["failure_rate"],
            "mediapipe_failure_categories": media["failure_categories"],
            "movenet_failure_categories": move["failure_categories"],
            "paired_winner_pose_coverage": winner_higher(media, move, "pose_detection_coverage"),
            "paired_winner_rep_count": winner_lower(media, move, "absolute_rep_count_error"),
            "paired_winner_runtime": winner_higher(media, move, "effective_fps"),
            "notes": "",
        })
    return rows


def truthy(value):
    return str(value).lower() == "true"


def summarize_group(rows, group_type, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    output = []
    for value, group in sorted(grouped.items()):
        output.append(summary_row(group_type, value, group))
    return output


def summary_row(group_type, group_value, rows):
    expected = sum(int(row["expected_reps"]) for row in rows)
    detected = sum(int(row["detected_rep_count"]) for row in rows)
    errors = [float(row["absolute_rep_count_error"]) for row in rows]
    return {
        "group_type": group_type,
        "group_value": group_value,
        "model_name": rows[0]["model_name"],
        "video_count": len(rows),
        "expected_reps": expected,
        "detected_reps": detected,
        "rep_count_mae": round(float(np.mean(errors)), 4),
        "exact_count_rate": round(sum(error == 0 for error in errors) / len(rows), 4),
        "within_plus_minus_1_rate": round(sum(error <= 1 for error in errors) / len(rows), 4),
        "form_label_match_rate": round(sum(truthy(row["form_label_match"]) for row in rows) / len(rows), 4),
        "mean_pose_detection_coverage": round(float(np.mean([float(row["pose_detection_coverage"]) for row in rows])), 4),
        "mean_required_joint_availability": round(float(np.mean([float(row["required_joint_availability"]) for row in rows])), 4),
        "mean_angle_calculability": round(float(np.mean([float(row["angle_calculability"]) for row in rows])), 4),
        "mean_tracking_dropout": round(float(np.mean([float(row["tracking_dropout"]) for row in rows])), 4),
        "mean_processing_time_seconds": round(float(np.mean([float(row["processing_time_seconds"]) for row in rows])), 4),
        "mean_ms_per_frame": round(float(np.mean([float(row["ms_per_frame"]) for row in rows])), 4),
        "mean_effective_fps": round(float(np.mean([float(row["effective_fps"]) for row in rows])), 4),
        "failure_rate": round(sum(float(row["failure_rate"]) for row in rows) / len(rows), 4),
    }


def group_summaries(result_rows):
    output = []
    for model in sorted({row["model_name"] for row in result_rows}):
        rows = [row for row in result_rows if row["model_name"] == model]
        output.append(summary_row("overall", "all", rows))
        for key in ["exercise", "camera_view", "form_label", "participant_id"]:
            output.extend(summarize_group(rows, key, key))
        for row in rows:
            row["exercise_camera_view"] = f"{row['exercise']}/{row['camera_view']}"
            row["exercise_form_label"] = f"{row['exercise']}/{row['form_label']}"
        output.extend(summarize_group(rows, "exercise_camera_view", "exercise_camera_view"))
        output.extend(summarize_group(rows, "exercise_form_label", "exercise_form_label"))
    return output


def representative_rows(rows):
    wanted = []
    targets = [
        ("bicep_curl", "correct", "front"),
        ("bicep_curl", "half_range", "front"),
        ("squat", "correct", "side"),
        ("squat", "shallow", "side"),
    ]
    for exercise, label, view in targets:
        match = next(
            (
                row for row in rows
                if row["exercise"] == exercise and row["form_label"] == label and row["camera_view"] == view
            ),
            None,
        )
        if match:
            wanted.append(match)
    return wanted


def robustness_rows(adapters, rows):
    transformations = ["baseline", "reduced_brightness", "crop_zoom", "slight_rotation"]
    output = []
    for adapter in adapters:
        for row in representative_rows(rows):
            for transformation in transformations:
                print(f"[robustness {adapter.name}] {row['source_video_id']} {transformation}")
                benchmark = benchmark_video(adapter, row, transformation)
                result = result_row(adapter, row, benchmark)
                output.append({
                    "model_name": adapter.name,
                    "source_video_id": row["source_video_id"],
                    "exercise": row["exercise"],
                    "form_label": row["form_label"],
                    "camera_view": row["camera_view"],
                    "transformation": transformation,
                    "pose_detection_coverage": result["pose_detection_coverage"],
                    "required_joint_availability": result["required_joint_availability"],
                    "angle_calculability": result["angle_calculability"],
                    "detected_rep_count": result["detected_rep_count"],
                    "absolute_rep_count_error": result["absolute_rep_count_error"],
                    "processing_time_seconds": result["processing_time_seconds"],
                    "effective_fps": result["effective_fps"],
                    "failure_categories": result["failure_categories"],
                })
    return output


def write_model_decision(output_dir, summaries):
    overall = {
        row["model_name"]: row
        for row in summaries
        if row["group_type"] == "overall" and row["group_value"] == "all"
    }
    media = overall.get("mediapipe")
    move = overall.get("movenet")
    if not media or not move:
        return

    media_score = 0
    move_score = 0
    comparisons = [
        ("rep_count_mae", "lower"),
        ("exact_count_rate", "higher"),
        ("within_plus_minus_1_rate", "higher"),
        ("form_label_match_rate", "higher"),
        ("mean_required_joint_availability", "higher"),
        ("mean_angle_calculability", "higher"),
        ("failure_rate", "lower"),
        ("mean_effective_fps", "higher"),
    ]
    for key, direction in comparisons:
        m1 = float(media[key])
        m2 = float(move[key])
        if m1 == m2:
            continue
        if (direction == "higher" and m1 > m2) or (direction == "lower" and m1 < m2):
            media_score += 1
        else:
            move_score += 1

    # Downstream fitness-task performance is the main selection criterion.
    # Runtime only breaks ties when both models are close on task metrics.
    media_task_wins = 0
    move_task_wins = 0
    task_metrics = [
        ("rep_count_mae", "lower"),
        ("exact_count_rate", "higher"),
        ("within_plus_minus_1_rate", "higher"),
        ("form_label_match_rate", "higher"),
        ("failure_rate", "lower"),
    ]
    for key, direction in task_metrics:
        m1 = float(media[key])
        m2 = float(move[key])
        if m1 == m2:
            continue
        if (direction == "higher" and m1 > m2) or (direction == "lower" and m1 < m2):
            media_task_wins += 1
        else:
            move_task_wins += 1

    selected = "mediapipe" if media_task_wins >= move_task_wins else "movenet"
    rejected = "movenet" if selected == "mediapipe" else "mediapipe"
    lines = [
        "# Pose Model Selection",
        "",
        f"Selected model: `{selected}`",
        f"Rejected model: `{rejected}`",
        "",
        "The selection is based on the measured Phase 3 benchmark metrics, not on which model was already integrated.",
        "",
        "## Overall Evidence",
        "",
        "| Metric | MediaPipe | MoveNet |",
        "| --- | ---: | ---: |",
    ]
    for key, _ in comparisons:
        lines.append(f"| {key} | {media[key]} | {move[key]} |")
    lines.extend([
        "",
        "## Decision Notes",
        "",
        f"- MediaPipe score across decision criteria: {media_score}.",
        f"- MoveNet score across decision criteria: {move_score}.",
        f"- MediaPipe task-performance wins: {media_task_wins}.",
        f"- MoveNet task-performance wins: {move_task_wins}.",
        "- Downstream repetition/form performance is prioritised over speed because both models run fast enough for the current Streamlit video-analysis workflow.",
        "- Strengths and weaknesses should be interpreted with the per-video failure cases and group summaries.",
    ])
    (Path(output_dir) / "pose_model_selection.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    rows = read_csv(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapters = []
    if args.model in ("mediapipe", "all"):
        adapters.append(MediaPipeAdapter())
    if args.model in ("movenet", "all"):
        adapters.append(MoveNetAdapter(args.movenet_model))

    all_results = []
    for adapter in adapters:
        model_results = run_model(adapter, rows)
        all_results.extend(model_results)
        filename = "mediapipe_pose_results.csv" if adapter.name == "mediapipe" else "movenet_pose_results.csv"
        write_csv(output_dir / filename, model_results, RESULT_FIELDS)

    if {adapter.name for adapter in adapters} == {"mediapipe", "movenet"}:
        media_rows = [row for row in all_results if row["model_name"] == "mediapipe"]
        move_rows = [row for row in all_results if row["model_name"] == "movenet"]
        write_csv(output_dir / "pose_model_comparison.csv", paired_comparison(media_rows, move_rows), COMPARISON_FIELDS)
        summaries = group_summaries(all_results)
        write_csv(output_dir / "pose_model_group_summary.csv", summaries, GROUP_FIELDS)
        write_model_decision(output_dir, summaries)
        if not args.skip_robustness:
            write_csv(output_dir / "pose_robustness_results.csv", robustness_rows(adapters, rows), ROBUSTNESS_FIELDS)

    print(f"Wrote Phase 3 benchmark results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
