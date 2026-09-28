import argparse
import sys
from pathlib import Path

import cv2

from dataset_common import ROOT_DIR, read_csv, write_csv

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pose.landmarks import Landmarks
from src.pose.pose_detector import PoseDetector

JOINTS = {
    "squat": {
        "joint_name": "knee",
        "left": (Landmarks.LEFT_HIP, Landmarks.LEFT_KNEE, Landmarks.LEFT_ANKLE),
        "right": (Landmarks.RIGHT_HIP, Landmarks.RIGHT_KNEE, Landmarks.RIGHT_ANKLE),
    },
    "bicep_curl": {
        "joint_name": "elbow",
        "left": (Landmarks.LEFT_SHOULDER, Landmarks.LEFT_ELBOW, Landmarks.LEFT_WRIST),
        "right": (Landmarks.RIGHT_SHOULDER, Landmarks.RIGHT_ELBOW, Landmarks.RIGHT_WRIST),
    },
}

SEGMENTATION_THRESHOLDS = {
    "squat": {
        "minimum_signal_range": 10.0,
        "minimum_prominence_floor": 10.0,
        "minimum_prominence_fraction": 0.25,
        "minimum_gap_seconds": 0.45,
        "minimum_duration_seconds": 0.25,
        "maximum_duration_seconds": 6.0,
        "search_window_seconds": 4.5,
    },
    "bicep_curl": {
        "minimum_signal_range": 8.0,
        "minimum_prominence_floor": 8.0,
        "minimum_prominence_fraction": 0.22,
        "minimum_gap_seconds": 0.35,
        "minimum_duration_seconds": 0.20,
        "maximum_duration_seconds": 5.0,
        "search_window_seconds": 4.0,
    },
}


CLIP_FIELDS = [
    "source_video_id",
    "source_filename",
    "original_video",
    "rep_clip_filename",
    "participant_id",
    "exercise",
    "form_label",
    "camera_view",
    "set_number",
    "repetition_number",
    "start_time",
    "end_time",
    "clip_duration",
    "segmentation_confidence",
    "notes",
    "source_path",
    "clip_path",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Segment full-set videos into single-repetition clips.")
    parser.add_argument(
        "--manifest",
        default=str(ROOT_DIR / "dataset" / "processed" / "manifests" / "raw_videos_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT_DIR / "dataset" / "processed" / "manifests" / "rep_clips_manifest.csv"),
    )
    parser.add_argument(
        "--clips-root",
        default=str(ROOT_DIR / "dataset" / "clips"),
    )
    parser.add_argument("--padding-seconds", type=float, default=0.4)
    parser.add_argument(
        "--frame-step",
        type=int,
        default=3,
        help="Analyze every Nth frame for faster boundary detection.",
    )
    parser.add_argument("--max-videos", type=int, default=None)
    return parser.parse_args()


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def percentile(values, percentage):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentage
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def joint_visibility(landmarks, points):
    values = [
        landmarks[point].get("visibility", 0)
        for point in points
        if point in landmarks
    ]
    return min(values) if len(values) == len(points) else None


def side_angle(detector, landmarks, points):
    if not landmarks:
        return None, None
    return detector.calculate_angle(landmarks, *points), joint_visibility(landmarks, points)


def side_stats(trace, side):
    values = [row[f"{side}_angle"] for row in trace if row.get(f"{side}_angle") is not None]
    visibilities = [
        row[f"{side}_visibility"]
        for row in trace
        if row.get(f"{side}_visibility") is not None
    ]
    if not values:
        return {
            "valid_ratio": 0,
            "mean_visibility": 0,
            "signal_range": 0,
            "score": 0,
        }
    p10 = percentile(values, 0.10)
    p90 = percentile(values, 0.90)
    signal_range = max(0, (p90 or 0) - (p10 or 0))
    valid_ratio = len(values) / len(trace) if trace else 0
    mean_visibility = mean(visibilities) or 0
    return {
        "valid_ratio": valid_ratio,
        "mean_visibility": mean_visibility,
        "signal_range": signal_range,
        "score": valid_ratio * mean_visibility * signal_range,
    }


def select_signal(trace, exercise, camera_view="unknown"):
    """Select a stable movement signal using visibility and observed range."""
    if not trace:
        return trace, {
            "selected_side": "",
            "selection_reason": "empty_trace",
            "mean_pose_visibility": 0,
        }

    left = side_stats(trace, "left")
    right = side_stats(trace, "right")
    best_side = "left" if left["score"] >= right["score"] else "right"
    other_side = "right" if best_side == "left" else "left"
    best_stats = left if best_side == "left" else right
    other_stats = right if best_side == "left" else left

    use_average = (
        camera_view == "front"
        and best_stats["score"] > 0
        and (
            (exercise == "squat" and other_stats["score"] >= best_stats["score"] * 0.75)
            or (exercise == "bicep_curl" and other_stats["score"] >= best_stats["score"] * 0.50)
        )
    )

    if use_average:
        selected_side = "average"
        reason = f"front_{exercise}_average_of_both_visible_sides"
    else:
        selected_side = best_side
        reason = f"selected_{best_side}_side_by_visibility_and_signal_range"

    selected_visibilities = []
    for row in trace:
        if selected_side == "average":
            selected_values = [
                row.get("left_angle"),
                row.get("right_angle"),
            ]
            selected_visibility = mean([
                row.get("left_visibility"),
                row.get("right_visibility"),
            ])
            row["signal"] = mean(selected_values)
        else:
            selected_visibility = row.get(f"{selected_side}_visibility")
            row["signal"] = row.get(f"{selected_side}_angle")
        row["selected_side"] = selected_side
        row["selection_reason"] = reason
        row["selected_visibility"] = selected_visibility
        if selected_visibility is not None:
            selected_visibilities.append(selected_visibility)

    return trace, {
        "selected_side": selected_side,
        "selection_reason": reason,
        "left_mean_visibility": round(left["mean_visibility"], 3),
        "right_mean_visibility": round(right["mean_visibility"], 3),
        "left_signal_range": round(left["signal_range"], 3),
        "right_signal_range": round(right["signal_range"], 3),
        "mean_pose_visibility": round(mean(selected_visibilities) or 0, 3),
    }


def build_signal_trace(video_path, exercise, frame_step, camera_view="unknown"):
    detector = PoseDetector()
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    trace = []
    frame_index = 0
    joint_config = JOINTS.get(exercise)

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
        left_angle, left_visibility = (None, None)
        right_angle, right_visibility = (None, None)
        if landmarks and joint_config:
            left_angle, left_visibility = side_angle(detector, landmarks, joint_config["left"])
            right_angle, right_visibility = side_angle(detector, landmarks, joint_config["right"])
        trace.append({
            "frame": current_frame,
            "time": current_frame / fps,
            "camera_view": camera_view,
            "left_angle": left_angle,
            "right_angle": right_angle,
            "left_visibility": left_visibility,
            "right_visibility": right_visibility,
            "signal": None,
        })

    capture.release()
    trace, metadata = select_signal(trace, exercise, camera_view)
    return trace, fps, frame_index, metadata


def smooth_trace(trace, window=5):
    values = [item["signal"] for item in trace]
    smoothed = []
    for index, value in enumerate(values):
        if value is None:
            smoothed.append(None)
            continue
        start = max(0, index - window // 2)
        end = min(len(values), index + window // 2 + 1)
        local = [candidate for candidate in values[start:end] if candidate is not None]
        smoothed.append(sum(local) / len(local) if local else None)
    return smoothed


def detect_segments(trace, exercise):
    if exercise == "squat":
        return detect_squat_segments(trace)
    if exercise == "bicep_curl":
        return detect_bicep_curl_segments(trace)
    return []


def detect_squat_segments(trace):
    """Detect squat cycles using movement thresholds, not correctness thresholds."""
    values = smooth_trace(trace)
    segments = []

    active_threshold = 150
    recovery_threshold = 155
    minimum_drop = 18
    entering_active = lambda value: value is not None and value < active_threshold
    leaving_active = lambda value: value is not None and value > recovery_threshold

    in_rep = False
    start_frame = None
    min_value = None
    max_value = None

    for index, value in enumerate(values):
        if value is None:
            continue

        if not in_rep and entering_active(value):
            in_rep = True
            start_frame = trace[index]["frame"]
            min_value = value
            max_value = value
            continue

        if in_rep:
            min_value = min(min_value, value)
            max_value = max(max_value, value)
            if leaving_active(value):
                end_frame = trace[index]["frame"]
                signal_range = max_value - min_value
                confidence = "high" if signal_range >= minimum_drop else "manual_review"
                notes = "" if confidence == "high" else "low_signal_range"
                if end_frame > start_frame:
                    segments.append({
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "signal_range": round(signal_range, 2),
                        "confidence": confidence,
                        "notes": notes or "squat_cycle_thresholds",
                    })
                in_rep = False
                start_frame = None

    return segments


def detect_bicep_curl_segments(trace):
    """Detect curl cycles without requiring a full-range/correct curl angle."""
    if trace and trace[0].get("camera_view") == "side":
        return detect_flexion_extension_cycles(trace, "bicep_curl")

    smoothed = smooth_trace(trace)
    segments = []
    min_prominence = 12
    recovery_floor = 160
    valley_ceiling = 166
    previous_valley_frame = -10**9

    for index in range(1, len(smoothed) - 1):
        value = smoothed[index]
        if value is None:
            continue
        if smoothed[index - 1] is None or smoothed[index + 1] is None:
            continue
        if value > valley_ceiling:
            continue
        if not (value <= smoothed[index - 1] and value <= smoothed[index + 1]):
            continue

        valley_frame = trace[index]["frame"]
        if valley_frame - previous_valley_frame < 18:
            continue

        previous_candidates = [
            (candidate_index, candidate_value)
            for candidate_index, candidate_value in enumerate(smoothed[:index])
            if (
                candidate_value is not None
                and trace[index]["frame"] - trace[candidate_index]["frame"] <= 90
                and candidate_value >= recovery_floor
            )
        ]
        next_candidates = [
            (candidate_index, candidate_value)
            for candidate_index, candidate_value in enumerate(smoothed[index + 1:], start=index + 1)
            if (
                candidate_value is not None
                and trace[candidate_index]["frame"] - trace[index]["frame"] <= 120
                and candidate_value >= recovery_floor
            )
        ]

        if not previous_candidates or not next_candidates:
            continue

        previous_peak_index, previous_peak = max(previous_candidates, key=lambda item: item[1])
        next_peak_index, next_peak = max(next_candidates, key=lambda item: item[1])
        prominence = min(previous_peak - value, next_peak - value)

        if prominence < min_prominence:
            continue

        start_frame = trace[previous_peak_index]["frame"]
        end_frame = trace[next_peak_index]["frame"]
        if end_frame <= start_frame:
            continue

        segments.append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "signal_range": round(prominence, 2),
            "confidence": "high",
            "notes": "adaptive_bicep_cycle",
        })
        previous_valley_frame = valley_frame

    return segments


def detect_flexion_extension_cycles(trace, exercise):
    """Detect high-angle -> low-angle -> high-angle cycles without form-quality cutoffs."""
    settings = SEGMENTATION_THRESHOLDS[exercise]
    smoothed = smooth_trace(trace, window=7)
    valid_values = [value for value in smoothed if value is not None]
    segments = []
    if len(valid_values) < 3:
        return segments

    p10 = percentile(valid_values, 0.10)
    p75 = percentile(valid_values, 0.75)
    p90 = percentile(valid_values, 0.90)
    signal_range = (p90 or 0) - (p10 or 0)
    if signal_range < settings["minimum_signal_range"]:
        return segments

    min_prominence = max(
        settings["minimum_prominence_floor"],
        signal_range * settings["minimum_prominence_fraction"],
    )
    previous_valley_time = -10**9
    previous_end_time = -10**9
    local_radius = 2

    for index in range(1, len(smoothed) - 1):
        value = smoothed[index]
        if value is None:
            continue
        local = [
            candidate
            for candidate in smoothed[max(0, index - local_radius): index + local_radius + 1]
            if candidate is not None
        ]
        if not local or value != min(local):
            continue
        if p75 is not None and value > p75:
            continue

        valley_time = trace[index]["time"]
        if valley_time - previous_valley_time < settings["minimum_gap_seconds"]:
            continue

        previous_candidates = []
        for candidate_index, candidate_value in enumerate(smoothed[:index]):
            if candidate_value is None:
                continue
            candidate_time = trace[candidate_index]["time"]
            if valley_time - candidate_time > settings["search_window_seconds"]:
                continue
            if candidate_time < previous_end_time - 0.05:
                continue
            previous_candidates.append((candidate_index, candidate_value))

        next_candidates = []
        for candidate_index, candidate_value in enumerate(smoothed[index + 1:], start=index + 1):
            if candidate_value is None:
                continue
            candidate_time = trace[candidate_index]["time"]
            if candidate_time - valley_time > settings["search_window_seconds"]:
                break
            next_candidates.append((candidate_index, candidate_value))

        if not previous_candidates or not next_candidates:
            continue

        previous_peak_index, previous_peak = max(previous_candidates, key=lambda item: item[1])
        next_peak_index, next_peak = max(next_candidates, key=lambda item: item[1])
        prominence = min(previous_peak - value, next_peak - value)
        if prominence < min_prominence:
            continue

        start_frame = trace[previous_peak_index]["frame"]
        end_frame = trace[next_peak_index]["frame"]
        start_time = trace[previous_peak_index]["time"]
        end_time = trace[next_peak_index]["time"]
        duration = end_time - start_time

        if end_frame <= start_frame:
            continue
        if duration < settings["minimum_duration_seconds"] or duration > settings["maximum_duration_seconds"]:
            continue

        segments.append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "signal_range": round(prominence, 2),
            "confidence": "high",
            "notes": f"adaptive_{exercise}_cycle",
        })
        previous_valley_time = valley_time
        previous_end_time = end_time

    return segments


def write_clip(source_path, destination_path, start_frame, end_frame, fps, padding_seconds):
    capture = cv2.VideoCapture(str(source_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    padded_start = max(0, int(start_frame - padding_seconds * fps))
    padded_end = min(frame_count - 1, int(end_frame + padding_seconds * fps))
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(destination_path), fourcc, fps, (width, height))

    capture.set(cv2.CAP_PROP_POS_FRAMES, padded_start)
    current = padded_start
    while current <= padded_end:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        current += 1

    capture.release()
    writer.release()
    return padded_start / fps, padded_end / fps


def process_row(row, clips_root, padding_seconds, frame_step):
    source_path = Path(row["processed_path"])
    trace, fps, _, metadata = build_signal_trace(
        source_path,
        row["exercise"],
        frame_step,
        row.get("camera_view", "unknown"),
    )
    segments = detect_segments(trace, row["exercise"])
    clip_rows = []

    for index, segment in enumerate(segments, start=1):
        stem = Path(row["new_filename"]).stem
        clip_filename = f"{stem}_rep{index:02d}.mp4"
        clip_path = (
            Path(clips_root)
            / row["exercise"]
            / row["form_label"]
            / row["camera_view"]
            / clip_filename
        )
        start_time, end_time = write_clip(
            source_path,
            clip_path,
            segment["start_frame"],
            segment["end_frame"],
            fps,
            padding_seconds,
        )
        clip_rows.append({
            "source_video_id": row.get("source_video_id", ""),
            "source_filename": row.get("source_filename", ""),
            "original_video": row["new_filename"],
            "rep_clip_filename": clip_filename,
            "participant_id": row["participant_id"],
            "exercise": row["exercise"],
            "form_label": row["form_label"],
            "camera_view": row["camera_view"],
            "set_number": row["set_number"],
            "repetition_number": index,
            "start_time": round(start_time, 3),
            "end_time": round(end_time, 3),
            "clip_duration": round(end_time - start_time, 3),
            "segmentation_confidence": segment["confidence"],
            "notes": "; ".join(
                item for item in [
                    segment["notes"],
                    metadata.get("selection_reason", ""),
                    f"mean_visibility={metadata.get('mean_pose_visibility', '')}",
                ]
                if item
            ),
            "source_path": row["processed_path"],
            "clip_path": str(clip_path),
        })

    if not clip_rows:
        clip_rows.append({
            "source_video_id": row.get("source_video_id", ""),
            "source_filename": row.get("source_filename", ""),
            "original_video": row["new_filename"],
            "rep_clip_filename": "",
            "participant_id": row["participant_id"],
            "exercise": row["exercise"],
            "form_label": row["form_label"],
            "camera_view": row["camera_view"],
            "set_number": row["set_number"],
            "repetition_number": "",
            "start_time": "",
            "end_time": "",
            "clip_duration": "",
            "segmentation_confidence": "manual_review",
            "notes": "; ".join(
                item for item in [
                    "no_repetitions_detected",
                    metadata.get("selection_reason", ""),
                    f"mean_visibility={metadata.get('mean_pose_visibility', '')}",
                ]
                if item
            ),
            "source_path": row["processed_path"],
            "clip_path": "",
        })

    return clip_rows


def main():
    args = parse_args()
    rows = read_csv(args.manifest)
    if args.max_videos is not None:
        rows = rows[:args.max_videos]
    if not rows:
        print("No manifest rows found. Run scripts/build_dataset_manifest.py first.")
        return 1

    clip_rows = []
    for index, row in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] segmenting {row['new_filename']}")
        clip_rows.extend(process_row(row, args.clips_root, args.padding_seconds, args.frame_step))

    write_csv(args.output, clip_rows, CLIP_FIELDS)
    created = sum(1 for row in clip_rows if row["rep_clip_filename"])
    review = sum(1 for row in clip_rows if row["segmentation_confidence"] != "high")
    print(f"Wrote {created} clip rows to {args.output}")
    print(f"Rows needing manual review: {review}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
