import argparse
import csv
import json
import sys
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.feedback_generator import generate_feedback
from src.pose.bicep_curl_analyzer import BicepCurlAnalyzer
from src.pose.pose_detector import PoseDetector
from src.pose.squat_analyzer import SquatAnalyzer


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

LABEL_TO_EXPECTED_ISSUES = {
    "correct": [],
    "shallow_depth": ["not_deep_enough"],
    "forward_lean": ["forward_lean"],
    "half_range": ["half_range_of_motion"],
    "body_swing": ["body_swinging"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process labelled exercise videos and export analysis results."
    )
    parser.add_argument(
        "--raw-dir",
        default=str(ROOT_DIR / "dataset" / "raw_videos"),
        help="Folder containing exercise/label/video files.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT_DIR / "dataset" / "processed" / "results" / "analysis_results.csv"),
        help="CSV file to write per-video results.",
    )
    parser.add_argument(
        "--summary-output",
        default=str(ROOT_DIR / "dataset" / "processed" / "results" / "evaluation_summary.csv"),
        help="CSV file to write grouped evaluation results.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Analyze every Nth frame. Keep 1 for most accurate rep counting.",
    )
    return parser.parse_args()


def discover_videos(raw_dir):
    raw_path = Path(raw_dir)
    videos = []

    for video_path in sorted(raw_path.rglob("*")):
        if not video_path.is_file():
            continue
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        relative_parts = video_path.relative_to(raw_path).parts
        if len(relative_parts) < 3:
            continue

        exercise = relative_parts[0]
        label = relative_parts[1]
        videos.append({
            "path": video_path,
            "exercise": exercise,
            "label": label,
        })

    return videos


def create_analyzer(exercise):
    if exercise == "squat":
        return SquatAnalyzer()

    if exercise == "bicep_curl":
        # Dataset clips may start mid-rep, so do not require a ready pose.
        return BicepCurlAnalyzer(require_ready=False)

    return None


def count_issues(rep_history):
    issue_counts = {}
    for rep in rep_history:
        for issue in rep.get("issues", []):
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    return issue_counts


def flatten_issues(issue_counts):
    return sorted(issue_counts.keys())


def issue_match(label, detected_issues):
    expected_issues = LABEL_TO_EXPECTED_ISSUES.get(label, [])
    detected_set = set(detected_issues)

    if not expected_issues:
        return len(detected_set) == 0

    return all(issue in detected_set for issue in expected_issues)


def process_video(video_path, exercise, frame_step):
    analyzer = create_analyzer(exercise)
    if analyzer is None:
        raise ValueError(f"Unsupported exercise: {exercise}")

    detector = PoseDetector()
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError("Could not open video")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

    frame_index = 0
    processed_frames = 0

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame_index += 1
        if frame_step > 1 and frame_index % frame_step != 0:
            continue

        processed_frames += 1
        frame = detector.find_pose(frame)
        landmarks = detector.get_landmarks(frame)

        if landmarks:
            analyzer.analyze(detector, landmarks, frame)

    capture.release()

    if hasattr(analyzer, "finalize"):
        analyzer.finalize()

    summary = analyzer.get_session_summary()
    feedback = generate_feedback(exercise, summary)
    rep_history = summary.get("rep_history", [])
    issue_counts = summary.get("issue_counts") or count_issues(rep_history)
    detected_issues = flatten_issues(issue_counts)
    good_reps = sum(1 for rep in rep_history if not rep.get("issues"))

    return {
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "detected_reps": summary.get("total_reps", 0),
        "good_reps": good_reps,
        "detected_issues": detected_issues,
        "issue_counts": issue_counts,
        "rep_history": rep_history,
        "feedback": feedback,
    }


def build_result_row(video_item, result=None, error=None):
    video_path = video_item["path"]
    exercise = video_item["exercise"]
    label = video_item["label"]
    expected_issues = LABEL_TO_EXPECTED_ISSUES.get(label, [])

    if error:
        return {
            "status": "error",
            "video_name": video_path.name,
            "video_path": str(video_path),
            "exercise": exercise,
            "label": label,
            "expected_issues": json.dumps(expected_issues),
            "detected_reps": 0,
            "good_reps": 0,
            "detected_issues": "[]",
            "issue_counts": "{}",
            "rep_detected": False,
            "issue_match": False,
            "evaluation_match": False,
            "total_frames": 0,
            "processed_frames": 0,
            "fps": 0,
            "width": 0,
            "height": 0,
            "feedback": "",
            "rep_history": "[]",
            "error": str(error),
        }

    detected_issues = result["detected_issues"]
    rep_detected = result["detected_reps"] > 0
    issues_match = issue_match(label, detected_issues)

    return {
        "status": "ok",
        "video_name": video_path.name,
        "video_path": str(video_path),
        "exercise": exercise,
        "label": label,
        "expected_issues": json.dumps(expected_issues),
        "detected_reps": result["detected_reps"],
        "good_reps": result["good_reps"],
        "detected_issues": json.dumps(detected_issues),
        "issue_counts": json.dumps(result["issue_counts"], sort_keys=True),
        "rep_detected": rep_detected,
        "issue_match": issues_match,
        "evaluation_match": rep_detected and issues_match,
        "total_frames": result["total_frames"],
        "processed_frames": result["processed_frames"],
        "fps": result["fps"],
        "width": result["width"],
        "height": result["height"],
        "feedback": result["feedback"],
        "rep_history": json.dumps(result["rep_history"]),
        "error": "",
    }


def write_csv(path, rows, fieldnames):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_evaluation_summary(rows):
    groups = {}

    for row in rows:
        key = (row["exercise"], row["label"])
        if key not in groups:
            groups[key] = {
                "exercise": row["exercise"],
                "label": row["label"],
                "videos": 0,
                "ok_videos": 0,
                "issue_matches": 0,
                "evaluation_matches": 0,
                "total_detected_reps": 0,
                "total_good_reps": 0,
                "error_videos": 0,
            }

        group = groups[key]
        group["videos"] += 1

        if row["status"] == "ok":
            group["ok_videos"] += 1
            group["total_detected_reps"] += int(row["detected_reps"])
            group["total_good_reps"] += int(row["good_reps"])
            if str(row["issue_match"]).lower() == "true":
                group["issue_matches"] += 1
            if str(row["evaluation_match"]).lower() == "true":
                group["evaluation_matches"] += 1
        else:
            group["error_videos"] += 1

    summary_rows = []
    for group in groups.values():
        ok_videos = group["ok_videos"]
        match_rate = 0
        evaluation_match_rate = 0
        average_reps = 0

        if ok_videos:
            match_rate = round(group["issue_matches"] / ok_videos, 3)
            evaluation_match_rate = round(group["evaluation_matches"] / ok_videos, 3)
            average_reps = round(group["total_detected_reps"] / ok_videos, 2)

        summary_rows.append({
            **group,
            "issue_match_rate": match_rate,
            "evaluation_match_rate": evaluation_match_rate,
            "average_detected_reps": average_reps,
        })

    return sorted(summary_rows, key=lambda row: (row["exercise"], row["label"]))


def main():
    args = parse_args()
    frame_step = max(args.frame_step, 1)
    videos = discover_videos(args.raw_dir)

    if args.max_videos is not None:
        videos = videos[:args.max_videos]

    if not videos:
        print("No videos found.")
        return 1

    print(f"Found {len(videos)} video(s).")
    rows = []

    for index, video_item in enumerate(videos, start=1):
        video_path = video_item["path"]
        exercise = video_item["exercise"]
        label = video_item["label"]
        print(f"[{index}/{len(videos)}] {exercise}/{label}/{video_path.name}")

        try:
            result = process_video(video_path, exercise, frame_step)
            row = build_result_row(video_item, result=result)
            print(
                f"  reps={row['detected_reps']} "
                f"issues={row['detected_issues']} "
                f"match={row['issue_match']}"
            )
        except Exception as error:
            row = build_result_row(video_item, error=error)
            print(f"  ERROR: {error}")

        rows.append(row)

    result_fields = [
        "status",
        "video_name",
        "video_path",
        "exercise",
        "label",
        "expected_issues",
        "detected_reps",
        "good_reps",
        "detected_issues",
        "issue_counts",
        "rep_detected",
        "issue_match",
        "evaluation_match",
        "total_frames",
        "processed_frames",
        "fps",
        "width",
        "height",
        "feedback",
        "rep_history",
        "error",
    ]

    write_csv(args.output, rows, result_fields)
    print(f"Wrote per-video results to {args.output}")

    summary_rows = build_evaluation_summary(rows)
    summary_fields = [
        "exercise",
        "label",
        "videos",
        "ok_videos",
        "issue_matches",
        "evaluation_matches",
        "issue_match_rate",
        "evaluation_match_rate",
        "total_detected_reps",
        "total_good_reps",
        "average_detected_reps",
        "error_videos",
    ]
    write_csv(args.summary_output, summary_rows, summary_fields)
    print(f"Wrote evaluation summary to {args.summary_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
