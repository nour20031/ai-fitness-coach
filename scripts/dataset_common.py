import csv
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

SOURCE_CLASS_MAP = {
    "bicep curl_correct": ("bicep_curl", "correct"),
    "bicep_curl_correct": ("bicep_curl", "correct"),
    "bicep curl_incorrect": ("bicep_curl", "half_range"),
    "bicep_curl_incorrect": ("bicep_curl", "half_range"),
    "bicep_curl_half_range": ("bicep_curl", "half_range"),
    "squat_correct": ("squat", "correct"),
    "squat_incorrect": ("squat", "shallow"),
    "squat_shallow": ("squat", "shallow"),
    "squat_shallow_depth": ("squat", "shallow"),
}

EXERCISE_LABELS = {
    "squat": ["correct", "shallow"],
    "bicep_curl": ["correct", "half_range"],
}

VIEWS = ["front", "side", "unknown"]

SOURCE_FILENAME_PATTERN = re.compile(
    r"^vid(?P<video_id>\d+)_id(?P<participant_id>\d+)_(?P<view>front|side)$",
    re.IGNORECASE,
)


def normalize_token(value):
    value = value.strip().lower().replace("-", "_")
    value = re.sub(r"\s+", " ", value)
    return value


def safe_stem(value):
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "video"


def is_video_file(path):
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def ensure_dataset_dirs(root_dir=ROOT_DIR):
    for base in ["raw", "clips", "augmented"]:
        for exercise, labels in EXERCISE_LABELS.items():
            for label in labels:
                for view in VIEWS:
                    (root_dir / "dataset" / base / exercise / label / view).mkdir(
                        parents=True,
                        exist_ok=True,
                    )

    for folder in [
        "dataset/processed/manifests",
        "dataset/processed/features",
        "dataset/processed/splits",
        "dataset/processed/results",
    ]:
        (root_dir / folder).mkdir(parents=True, exist_ok=True)


def infer_exercise_label_from_source(path):
    folder = normalize_token(path.parent.name)
    return SOURCE_CLASS_MAP.get(folder)


def parse_source_filename(path):
    match = SOURCE_FILENAME_PATTERN.match(path.stem)
    if not match:
        return {
            "source_video_id": "",
            "participant_id": "unknown",
            "camera_view": "unknown",
            "parsed": False,
            "notes": "filename_pattern_not_matched",
            "video_sequence": None,
        }

    video_sequence = int(match.group("video_id"))
    participant_sequence = int(match.group("participant_id"))
    return {
        "source_video_id": f"vid{video_sequence:02d}",
        "participant_id": f"participant{participant_sequence:02d}",
        "camera_view": match.group("view").lower(),
        "parsed": True,
        "notes": "parsed_from_authoritative_filename",
        "video_sequence": video_sequence,
    }


def infer_camera_view(path):
    tokens = normalize_token(" ".join(path.parts))
    if re.search(r"\bfront\b", tokens):
        return "front", "inferred_from_path"
    if re.search(r"\bside\b", tokens):
        return "side", "inferred_from_path"
    return "unknown", "needs_manual_label"


def infer_participant_id(path):
    text = normalize_token(path.stem)
    patterns = [
        r"\bparticipant[_ ]?(\d+)\b",
        r"\bsubject[_ ]?(\d+)\b",
        r"\bvolunteer[_ ]?(\d+)\b",
        r"\bp[_ ]?(\d+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return f"participant{int(match.group(1)):02d}", "inferred_from_filename"
    return "unknown", "needs_manual_label"


def get_video_metadata(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {
            "readable": False,
            "duration_seconds": 0,
            "fps": 0,
            "width": 0,
            "height": 0,
            "frame_count": 0,
        }

    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    capture.release()

    duration = frame_count / fps if fps else 0
    return {
        "readable": True,
        "duration_seconds": round(duration, 3),
        "fps": round(fps, 3),
        "width": width,
        "height": height,
        "frame_count": frame_count,
    }


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def copy_file_preserving_source(source_path, destination_path):
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if not destination_path.exists():
        shutil.copy2(source_path, destination_path)
    return destination_path


def reset_managed_directory(path, expected_relative_parts):
    path = Path(path).resolve()
    expected_path = ROOT_DIR.joinpath(*expected_relative_parts).resolve()

    if path != expected_path:
        raise ValueError(f"Refusing to reset unexpected managed directory: {path}")

    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def group_rows(rows, *keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    return groups
