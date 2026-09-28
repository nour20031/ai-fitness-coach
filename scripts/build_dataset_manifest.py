import argparse
from collections import defaultdict
from pathlib import Path

from dataset_common import (
    ROOT_DIR,
    copy_file_preserving_source,
    ensure_dataset_dirs,
    get_video_metadata,
    infer_exercise_label_from_source,
    is_video_file,
    parse_source_filename,
    reset_managed_directory,
    write_csv,
)


MANIFEST_FIELDS = [
    "source_filename",
    "source_video_id",
    "new_filename",
    "participant_id",
    "exercise",
    "form_label",
    "camera_view",
    "set_number",
    "duration_seconds",
    "fps",
    "width",
    "height",
    "frame_count",
    "source_path",
    "processed_path",
    "notes",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy volunteer source videos into the safe dataset/raw layout and build a manifest."
    )
    parser.add_argument(
        "--source-root",
        default=str(ROOT_DIR / "dataset" / "external" / "workoutfitness-video"),
        help="Root folder containing collected class folders.",
    )
    parser.add_argument(
        "--raw-root",
        default=str(ROOT_DIR / "dataset" / "raw"),
        help="Destination root for copied raw videos.",
    )
    parser.add_argument(
        "--manifest",
        default=str(ROOT_DIR / "dataset" / "processed" / "manifests" / "raw_videos_manifest.csv"),
    )
    parser.add_argument(
        "--manual-labels",
        default=str(ROOT_DIR / "dataset" / "processed" / "manifests" / "needs_manual_labels.csv"),
    )
    parser.add_argument(
        "--participant-summary",
        default=str(ROOT_DIR / "dataset" / "processed" / "results" / "participant_dataset_summary.csv"),
    )
    parser.add_argument(
        "--no-clean-raw",
        action="store_true",
        help="Do not clear dataset/raw before rebuilding managed copies.",
    )
    return parser.parse_args()


def numeric_video_sort_key(path):
    parsed = parse_source_filename(path)
    return (
        parsed["video_sequence"] is None,
        parsed["video_sequence"] if parsed["video_sequence"] is not None else 10**9,
        path.name.lower(),
    )


def build_participant_summary(rows):
    participants = sorted({row["participant_id"] for row in rows if row["participant_id"] != "unknown"})
    fields = [
        "participant_id",
        "bicep_curl_correct_front",
        "bicep_curl_correct_side",
        "bicep_curl_half_range_front",
        "bicep_curl_half_range_side",
        "squat_correct_front",
        "squat_correct_side",
        "squat_shallow_front",
        "squat_shallow_side",
        "total_videos",
    ]
    summary_rows = []

    for participant_id in participants:
        participant_rows = [row for row in rows if row["participant_id"] == participant_id]
        summary = {field: 0 for field in fields}
        summary["participant_id"] = participant_id
        for row in participant_rows:
            key = f"{row['exercise']}_{row['form_label']}_{row['camera_view']}"
            if key in summary:
                summary[key] += 1
            summary["total_videos"] += 1
        summary_rows.append(summary)

    return summary_rows, fields


def print_participant_summary(summary_rows, fields):
    if not summary_rows:
        print("No parsed participants found.")
        return

    widths = {
        field: max(len(field), *(len(str(row[field])) for row in summary_rows))
        for field in fields
    }
    header = " | ".join(field.ljust(widths[field]) for field in fields)
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(" | ".join(str(row[field]).ljust(widths[field]) for field in fields))


def build_manifest(source_root, raw_root, clean_raw=True):
    ensure_dataset_dirs()
    source_root = Path(source_root)
    raw_root = Path(raw_root)

    if clean_raw:
        reset_managed_directory(raw_root, ("dataset", "raw"))
        ensure_dataset_dirs()

    videos = sorted(
        (path for path in source_root.rglob("*") if is_video_file(path)),
        key=numeric_video_sort_key,
    )
    sequence_by_participant_class_view = defaultdict(int)
    rows = []
    manual_rows = []

    for video_path in videos:
        class_info = infer_exercise_label_from_source(video_path)
        if class_info is None:
            continue

        exercise, label = class_info
        parsed = parse_source_filename(video_path)
        view = parsed["camera_view"]
        participant_id = parsed["participant_id"]

        class_key = (participant_id, exercise, label, view)
        sequence_by_participant_class_view[class_key] += 1
        sequence = sequence_by_participant_class_view[class_key]
        set_number = f"set{sequence:02d}"

        filename_participant = participant_id if participant_id != "unknown" else f"unknown{sequence:02d}"
        new_filename = (
            f"{filename_participant}_{exercise}_{label}_{view}_{set_number}"
            f"{video_path.suffix.lower()}"
        )
        destination = raw_root / exercise / label / view / new_filename
        copy_file_preserving_source(video_path, destination)

        metadata = get_video_metadata(destination)
        notes = []
        if not parsed["parsed"]:
            notes.append(parsed["notes"])

        row = {
            "source_filename": video_path.name,
            "source_video_id": parsed["source_video_id"],
            "new_filename": new_filename,
            "participant_id": participant_id,
            "exercise": exercise,
            "form_label": label,
            "camera_view": view,
            "set_number": set_number,
            "duration_seconds": metadata["duration_seconds"],
            "fps": metadata["fps"],
            "width": metadata["width"],
            "height": metadata["height"],
            "frame_count": metadata["frame_count"],
            "source_path": str(video_path),
            "processed_path": str(destination),
            "notes": "; ".join(notes),
        }
        rows.append(row)

        if not parsed["parsed"]:
            manual_rows.append(row)

    return rows, manual_rows


def main():
    args = parse_args()
    rows, manual_rows = build_manifest(args.source_root, args.raw_root, clean_raw=not args.no_clean_raw)
    write_csv(args.manifest, rows, MANIFEST_FIELDS)
    write_csv(args.manual_labels, manual_rows, MANIFEST_FIELDS)
    summary_rows, summary_fields = build_participant_summary(rows)
    write_csv(args.participant_summary, summary_rows, summary_fields)
    print(f"Wrote {len(rows)} raw video rows to {args.manifest}")
    print(f"Wrote {len(manual_rows)} manual-label rows to {args.manual_labels}")
    print(f"Wrote participant summary to {args.participant_summary}")
    print()
    print_participant_summary(summary_rows, summary_fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
