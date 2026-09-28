import argparse
import random
from collections import defaultdict
from pathlib import Path

from dataset_common import ROOT_DIR, read_csv, write_csv


SPLIT_FIELDS = [
    "source_manifest",
    "split",
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
    parser = argparse.ArgumentParser(description="Create leakage-aware train/validation/test manifests.")
    parser.add_argument(
        "--manifest",
        default=str(ROOT_DIR / "dataset" / "processed" / "manifests" / "raw_videos_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "dataset" / "processed" / "splits"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def group_for_leakage(row):
    participant = row.get("participant_id", "unknown")
    if participant and participant != "unknown":
        return f"participant::{participant}"
    return f"video::{row['new_filename']}"


def choose_splits(groups, seed):
    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)

    if len(group_items) < 3:
        return {key: "train" for key, _ in group_items}, "too_few_groups_for_valid_test_split"

    total = sum(len(rows) for _, rows in group_items)
    train_target = total * 0.70
    validation_target = total * 0.15

    assignments = {}
    counts = {"train": 0, "validation": 0, "test": 0}

    for key, rows in group_items:
        if counts["train"] < train_target:
            split = "train"
        elif counts["validation"] < validation_target:
            split = "validation"
        else:
            split = "test"
        assignments[key] = split
        counts[split] += len(rows)

    if counts["validation"] == 0 or counts["test"] == 0:
        return assignments, "small_dataset_split_may_be_imbalanced"
    return assignments, ""


def main():
    args = parse_args()
    rows = read_csv(args.manifest)
    if not rows:
        print("No manifest rows found. Run scripts/build_dataset_manifest.py first.")
        return 1

    groups = defaultdict(list)
    for row in rows:
        groups[group_for_leakage(row)].append(row)

    assignments, warning = choose_splits(groups, args.seed)
    split_rows = {"train": [], "validation": [], "test": []}

    for key, grouped_rows in groups.items():
        split = assignments[key]
        for row in grouped_rows:
            split_rows[split].append({
                "source_manifest": str(args.manifest),
                "split": split,
                **row,
            })

    output_dir = Path(args.output_dir)
    for split, split_data in split_rows.items():
        write_csv(output_dir / f"{split}_manifest.csv", split_data, SPLIT_FIELDS)
        print(f"Wrote {len(split_data)} rows to {output_dir / f'{split}_manifest.csv'}")

    note_path = output_dir / "split_notes.md"
    note_path.write_text(
        "\n".join([
            "# Dataset Split Notes",
            "",
            f"- Random seed: {args.seed}",
            "- Leakage rule: participant-level when participant IDs are known; otherwise original-video-level.",
            "- No clips or augmented variants should cross split boundaries from their source video.",
            f"- Warning: {warning or 'none'}",
            "",
            "## Counts",
            "",
            f"- Train: {len(split_rows['train'])}",
            f"- Validation: {len(split_rows['validation'])}",
            f"- Test: {len(split_rows['test'])}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote split notes to {note_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
