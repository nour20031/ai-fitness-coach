import argparse
from pathlib import Path

import cv2
import numpy as np

from dataset_common import ROOT_DIR, read_csv, write_csv


AUGMENTATION_FIELDS = [
    "augmented_filename",
    "original_source_clip",
    "augmentation_type",
    "augmentation_parameters",
    "split",
    "source_path",
    "augmented_path",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Create conservative augmentations for training data only.")
    parser.add_argument(
        "--train-manifest",
        default=str(ROOT_DIR / "dataset" / "processed" / "splits" / "train_manifest.csv"),
    )
    parser.add_argument(
        "--rep-clips-manifest",
        default=str(ROOT_DIR / "dataset" / "processed" / "manifests" / "rep_clips_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT_DIR / "dataset" / "processed" / "manifests" / "augmentation_manifest.csv"),
    )
    parser.add_argument(
        "--augmented-root",
        default=str(ROOT_DIR / "dataset" / "augmented"),
    )
    parser.add_argument("--variants-per-clip", type=int, default=1)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def training_sources(train_manifest, rep_clips_manifest):
    train_rows = read_csv(train_manifest)
    if not train_rows:
        return []

    train_originals = {row["new_filename"] for row in train_rows}
    clip_rows = [
        row for row in read_csv(rep_clips_manifest)
        if row.get("original_video") in train_originals and row.get("clip_path")
    ]
    if clip_rows:
        return [
            {
                "filename": row["rep_clip_filename"],
                "path": row["clip_path"],
                "exercise": row["exercise"],
                "label": row["label"],
                "camera_view": row["camera_view"],
            }
            for row in clip_rows
        ]

    return [
        {
            "filename": row["new_filename"],
            "path": row["processed_path"],
            "exercise": row["exercise"],
            "label": row["form_label"],
            "camera_view": row["camera_view"],
        }
        for row in train_rows
    ]


def transform_frame(frame, augmentation_type):
    if augmentation_type == "brightness01":
        return cv2.convertScaleAbs(frame, alpha=1.0, beta=18)
    if augmentation_type == "contrast01":
        return cv2.convertScaleAbs(frame, alpha=1.12, beta=0)
    if augmentation_type == "zoom01":
        height, width = frame.shape[:2]
        crop_ratio = 0.04
        x = int(width * crop_ratio)
        y = int(height * crop_ratio)
        cropped = frame[y:height - y, x:width - x]
        return cv2.resize(cropped, (width, height))
    if augmentation_type == "rotate01":
        height, width = frame.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 2.0, 1.0)
        return cv2.warpAffine(frame, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    if augmentation_type == "flip01":
        return cv2.flip(frame, 1)
    return frame


def augment_video(source_path, destination_path, augmentation_type):
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {source_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(transform_frame(frame, augmentation_type))

    capture.release()
    writer.release()


def main():
    args = parse_args()
    sources = training_sources(args.train_manifest, args.rep_clips_manifest)
    if args.max_files is not None:
        sources = sources[:args.max_files]
    if not sources:
        print("No training sources found. Run manifest, split, and optionally segmentation first.")
        return 1

    augmentation_types = ["brightness01", "contrast01", "zoom01", "rotate01", "flip01"]
    rows = []
    for source in sources:
        for variant_index in range(max(args.variants_per_clip, 1)):
            augmentation_type = augmentation_types[variant_index % len(augmentation_types)]
            stem = Path(source["filename"]).stem
            augmented_filename = f"{stem}_aug_{augmentation_type}.mp4"
            destination = (
                Path(args.augmented_root)
                / source["exercise"]
                / source["label"]
                / source["camera_view"]
                / augmented_filename
            )
            augment_video(Path(source["path"]), destination, augmentation_type)
            rows.append({
                "augmented_filename": augmented_filename,
                "original_source_clip": source["filename"],
                "augmentation_type": augmentation_type,
                "augmentation_parameters": "conservative_opencv_transform",
                "split": "train",
                "source_path": source["path"],
                "augmented_path": str(destination),
            })

    write_csv(args.output, rows, AUGMENTATION_FIELDS)
    print(f"Wrote {len(rows)} augmented training rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
