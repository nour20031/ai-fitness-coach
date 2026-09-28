import argparse
import csv
import statistics
import sys
import time
import wave
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS = ["tiny", "base", "small"]
MODEL_SIZE_MB = {
    "tiny": 72.1,
    "base": 138.5,
    "small": 461.2,
}

RESULT_FIELDS = [
    "model_name",
    "library_version",
    "device",
    "audio_id",
    "category",
    "reference_text",
    "predicted_text",
    "wer",
    "exact_match",
    "failed_or_empty",
    "latency_seconds",
    "audio_duration_seconds",
    "real_time_factor",
    "audio_path",
]

SUMMARY_FIELDS = [
    "model_name",
    "library_version",
    "device",
    "model_size_mb",
    "test_recordings",
    "average_wer",
    "median_wer",
    "exact_match_rate",
    "failed_transcriptions",
    "mean_latency_seconds",
    "mean_real_time_factor",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Whisper tiny/base/small on the fixed STT test set.")
    parser.add_argument(
        "--manifest",
        default=str(ROOT_DIR / "dataset" / "stt" / "stt_test_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "dataset" / "processed" / "evaluation" / "stt"),
    )
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_text(text):
    text = text.lower().strip()
    output = []
    for character in text:
        if character.isalnum() or character.isspace():
            output.append(character)
        else:
            output.append(" ")
    return " ".join("".join(output).split())


def word_error_rate(reference, prediction):
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(prediction).split()
    if not ref_words:
        return 0 if not hyp_words else 1

    distances = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        distances[i][0] = i
    for j in range(len(hyp_words) + 1):
        distances[0][j] = j

    for i, ref_word in enumerate(ref_words, start=1):
        for j, hyp_word in enumerate(hyp_words, start=1):
            substitution = 0 if ref_word == hyp_word else 1
            distances[i][j] = min(
                distances[i - 1][j] + 1,
                distances[i][j - 1] + 1,
                distances[i - 1][j - 1] + substitution,
            )
    return distances[-1][-1] / len(ref_words)


def audio_duration(path):
    with wave.open(str(path), "rb") as file:
        frames = file.getnframes()
        rate = file.getframerate()
        return frames / rate if rate else 0


def validate_recordings(rows):
    missing = []
    for row in rows:
        audio_path = ROOT_DIR / row["audio_path"]
        if not audio_path.exists():
            missing.append(row["audio_path"])
    return missing


def load_whisper():
    try:
        import whisper
    except ImportError:
        print("The Whisper library is not installed yet.")
        print("After recording the STT test set, install the same local Whisper implementation for all variants.")
        print("Example: pip install openai-whisper")
        return None
    return whisper


def run_model(whisper, model_name, rows, device):
    download_root = ROOT_DIR / "models" / "whisper"
    download_root.mkdir(parents=True, exist_ok=True)
    model = whisper.load_model(model_name, device=device, download_root=str(download_root))
    library_version = getattr(whisper, "__version__", "unknown")
    results = []

    for index, row in enumerate(rows, start=1):
        audio_path = ROOT_DIR / row["audio_path"]
        print(f"[{model_name} {index}/{len(rows)}] {row['audio_id']}")
        duration = audio_duration(audio_path)
        started = time.perf_counter()
        try:
            result = model.transcribe(str(audio_path), language="en", fp16=False)
            prediction = result.get("text", "").strip()
        except Exception as error:
            prediction = ""
            print(f"  transcription failed: {error}")
        latency = time.perf_counter() - started
        wer = word_error_rate(row["reference_text"], prediction)
        failed = not prediction
        results.append({
            "model_name": f"whisper_{model_name}",
            "library_version": library_version,
            "device": device,
            "audio_id": row["audio_id"],
            "category": row["category"],
            "reference_text": row["reference_text"],
            "predicted_text": prediction,
            "wer": round(wer, 4),
            "exact_match": normalize_text(row["reference_text"]) == normalize_text(prediction),
            "failed_or_empty": failed,
            "latency_seconds": round(latency, 4),
            "audio_duration_seconds": round(duration, 4),
            "real_time_factor": round(latency / duration, 4) if duration else "",
            "audio_path": row["audio_path"],
        })
    return results


def summarize(rows):
    wers = [float(row["wer"]) for row in rows]
    latencies = [float(row["latency_seconds"]) for row in rows]
    rtfs = [float(row["real_time_factor"]) for row in rows if row["real_time_factor"] != ""]
    return {
        "model_name": rows[0]["model_name"],
        "library_version": rows[0]["library_version"],
        "device": rows[0]["device"],
        "model_size_mb": MODEL_SIZE_MB.get(rows[0]["model_name"].replace("whisper_", ""), ""),
        "test_recordings": len(rows),
        "average_wer": round(statistics.mean(wers), 4),
        "median_wer": round(statistics.median(wers), 4),
        "exact_match_rate": round(sum(str(row["exact_match"]).lower() == "true" for row in rows) / len(rows), 4),
        "failed_transcriptions": sum(str(row["failed_or_empty"]).lower() == "true" for row in rows),
        "mean_latency_seconds": round(statistics.mean(latencies), 4),
        "mean_real_time_factor": round(statistics.mean(rtfs), 4) if rtfs else "",
    }


def select_model(summaries):
    candidates = sorted(
        summaries,
        key=lambda row: (
            float(row["average_wer"]),
            int(row["failed_transcriptions"]),
            float(row["mean_latency_seconds"]),
        ),
    )
    return candidates[0]["model_name"]


def write_selection(output_dir, summaries):
    selected = select_model(summaries)
    lines = [
        "# Whisper Model Selection",
        "",
        f"Selected variant: `{selected}`",
        "",
        "The selection is based on measured WER, failed transcription count, and latency on the fixed recorded STT test set.",
        "",
        "| Model | Size MB | Average WER | Median WER | Exact Match Rate | Failed | Mean Latency | RTF |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['model_name']} | {row['model_size_mb']} | {row['average_wer']} | {row['median_wer']} | "
            f"{row['exact_match_rate']} | {row['failed_transcriptions']} | "
            f"{row['mean_latency_seconds']} | {row['mean_real_time_factor']} |"
        )
    (Path(output_dir) / "whisper_model_selection.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    rows = read_csv(args.manifest)
    missing = validate_recordings(rows)
    if missing:
        print("STT recordings are missing. Record the fixed test set before benchmarking.")
        print("Missing files:")
        for path in missing:
            print(f"  {path}")
        print()
        print("Run:")
        print("  .\\venv\\Scripts\\python.exe scripts\\record_stt_test_set.py")
        return 2

    whisper = load_whisper()
    if whisper is None:
        return 3

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    summaries = []
    for model_name in args.models:
        results = run_model(whisper, model_name, rows, args.device)
        all_results.extend(results)
        summaries.append(summarize(results))
        write_csv(output_dir / f"whisper_{model_name}_results.csv", results, RESULT_FIELDS)

    write_csv(output_dir / "whisper_model_comparison.csv", all_results, RESULT_FIELDS)
    write_csv(output_dir / "stt_group_summary.csv", summaries, SUMMARY_FIELDS)
    write_selection(output_dir, summaries)
    print(f"Wrote Whisper benchmark results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
