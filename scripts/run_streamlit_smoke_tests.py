import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

from dataset_common import ROOT_DIR

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.audio import TextToSpeechResult
from src.orchestration import FitnessCoachOrchestrator


MANIFEST = ROOT_DIR / "dataset" / "processed" / "evaluation" / "pose_benchmark_manifest.csv"
OUTPUT_DIR = ROOT_DIR / "dataset" / "processed" / "evaluation" / "streamlit"
RESULTS_CSV = OUTPUT_DIR / "streamlit_smoke_test_results.csv"
EXAMPLES_MD = OUTPUT_DIR / "streamlit_smoke_test_examples.md"


FIELDS = [
    "scenario_id",
    "description",
    "source_video_id",
    "exercise",
    "status",
    "success",
    "failure_stage",
    "coaching_blocked",
    "confidence_status",
    "detected_reps",
    "form_label",
    "whisper_status",
    "whisper_transcription",
    "gemma_status",
    "tts_status",
    "pose_preview_available",
    "final_response",
    "error",
]


class FailingTextToSpeech:
    def synthesize(self, text, output_name=None, output_dir=None):
        return TextToSpeechResult(
            status="failed",
            engine="test_failing_tts",
            error="Intentional Phase 10 UI failure-handling smoke test.",
        )


def load_manifest():
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_by_video_id(rows, video_id):
    for row in rows:
        if row["source_video_id"] == video_id:
            return row
    raise KeyError(f"Could not find source video id {video_id}")


def summarize(scenario, result, pose_preview_available=False):
    structured = result.get("structured_result") or {}
    transcription = result.get("transcription") or {}
    gemma = result.get("gemma") or {}
    tts = result.get("tts") or {}
    return {
        "scenario_id": scenario["scenario_id"],
        "description": scenario["description"],
        "source_video_id": scenario.get("source_video_id", ""),
        "exercise": scenario.get("exercise", ""),
        "status": result.get("status", ""),
        "success": result.get("success", False),
        "failure_stage": result.get("failure_stage", ""),
        "coaching_blocked": result.get("coaching_blocked", False),
        "confidence_status": structured.get("confidence_status", ""),
        "detected_reps": structured.get("rep_count", ""),
        "form_label": structured.get("form_label", ""),
        "whisper_status": transcription.get("status", ""),
        "whisper_transcription": transcription.get("transcription", ""),
        "gemma_status": gemma.get("status", ""),
        "tts_status": tts.get("status", ""),
        "pose_preview_available": pose_preview_available,
        "final_response": result.get("final_response", ""),
        "error": result.get("error", ""),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = load_manifest()
    selected = {
        "vid16": row_by_video_id(rows, "vid16"),
        "vid24": row_by_video_id(rows, "vid24"),
        "vid06": row_by_video_id(rows, "vid06"),
        "vid12": row_by_video_id(rows, "vid12"),
        "vid23": row_by_video_id(rows, "vid23"),
    }
    orchestrator = FitnessCoachOrchestrator()
    results = []
    examples = [
        "# Phase 10 Streamlit Integration Smoke Tests",
        "",
        "These are UI/backend integration smoke checks only. They are not the formal Phase 11 technical evaluation.",
        "",
    ]

    scenarios = [
        {
            "scenario_id": "P10-A",
            "description": "Correct squat uploaded-video path",
            "source_video_id": "vid16",
            "exercise": "squat",
            "video_path": selected["vid16"]["source_path"],
        },
        {
            "scenario_id": "P10-B",
            "description": "Shallow squat uploaded-video path",
            "source_video_id": "vid24",
            "exercise": "squat",
            "video_path": selected["vid24"]["source_path"],
        },
        {
            "scenario_id": "P10-C",
            "description": "Correct bicep-curl uploaded-video path",
            "source_video_id": "vid06",
            "exercise": "bicep_curl",
            "video_path": selected["vid06"]["source_path"],
        },
        {
            "scenario_id": "P10-D",
            "description": "Half-range bicep-curl uploaded-video path",
            "source_video_id": "vid12",
            "exercise": "bicep_curl",
            "video_path": selected["vid12"]["source_path"],
        },
        {
            "scenario_id": "P10-E",
            "description": "LOW-confidence re-record guidance path",
            "source_video_id": "vid23",
            "exercise": "squat",
            "video_path": selected["vid23"]["source_path"],
        },
    ]

    baseline_structured = None
    for scenario in scenarios:
        print(f"Running {scenario['scenario_id']}: {scenario['description']}")
        result = orchestrator.run(
            video_path=scenario["video_path"],
            exercise=scenario["exercise"],
            user_question="Give concise coaching feedback based on this result.",
        )
        preview = orchestrator.create_pose_preview_frame(scenario["video_path"])
        row = summarize(scenario, result, pose_preview_available=preview is not None)
        results.append(row)
        if scenario["scenario_id"] == "P10-A":
            baseline_structured = result.get("structured_result")

    follow_up = {
        "scenario_id": "P10-F",
        "description": "Text follow-up reuses stored structured result",
        "exercise": "squat",
    }
    print("Running P10-F: text follow-up")
    follow_result = orchestrator.ask_follow_up(
        baseline_structured,
        user_question="How many reps did I complete?",
    )
    results.append(summarize(follow_up, follow_result, pose_preview_available=True))

    voice = {
        "scenario_id": "P10-G",
        "description": "Voice question plus visible Whisper transcription plus TTS",
        "exercise": "squat",
    }
    print("Running P10-G: voice question and TTS")
    voice_result = orchestrator.ask_follow_up(
        baseline_structured,
        audio_path=ROOT_DIR / "dataset" / "stt" / "audio" / "stt_002.wav",
        enable_tts=True,
        tts_output_name="streamlit_smoke_voice_answer.wav",
    )
    results.append(summarize(voice, voice_result, pose_preview_available=True))

    failure = {
        "scenario_id": "P10-H",
        "description": "TTS failure preserves text answer",
        "exercise": "squat",
    }
    print("Running P10-H: TTS failure handling")
    failing_orchestrator = FitnessCoachOrchestrator(tts_service=FailingTextToSpeech())
    failure_result = failing_orchestrator.ask_follow_up(
        baseline_structured,
        user_question="How many reps did I complete?",
        enable_tts=True,
    )
    results.append(summarize(failure, failure_result, pose_preview_available=True))

    ready = orchestrator.check_llm_ready()
    examples.append(f"LLM readiness: `{json.dumps(ready)}`")
    examples.append("")
    for row in results:
        examples.extend(
            [
                f"## {row['scenario_id']} - {row['description']}",
                "",
                f"- Status: `{row['status']}`",
                f"- Confidence: `{row['confidence_status']}`",
                f"- Reps: `{row['detected_reps']}`",
                f"- Form: `{row['form_label']}`",
                f"- Coaching blocked: `{row['coaching_blocked']}`",
                f"- Whisper transcription: `{row['whisper_transcription']}`",
                f"- Gemma status: `{row['gemma_status']}`",
                f"- TTS status: `{row['tts_status']}`",
                f"- Pose preview available: `{row['pose_preview_available']}`",
                "",
                "Final response:",
                "",
                row["final_response"],
                "",
            ]
        )

    write_csv(RESULTS_CSV, results)
    EXAMPLES_MD.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLES_MD.write_text("\n".join(examples), encoding="utf-8")

    print(f"Wrote {RESULTS_CSV}")
    print(f"Wrote {EXAMPLES_MD}")


if __name__ == "__main__":
    main()
