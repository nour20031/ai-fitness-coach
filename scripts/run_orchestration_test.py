import csv
import json
import sys
import tempfile
from pathlib import Path

from dataset_common import ROOT_DIR

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.orchestration import FitnessCoachOrchestrator


MANIFEST = ROOT_DIR / "dataset" / "processed" / "evaluation" / "pose_benchmark_manifest.csv"
OUTPUT_DIR = ROOT_DIR / "dataset" / "processed" / "evaluation" / "orchestration"
RESULTS_CSV = OUTPUT_DIR / "orchestration_test_results.csv"
EXAMPLES_MD = OUTPUT_DIR / "orchestration_examples.md"
TRACE_JSON = OUTPUT_DIR / "orchestration_trace_examples.json"


RESULT_FIELDS = [
    "scenario_id",
    "description",
    "source_video_id",
    "exercise",
    "audio_id",
    "status",
    "success",
    "failure_stage",
    "coaching_blocked",
    "confidence_status",
    "detected_reps",
    "form_label",
    "detected_issues",
    "whisper_status",
    "whisper_transcription",
    "gemma_status",
    "final_response_excerpt",
    "movement_processing_seconds",
    "whisper_latency_seconds",
    "gemma_latency_seconds",
    "total_latency_seconds",
    "error",
]


def load_manifest():
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_by_video_id(rows, video_id):
    for row in rows:
        if row["source_video_id"] == video_id:
            return row
    raise KeyError(f"Could not find source video id {video_id}")


def run_scenario(orchestrator, scenario):
    if scenario.get("llm_model_tag"):
        orchestrator = FitnessCoachOrchestrator(llm_model_tag=scenario["llm_model_tag"])

    return orchestrator.run(
        video_path=scenario["video_path"],
        exercise=scenario["exercise"],
        user_question=scenario.get("user_question"),
        audio_path=scenario.get("audio_path"),
    )


def summarize_result(scenario, result):
    structured = result.get("structured_result") or {}
    transcription = result.get("transcription") or {}
    gemma = result.get("gemma") or {}
    latency = result.get("latency") or {}
    return {
        "scenario_id": scenario["scenario_id"],
        "description": scenario["description"],
        "source_video_id": scenario.get("source_video_id", ""),
        "exercise": scenario.get("exercise", ""),
        "audio_id": scenario.get("audio_id", ""),
        "status": result.get("status", ""),
        "success": result.get("success", False),
        "failure_stage": result.get("failure_stage", ""),
        "coaching_blocked": result.get("coaching_blocked", False),
        "confidence_status": structured.get("confidence_status", ""),
        "detected_reps": structured.get("rep_count", ""),
        "form_label": structured.get("form_label", ""),
        "detected_issues": ";".join(structured.get("detected_issues", [])),
        "whisper_status": transcription.get("status", ""),
        "whisper_transcription": transcription.get("transcription", ""),
        "gemma_status": gemma.get("status", ""),
        "final_response_excerpt": (result.get("final_response", "") or "")[:260].replace("\n", " "),
        "movement_processing_seconds": latency.get("movement_processing_seconds", 0),
        "whisper_latency_seconds": latency.get("whisper_latency_seconds", 0),
        "gemma_latency_seconds": latency.get("gemma_latency_seconds", 0),
        "total_latency_seconds": latency.get("total_latency_seconds", 0),
        "error": result.get("error", ""),
    }


def markdown_section(scenario, result):
    row = summarize_result(scenario, result)
    structured = result.get("structured_result") or {}
    transcription = result.get("transcription") or {}
    gemma = result.get("gemma") or {}
    trace = result.get("trace") or {}
    lines = [
        f"## {scenario['scenario_id']} - {scenario['description']}",
        "",
        f"- Video: `{scenario.get('source_video_id', '')}`",
        f"- Exercise: `{scenario.get('exercise', '')}`",
        f"- Status: `{row['status']}`",
        f"- Confidence: `{row['confidence_status']}`",
        f"- Coaching blocked: `{row['coaching_blocked']}`",
        f"- Detected reps: `{row['detected_reps']}`",
        f"- Form label: `{row['form_label']}`",
        f"- Whisper transcription: `{row['whisper_transcription']}`",
        f"- Gemma status: `{gemma.get('status', '')}`",
        f"- Total latency: `{row['total_latency_seconds']}` seconds",
        "",
        "Structured result excerpt:",
        "",
        "```json",
        json.dumps(
            {
                "exercise": structured.get("exercise"),
                "rep_count": structured.get("rep_count"),
                "form_label": structured.get("form_label"),
                "detected_issues": structured.get("detected_issues"),
                "measured_features": structured.get("measured_features"),
                "confidence_status": structured.get("confidence_status"),
                "confidence_reasons": structured.get("confidence_reasons"),
            },
            indent=2,
        ),
        "```",
        "",
        "Final response:",
        "",
        result.get("final_response", ""),
        "",
        "Trace stages:",
        "",
        "```json",
        json.dumps(list((trace.get("stage_outputs") or {}).keys()), indent=2),
        "```",
        "",
    ]
    if transcription.get("error"):
        lines.extend(["Transcription error:", "", transcription["error"], ""])
    if gemma.get("error"):
        lines.extend(["Gemma error:", "", gemma["error"], ""])
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_manifest()
    selected = {
        "vid16": row_by_video_id(rows, "vid16"),
        "vid24": row_by_video_id(rows, "vid24"),
        "vid05": row_by_video_id(rows, "vid05"),
        "vid06": row_by_video_id(rows, "vid06"),
        "vid12": row_by_video_id(rows, "vid12"),
        "vid09": row_by_video_id(rows, "vid09"),
        "vid23": row_by_video_id(rows, "vid23"),
    }

    empty_audio = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
    try:
        scenarios = [
            {
                "scenario_id": "A",
                "description": "Correct squat with text coaching request",
                "source_video_id": "vid16",
                "exercise": "squat",
                "video_path": selected["vid16"]["source_path"],
                "user_question": "Give me concise coaching feedback for this squat set.",
            },
            {
                "scenario_id": "B",
                "description": "Shallow squat with coaching request",
                "source_video_id": "vid24",
                "exercise": "squat",
                "video_path": selected["vid24"]["source_path"],
                "user_question": "Why was this squat marked as shallow?",
            },
            {
                "scenario_id": "C",
                "description": "Correct bicep curl",
                "source_video_id": "vid06",
                "exercise": "bicep_curl",
                "video_path": selected["vid06"]["source_path"],
                "user_question": "Did I do the curls correctly?",
            },
            {
                "scenario_id": "D",
                "description": "Half-range bicep curl",
                "source_video_id": "vid12",
                "exercise": "bicep_curl",
                "video_path": selected["vid12"]["source_path"],
                "user_question": "Why was my curl marked as half range?",
            },
            {
                "scenario_id": "E",
                "description": "Complete three-model path with real recorded voice question",
                "source_video_id": "vid09",
                "exercise": "bicep_curl",
                "video_path": selected["vid09"]["source_path"],
                "audio_id": "stt_002",
                "audio_path": str(ROOT_DIR / "dataset" / "stt" / "audio" / "stt_002.wav"),
            },
            {
                "scenario_id": "E2",
                "description": "Clean three-model path with accurate 10-rep result",
                "source_video_id": "vid16",
                "exercise": "squat",
                "video_path": selected["vid16"]["source_path"],
                "audio_id": "stt_002",
                "audio_path": str(ROOT_DIR / "dataset" / "stt" / "audio" / "stt_002.wav"),
            },
            {
                "scenario_id": "F",
                "description": "LOW-confidence shallow squat blocked from coaching",
                "source_video_id": "vid23",
                "exercise": "squat",
                "video_path": selected["vid23"]["source_path"],
                "user_question": "Can you tell me if my squats were good?",
            },
            {
                "scenario_id": "G",
                "description": "Whisper empty-audio failure handling",
                "source_video_id": "vid05",
                "exercise": "bicep_curl",
                "video_path": selected["vid05"]["source_path"],
                "audio_id": "empty_audio",
                "audio_path": str(empty_audio),
            },
            {
                "scenario_id": "H",
                "description": "Ollama/Gemma unavailable handling",
                "source_video_id": "vid05",
                "exercise": "bicep_curl",
                "video_path": selected["vid05"]["source_path"],
                "user_question": "Give me concise coaching feedback.",
                "llm_model_tag": "missing-gemma-model-for-test",
            },
            {
                "scenario_id": "I",
                "description": "STT error propagation demonstration",
                "source_video_id": "vid09",
                "exercise": "bicep_curl",
                "video_path": selected["vid09"]["source_path"],
                "audio_id": "stt_005",
                "audio_path": str(ROOT_DIR / "dataset" / "stt" / "audio" / "stt_005.wav"),
            },
        ]

        orchestrator = FitnessCoachOrchestrator()
        result_rows = []
        trace_examples = {}
        markdown = [
            "# Phase 8 Orchestration Examples",
            "",
            "These examples demonstrate integration of MediaPipe Pose / BlazePose, Whisper small, and gemma2:2b through the orchestration layer. They are not a full system-performance evaluation.",
            "",
            "Scenario E2 is the clean three-model success case: the video result is reliable, Whisper transcribes the question correctly, and Gemma answers from the structured facts.",
            "",
            "Scenario E is kept as upstream movement-error propagation evidence: Gemma stays grounded to the supplied structured facts, but the structured facts already contain a repetition-count error.",
            "",
            "Scenario I is kept as STT error-propagation evidence: Whisper distorts the user question, and Gemma responds to that distorted text.",
            "",
        ]

        for scenario in scenarios:
            print(f"Running scenario {scenario['scenario_id']}: {scenario['description']}")
            result = run_scenario(orchestrator, scenario)
            result_rows.append(summarize_result(scenario, result))
            trace_examples[scenario["scenario_id"]] = result.get("trace", {})
            markdown.append(markdown_section(scenario, result))

        with RESULTS_CSV.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(result_rows)

        EXAMPLES_MD.write_text("\n".join(markdown), encoding="utf-8")
        TRACE_JSON.write_text(json.dumps(trace_examples, indent=2, default=str), encoding="utf-8")

        print(f"Wrote {RESULTS_CSV}")
        print(f"Wrote {EXAMPLES_MD}")
        print(f"Wrote {TRACE_JSON}")
    finally:
        try:
            empty_audio.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    main()
