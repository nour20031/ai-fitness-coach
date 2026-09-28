import csv
import json
import sys
from pathlib import Path

from dataset_common import ROOT_DIR

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.audio import TextToSpeechResult, WindowsSapiTextToSpeech
from src.orchestration import FitnessCoachOrchestrator


MANIFEST = ROOT_DIR / "dataset" / "processed" / "evaluation" / "pose_benchmark_manifest.csv"
TEST_CASES_JSON = ROOT_DIR / "dataset" / "tts" / "tts_test_cases.json"
OUTPUT_DIR = ROOT_DIR / "dataset" / "processed" / "evaluation" / "tts"
AUDIO_DIR = OUTPUT_DIR / "audio"
RESULTS_CSV = OUTPUT_DIR / "tts_test_results.csv"
MANUAL_REVIEW_CSV = OUTPUT_DIR / "tts_manual_review.csv"
INTEGRATION_CSV = OUTPUT_DIR / "tts_integration_results.csv"
EXAMPLES_MD = OUTPUT_DIR / "tts_examples.md"


TTS_RESULT_FIELDS = [
    "case_id",
    "category",
    "text",
    "status",
    "audio_path",
    "latency_seconds",
    "engine",
    "voice",
    "rate",
    "audio_format",
    "audio_duration_seconds",
    "file_size_bytes",
    "error",
]

MANUAL_FIELDS = [
    "case_id",
    "audio_path",
    "text",
    "intelligibility_0_2",
    "naturalness_0_2",
    "notes",
]

INTEGRATION_FIELDS = [
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
    "whisper_status",
    "whisper_transcription",
    "gemma_status",
    "tts_status",
    "tts_audio_path",
    "final_response",
    "movement_processing_seconds",
    "whisper_latency_seconds",
    "gemma_latency_seconds",
    "tts_latency_seconds",
    "total_latency_seconds",
    "error",
]


class FailingTextToSpeech:
    def synthesize(self, text, output_name=None, output_dir=None):
        return TextToSpeechResult(
            status="failed",
            engine="test_failing_tts",
            error="Intentional Phase 9 failure-handling test.",
        )


def load_manifest():
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_by_video_id(rows, video_id):
    for row in rows:
        if row["source_video_id"] == video_id:
            return row
    raise KeyError(f"Could not find source video id {video_id}")


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_fixed_tts_cases():
    service = WindowsSapiTextToSpeech(output_dir=AUDIO_DIR)
    cases = json.loads(TEST_CASES_JSON.read_text(encoding="utf-8"))
    result_rows = []
    manual_rows = []

    for case in cases:
        output_name = f"{case['case_id']}.wav"
        print(f"Generating TTS case {case['case_id']}: {case['category']}")
        result = service.synthesize(case["text"], output_name=output_name)
        audio_path = result.audio_path
        file_size = Path(audio_path).stat().st_size if audio_path and Path(audio_path).exists() else 0
        result_rows.append(
            {
                "case_id": case["case_id"],
                "category": case["category"],
                "text": case["text"],
                "status": result.status,
                "audio_path": audio_path,
                "latency_seconds": result.latency_seconds,
                "engine": result.engine,
                "voice": result.voice,
                "rate": result.rate,
                "audio_format": result.audio_format,
                "audio_duration_seconds": result.audio_duration_seconds,
                "file_size_bytes": file_size,
                "error": result.error,
            }
        )
        manual_rows.append(
            {
                "case_id": case["case_id"],
                "audio_path": audio_path,
                "text": case["text"],
                "intelligibility_0_2": "",
                "naturalness_0_2": "",
                "notes": "",
            }
        )

    write_csv(RESULTS_CSV, result_rows, TTS_RESULT_FIELDS)
    write_csv(MANUAL_REVIEW_CSV, manual_rows, MANUAL_FIELDS)
    return result_rows


def summarize_orchestration_result(scenario, result):
    structured = result.get("structured_result") or {}
    transcription = result.get("transcription") or {}
    gemma = result.get("gemma") or {}
    tts = result.get("tts") or {}
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
        "whisper_status": transcription.get("status", ""),
        "whisper_transcription": transcription.get("transcription", ""),
        "gemma_status": gemma.get("status", ""),
        "tts_status": tts.get("status", ""),
        "tts_audio_path": tts.get("audio_path", ""),
        "final_response": result.get("final_response", ""),
        "movement_processing_seconds": latency.get("movement_processing_seconds", 0),
        "whisper_latency_seconds": latency.get("whisper_latency_seconds", 0),
        "gemma_latency_seconds": latency.get("gemma_latency_seconds", 0),
        "tts_latency_seconds": latency.get("tts_latency_seconds", 0),
        "total_latency_seconds": latency.get("total_latency_seconds", 0),
        "error": result.get("error", ""),
    }


def run_integration_tests():
    rows = load_manifest()
    selected = {
        "vid16": row_by_video_id(rows, "vid16"),
        "vid23": row_by_video_id(rows, "vid23"),
    }
    scenarios = [
        {
            "scenario_id": "P9-A",
            "description": "Clean video plus Whisper question plus Gemma answer plus TTS",
            "source_video_id": "vid16",
            "exercise": "squat",
            "video_path": selected["vid16"]["source_path"],
            "audio_id": "stt_002",
            "audio_path": str(ROOT_DIR / "dataset" / "stt" / "audio" / "stt_002.wav"),
            "tts_output_name": "integration_clean_e2.wav",
        },
        {
            "scenario_id": "P9-B",
            "description": "LOW-confidence re-record guidance spoken by TTS",
            "source_video_id": "vid23",
            "exercise": "squat",
            "video_path": selected["vid23"]["source_path"],
            "user_question": "Can you tell me if my squats were good?",
            "tts_output_name": "integration_low_confidence.wav",
        },
        {
            "scenario_id": "P9-C",
            "description": "TTS failure does not remove text coaching",
            "source_video_id": "vid16",
            "exercise": "squat",
            "video_path": selected["vid16"]["source_path"],
            "user_question": "How many reps did I complete?",
            "tts_output_name": "integration_failure_test.wav",
            "force_tts_failure": True,
        },
    ]

    result_rows = []
    markdown = [
        "# Phase 9 TTS Examples",
        "",
        "These examples demonstrate Text-to-Speech as an additional output layer after the final text response. TTS does not alter pose, movement-analysis, confidence, Whisper, or Gemma outputs.",
        "",
    ]

    for scenario in scenarios:
        print(f"Running TTS integration scenario {scenario['scenario_id']}: {scenario['description']}")
        tts_service = FailingTextToSpeech() if scenario.get("force_tts_failure") else WindowsSapiTextToSpeech(output_dir=AUDIO_DIR)
        orchestrator = FitnessCoachOrchestrator(tts_service=tts_service)
        result = orchestrator.run(
            video_path=scenario["video_path"],
            exercise=scenario["exercise"],
            user_question=scenario.get("user_question"),
            audio_path=scenario.get("audio_path"),
            enable_tts=True,
            tts_output_name=scenario.get("tts_output_name"),
        )
        row = summarize_orchestration_result(scenario, result)
        result_rows.append(row)
        markdown.extend(
            [
                f"## {scenario['scenario_id']} - {scenario['description']}",
                "",
                f"- Status: `{row['status']}`",
                f"- Confidence: `{row['confidence_status']}`",
                f"- Coaching blocked: `{row['coaching_blocked']}`",
                f"- Detected reps: `{row['detected_reps']}`",
                f"- Form label: `{row['form_label']}`",
                f"- Whisper transcription: `{row['whisper_transcription']}`",
                f"- Gemma status: `{row['gemma_status']}`",
                f"- TTS status: `{row['tts_status']}`",
                f"- TTS audio: `{row['tts_audio_path']}`",
                f"- Total latency: `{row['total_latency_seconds']}` seconds",
                "",
                "Final response:",
                "",
                row["final_response"],
                "",
            ]
        )

    write_csv(INTEGRATION_CSV, result_rows, INTEGRATION_FIELDS)
    EXAMPLES_MD.write_text("\n".join(markdown), encoding="utf-8")
    return result_rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    tts_rows = run_fixed_tts_cases()
    integration_rows = run_integration_tests()

    print(f"Wrote {RESULTS_CSV}")
    print(f"Wrote {MANUAL_REVIEW_CSV}")
    print(f"Wrote {INTEGRATION_CSV}")
    print(f"Wrote {EXAMPLES_MD}")
    print(f"Generated {sum(1 for row in tts_rows if row['status'] == 'success')} fixed-case audio files")
    print(f"Integration rows: {len(integration_rows)}")


if __name__ == "__main__":
    main()
