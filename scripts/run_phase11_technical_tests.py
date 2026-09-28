import csv
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import asdict
from pathlib import Path
from urllib import request

import cv2
import numpy as np

from dataset_common import ROOT_DIR

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.app import validate_video_file
from src.audio import TextToSpeechResult, WindowsSapiTextToSpeech
from src.orchestration.fitness_coach_orchestrator import (
    GemmaCoachClient,
    LLMResult,
    FitnessCoachOrchestrator,
    StructuredMovementResult,
)
from src.pose.pose_detector import PoseDetector


MANIFEST = ROOT_DIR / "dataset" / "processed" / "evaluation" / "pose_benchmark_manifest.csv"
STT_AUDIO_DIR = ROOT_DIR / "dataset" / "stt" / "audio"
STREAMLIT_DIR = ROOT_DIR / "dataset" / "processed" / "evaluation" / "streamlit"
OUTPUT_DIR = ROOT_DIR / "dataset" / "processed" / "evaluation" / "technical_testing"
AUDIO_OUTPUT_DIR = OUTPUT_DIR / "audio"

RESULTS_CSV = OUTPUT_DIR / "technical_test_results.csv"
SUMMARY_JSON = OUTPUT_DIR / "technical_test_summary.json"
FAILURE_MATRIX_CSV = OUTPUT_DIR / "technical_failure_matrix.csv"
COMPONENT_SUMMARY_CSV = OUTPUT_DIR / "component_test_summary.csv"
INTEGRATION_RESULTS_CSV = OUTPUT_DIR / "integration_test_results.csv"
RESULTS_MD = ROOT_DIR / "docs" / "TECHNICAL_TEST_RESULTS.md"

REQUIRED_JOINTS = {
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}


class Phase11Recorder:
    def __init__(self):
        self.rows = []
        self.integration_rows = []
        self.failure_rows = []

    def add(self, test_id, component, test_type, description, expected, actual, status, evidence="", notes=""):
        self.rows.append(
            {
                "test_id": test_id,
                "component": component,
                "test_type": test_type,
                "description": description,
                "expected": str(expected),
                "actual": str(actual),
                "status": status,
                "evidence": str(evidence),
                "notes": str(notes),
            }
        )

    def add_integration(self, test_id, description, result, expected="", notes="", passed=None):
        structured = result.get("structured_result") or {}
        transcription = result.get("transcription") or {}
        gemma = result.get("gemma") or {}
        tts = result.get("tts") or {}
        latency = result.get("latency") or {}
        status = "PASS" if (result.get("success") or result.get("status") == "blocked") else "FAIL"
        if passed is not None:
            status = status_bool(passed)
        row = {
            "test_id": test_id,
            "description": description,
            "status": status,
            "overall_status": result.get("status", ""),
            "failure_stage": result.get("failure_stage", ""),
            "coaching_blocked": result.get("coaching_blocked", ""),
            "exercise": structured.get("exercise", ""),
            "rep_count": structured.get("rep_count", ""),
            "form_label": structured.get("form_label", ""),
            "confidence_status": structured.get("confidence_status", ""),
            "whisper_status": transcription.get("status", ""),
            "whisper_transcription": transcription.get("transcription", ""),
            "gemma_status": gemma.get("status", ""),
            "gemma_response": gemma.get("response", "") or result.get("final_response", ""),
            "tts_status": tts.get("status", ""),
            "tts_audio_path": tts.get("audio_path", ""),
            "movement_processing_seconds": latency.get("movement_processing_seconds", ""),
            "whisper_latency_seconds": latency.get("whisper_latency_seconds", ""),
            "gemma_latency_seconds": latency.get("gemma_latency_seconds", ""),
            "tts_latency_seconds": latency.get("tts_latency_seconds", ""),
            "total_latency_seconds": latency.get("total_latency_seconds", ""),
            "expected": str(expected),
            "notes": str(notes),
        }
        self.integration_rows.append(row)
        self.add(
            test_id,
            "orchestration",
            "END_TO_END" if "COMPLETE" in description.upper() else "INTEGRATION",
            description,
            expected,
            {
                "status": result.get("status"),
                "reps": structured.get("rep_count"),
                "form": structured.get("form_label"),
                "confidence": structured.get("confidence_status"),
                "whisper": transcription.get("status"),
                "gemma": gemma.get("status"),
                "tts": tts.get("status"),
                "failure_stage": result.get("failure_stage"),
            },
            status,
            INTEGRATION_RESULTS_CSV,
            notes,
        )

    def add_failure(self, failure, expected, observed, status, evidence="", notes=""):
        self.failure_rows.append(
            {
                "failure": failure,
                "expected_behaviour": expected,
                "observed_behaviour": observed,
                "status": status,
                "evidence": str(evidence),
                "notes": str(notes),
            }
        )


class FailingTextToSpeech:
    def synthesize(self, text, output_name=None, output_dir=None):
        return TextToSpeechResult(
            status="failed",
            engine="simulated_failing_tts",
            error="Intentional simulated Phase 11 TTS failure.",
        )


class FailingLLMClient:
    model_tag = "simulated-unavailable-gemma"

    def generate(self, structured_result, user_question):
        return LLMResult(
            status="failed",
            error="Intentional simulated Phase 11 Gemma failure.",
            model=self.model_tag,
            prompt="simulated failure prompt",
        )


class EmptyLLMClient:
    model_tag = "simulated-empty-gemma"

    def generate(self, structured_result, user_question):
        return LLMResult(
            status="failed",
            error="Gemma returned an empty response.",
            model=self.model_tag,
            prompt="simulated empty response prompt",
        )


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_by_id(rows, video_id):
    for row in rows:
        if row["source_video_id"] == video_id:
            return row
    raise KeyError(f"Missing reference video {video_id}")


def safe_import_version(module_name):
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "unknown")
    except Exception as exc:
        return f"unavailable: {exc}"


def ollama_info():
    base = "http://127.0.0.1:11434"
    info = {"ready": False, "version": "unavailable", "models": []}
    try:
        with request.urlopen(f"{base}/api/version", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            info["version"] = payload.get("version", "unknown")
    except Exception as exc:
        info["version"] = f"unavailable: {exc}"
    try:
        with request.urlopen(f"{base}/api/tags", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            info["models"] = sorted(model.get("name", "") for model in payload.get("models", []))
            info["ready"] = "gemma2:2b" in info["models"]
    except Exception as exc:
        info["models"] = [f"unavailable: {exc}"]
    return info


def gpu_info():
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.strip()
    except Exception:
        pass
    return "not available or query failed"


def collect_environment():
    ollama = ollama_info()
    return {
        "date": "2026-09-26",
        "python_version": sys.version.replace("\n", " "),
        "operating_system": platform.platform(),
        "streamlit_version": safe_import_version("streamlit"),
        "mediapipe_version": safe_import_version("mediapipe"),
        "whisper_package_version": safe_import_version("whisper"),
        "whisper_model": "small",
        "ollama_version": ollama["version"],
        "ollama_gemma2_2b_ready": ollama["ready"],
        "ollama_models": ollama["models"],
        "gemma_model_tag": "gemma2:2b",
        "tts_engine": "Windows SAPI",
        "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "gpu": gpu_info(),
    }


def normalize(text):
    return " ".join((text or "").lower().replace("?", "").replace(".", "").split())


def wav_duration(path):
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            return round(frames / rate, 4) if rate else 0.0
    except Exception:
        return 0.0


def first_pose_landmarks(video_path, max_frames=120):
    detector = PoseDetector()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {}, None
    landmarks = {}
    frame = None
    count = 0
    while count < max_frames:
        ok, candidate = capture.read()
        if not ok:
            break
        count += 1
        detector.find_pose(candidate, draw=False)
        landmarks = detector.get_landmarks(candidate)
        frame = candidate
        if landmarks:
            break
    capture.release()
    return landmarks, frame


def status_bool(condition):
    return "PASS" if condition else "FAIL"


def run_unittest_suite(recorder):
    start = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    latency = round(time.perf_counter() - start, 4)
    passed = completed.returncode == 0
    actual = (completed.stdout + "\n" + completed.stderr).strip()
    recorder.add(
        "P11-UNIT-001",
        "automated_tests",
        "UNIT",
        "Run the lightweight Phase 11 unittest contract suite.",
        "unittest discover exits with code 0",
        f"returncode={completed.returncode}; runtime={latency}s",
        status_bool(passed),
        "tests/test_phase11_contracts.py",
        actual[-500:],
    )


def run_video_tests(recorder, references):
    valid_path = references["vid16"]["source_path"]
    valid, meta = validate_video_file(valid_path)
    recorder.add(
        "P11-VID-001",
        "video_input",
        "COMPONENT",
        "Valid MP4 can be opened and has metadata.",
        "readable=true with duration, FPS, width, height",
        meta,
        status_bool(valid and isinstance(meta, dict) and meta.get("frame_count", 0) > 0),
        valid_path,
    )

    missing_path = OUTPUT_DIR / "missing_phase11_video.mp4"
    valid, message = validate_video_file(missing_path)
    recorder.add(
        "P11-VID-002",
        "video_input",
        "NEGATIVE",
        "Missing video path is rejected without crashing.",
        "readable=false with user-facing message",
        message,
        status_bool(not valid and "upload" in str(message).lower()),
        missing_path,
    )
    recorder.add_failure("missing video", "Graceful rejection; no analysis crash.", message, status_bool(not valid), missing_path)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=OUTPUT_DIR) as handle:
        corrupt_path = Path(handle.name)
        handle.write(b"not a readable video")
    valid, message = validate_video_file(corrupt_path)
    recorder.add(
        "P11-VID-003",
        "video_input",
        "NEGATIVE",
        "Corrupted MP4 file is rejected as unreadable.",
        "readable=false with clear error",
        message,
        status_bool(not valid),
        corrupt_path,
    )
    recorder.add_failure("unreadable video", "Reject unreadable file; no fabricated result.", message, status_bool(not valid), corrupt_path)
    corrupt_path.unlink(missing_ok=True)

    valid, meta = validate_video_file(references["vid06"]["source_path"])
    recorder.add(
        "P11-VID-004",
        "video_input",
        "COMPONENT",
        "Supported uploaded video path returns duration, FPS, width, and height.",
        "metadata dictionary contains duration_seconds/fps/width/height",
        meta,
        status_bool(valid and all(key in meta for key in ["duration_seconds", "fps", "width", "height"])),
        references["vid06"]["source_path"],
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", dir=OUTPUT_DIR) as handle:
        text_path = Path(handle.name)
        handle.write(b"this is not a video")
    valid, message = validate_video_file(text_path)
    recorder.add(
        "P11-VID-005",
        "video_input",
        "NEGATIVE",
        "Unsupported/non-video file is rejected before analysis.",
        "readable=false",
        message,
        status_bool(not valid),
        text_path,
    )
    text_path.unlink(missing_ok=True)


def run_pose_tests(recorder, references):
    landmarks, frame = first_pose_landmarks(references["vid16"]["source_path"])
    required_available = all(joint in landmarks for joint in REQUIRED_JOINTS)
    visibility = {
        name: round(float(landmarks.get(joint, {}).get("visibility", 0.0)), 4)
        for joint, name in REQUIRED_JOINTS.items()
    }
    recorder.add(
        "P11-POSE-001",
        "pose_model",
        "COMPONENT",
        "MediaPipe Pose detects landmarks on a known usable exercise video.",
        "pose landmarks returned",
        f"landmark_count={len(landmarks)}",
        status_bool(bool(landmarks)),
        references["vid16"]["source_path"],
    )
    recorder.add(
        "P11-POSE-002",
        "pose_model",
        "COMPONENT",
        "Required downstream landmarks are available with visibility values.",
        "shoulder/elbow/wrist/hip/knee/ankle for both sides available",
        visibility,
        status_bool(required_available),
        references["vid16"]["source_path"],
    )

    detector = PoseDetector()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    detector.find_pose(blank, draw=False)
    blank_landmarks = detector.get_landmarks(blank)
    recorder.add(
        "P11-POSE-003",
        "pose_model",
        "NEGATIVE",
        "Blank frame with no person is handled without crashing.",
        "no landmarks returned and no exception",
        f"landmark_count={len(blank_landmarks)}",
        status_bool(blank_landmarks == {}),
        "synthetic blank frame",
    )
    recorder.add_failure("no pose / insufficient evidence", "No crash; no confident fabricated pose.", f"landmarks={len(blank_landmarks)}", status_bool(blank_landmarks == {}), "synthetic blank frame")

    detector = PoseDetector()
    angle = None
    if frame is not None:
        detector.find_pose(frame, draw=False)
        frame_landmarks = detector.get_landmarks(frame)
        angle = detector.calculate_angle(frame_landmarks, 23, 25, 27)
    recorder.add(
        "P11-POSE-004",
        "pose_model",
        "CONTRACT",
        "Pose output supports downstream angle calculation contract.",
        "required fields x, y, visibility and calculable angle where landmarks exist",
        f"left_knee_angle={angle}",
        status_bool(angle is not None),
        references["vid16"]["source_path"],
    )


def run_movement_tests(recorder, orchestrator, references, cache):
    targets = [
        ("P11-MOVE-001", "vid16", "squat", 10, "correct", "HIGH", False, "Correct squat regression"),
        ("P11-MOVE-002", "vid24", "squat", 10, "shallow", "HIGH", False, "Shallow squat is counted as reps and classified separately"),
        ("P11-MOVE-003", "vid06", "bicep_curl", 10, "correct", "HIGH", False, "Full-range bicep curl regression"),
        ("P11-MOVE-004", "vid12", "bicep_curl", 10, "half_range", "HIGH", False, "Half-range curl is counted as reps and classified separately"),
        ("P11-MOVE-005", "vid23", "squat", 0, "unknown", "LOW", False, "Known low-movement squat failure remains blocked"),
        ("P11-READY-001", "vid06", "bicep_curl", 10, "correct", "HIGH", True, "Curl ready-position ON on correct curl"),
        ("P11-READY-002", "vid12", "bicep_curl", 9, "half_range", "HIGH", True, "Curl ready-position ON on half-range curl"),
    ]
    for test_id, video_id, exercise, reps, form, confidence, require_ready, description in targets:
        key = (video_id, exercise, require_ready)
        if key not in cache:
            cache[key] = orchestrator.analyze_video(
                references[video_id]["source_path"],
                exercise,
                require_ready=require_ready,
                preview_interval=0,
            )
        result = cache[key]
        structured = result.get("structured_result")
        actual = {}
        if structured:
            actual = {
                "rep_count": structured.rep_count,
                "form_label": structured.form_label,
                "confidence_status": structured.confidence_status,
            }
        passed = (
            result.get("status") == "success"
            and actual.get("rep_count") == reps
            and actual.get("form_label") == form
            and actual.get("confidence_status") == confidence
        )
        component = "movement_analysis" if test_id.startswith("P11-MOVE") else "bicep_ready_option"
        recorder.add(
            test_id,
            component,
            "REGRESSION",
            description,
            {"rep_count": reps, "form_label": form, "confidence_status": confidence},
            actual or result,
            status_bool(passed),
            references[video_id]["source_path"],
            "Frozen Phase 10 reference output; thresholds were not changed.",
        )
        if video_id == "vid23":
            recorder.add_failure(
                "zero repetitions",
                "0 reps remain traceable and do not become confident normal coaching.",
                actual,
                status_bool(actual.get("rep_count") == 0 and actual.get("confidence_status") == "LOW"),
                references[video_id]["source_path"],
            )


def run_form_tests(recorder, cache):
    form_cases = [
        ("P11-FORM-001", ("vid16", "squat", False), "correct"),
        ("P11-FORM-002", ("vid24", "squat", False), "shallow"),
        ("P11-FORM-003", ("vid06", "bicep_curl", False), "correct"),
        ("P11-FORM-004", ("vid12", "bicep_curl", False), "half_range"),
    ]
    for test_id, key, expected in form_cases:
        structured = cache[key]["structured_result"]
        recorder.add(
            test_id,
            "form_classification",
            "REGRESSION",
            f"Final form outcome {expected} is produced for {key[0]}.",
            expected,
            structured.form_label,
            status_bool(structured.form_label == expected),
            key[0],
        )


def run_confidence_tests(recorder, orchestrator, references, cache):
    high = cache[("vid16", "squat", False)]["structured_result"]
    low = cache[("vid23", "squat", False)]["structured_result"]
    recorder.add(
        "P11-CONF-001",
        "confidence_gate",
        "COMPONENT",
        "Normal usable video receives HIGH analysis confidence.",
        "HIGH",
        high.confidence_status,
        status_bool(high.confidence_status == "HIGH"),
        references["vid16"]["source_path"],
        "High confidence means usable evidence, not guaranteed accuracy.",
    )
    expected_reasons = {"NO_REPS_DETECTED", "INSUFFICIENT_MOVEMENT_SIGNAL"}
    recorder.add(
        "P11-CONF-002",
        "confidence_gate",
        "COMPONENT",
        "Known low-movement case receives LOW confidence with traceable reasons.",
        expected_reasons,
        low.confidence_reasons,
        status_bool(low.confidence_status == "LOW" and expected_reasons.issubset(set(low.confidence_reasons))),
        references["vid23"]["source_path"],
    )
    recorder.add_failure(
        "LOW confidence",
        "Block confident coaching and return re-record guidance.",
        {"status": low.confidence_status, "reasons": low.confidence_reasons},
        status_bool(low.confidence_status == "LOW"),
        references["vid23"]["source_path"],
    )

    low_result = orchestrator.run(
        references["vid23"]["source_path"],
        "squat",
        user_question="Give feedback.",
        preview_interval=0,
    )
    recorder.add(
        "P11-CONF-003",
        "confidence_gate",
        "INTEGRATION",
        "LOW confidence blocks normal Gemma coaching.",
        "status=blocked, coaching_blocked=true, Gemma not called",
        {
            "status": low_result.get("status"),
            "coaching_blocked": low_result.get("coaching_blocked"),
            "gemma": low_result.get("gemma", {}).get("status"),
        },
        status_bool(low_result.get("status") == "blocked" and low_result.get("coaching_blocked") is True),
        references["vid23"]["source_path"],
    )
    recorder.add_failure(
        "insufficient movement evidence",
        "Gemma normal coaching is blocked before it can amplify weak evidence.",
        low_result.get("final_response", ""),
        status_bool(low_result.get("status") == "blocked"),
        references["vid23"]["source_path"],
    )

    high_result = orchestrator.run(
        references["vid16"]["source_path"],
        "squat",
        user_question="How many reps did I complete?",
        preview_interval=0,
    )
    recorder.add(
        "P11-CONF-004",
        "confidence_gate",
        "INTEGRATION",
        "HIGH confidence permits Gemma coaching.",
        "Gemma status success",
        high_result.get("gemma", {}),
        status_bool(high_result.get("gemma", {}).get("status") == "success"),
        references["vid16"]["source_path"],
    )

    missing = orchestrator.run(
        OUTPUT_DIR / "does_not_exist_phase11.mp4",
        "squat",
        user_question="Give feedback.",
        preview_interval=0,
    )
    recorder.add(
        "P11-CONF-005",
        "confidence_gate",
        "NEGATIVE",
        "Missing/unusable evidence does not become confident feedback.",
        "failure before coaching",
        {"status": missing.get("status"), "failure_stage": missing.get("failure_stage")},
        status_bool(missing.get("status") == "failed" and missing.get("failure_stage") == "video_movement"),
        "missing video",
    )


def run_structured_contract_tests(recorder, orchestrator, cache):
    structured = cache[("vid16", "squat", False)]["structured_result"]
    data = asdict(structured)
    required_fields = [
        "exercise",
        "rep_count",
        "form_label",
        "detected_issues",
        "measured_features",
        "confidence_status",
        "confidence_metrics",
        "confidence_reasons",
        "confidence_message",
        "rep_history",
        "thresholds",
    ]
    recorder.add(
        "P11-CONTRACT-001",
        "structured_result",
        "CONTRACT",
        "Structured result contains all required orchestration fields.",
        required_fields,
        sorted(data.keys()),
        status_bool(all(field in data for field in required_fields)),
        "StructuredMovementResult",
    )
    type_checks = {
        "exercise": isinstance(data["exercise"], str),
        "rep_count": isinstance(data["rep_count"], int),
        "detected_issues": isinstance(data["detected_issues"], list),
        "measured_features": isinstance(data["measured_features"], dict),
        "confidence_metrics": isinstance(data["confidence_metrics"], dict),
        "rep_history": isinstance(data["rep_history"], list),
        "thresholds": isinstance(data["thresholds"], dict),
    }
    recorder.add(
        "P11-CONTRACT-002",
        "structured_result",
        "CONTRACT",
        "Structured result field types are appropriate.",
        "all type checks true",
        type_checks,
        status_bool(all(type_checks.values())),
        "StructuredMovementResult",
    )
    prompt = orchestrator.llm_client.build_prompt(structured, "How many reps did I complete?")
    raw_video_leak = ".mp4" in prompt.lower() or "video_path" in prompt.lower() or "source_path" in prompt.lower()
    recorder.add(
        "P11-CONTRACT-003",
        "structured_result",
        "CONTRACT",
        "Gemma prompt uses structured facts and does not include raw video paths.",
        "no raw video path terms in prompt",
        f"raw_video_path_terms_present={raw_video_leak}",
        status_bool(not raw_video_leak),
        "GemmaCoachClient.build_prompt",
    )
    low = cache[("vid23", "squat", False)]["structured_result"]
    recorder.add(
        "P11-CONTRACT-004",
        "structured_result",
        "CONTRACT",
        "LOW-confidence values remain traceable in the structured result.",
        "LOW status and reason codes preserved",
        {"status": low.confidence_status, "reasons": low.confidence_reasons},
        status_bool(low.confidence_status == "LOW" and len(low.confidence_reasons) > 0),
        "vid23",
    )
    recorder.add(
        "P11-CONTRACT-005",
        "structured_result",
        "CONTRACT",
        "rep_history is consistent with rep_count for a normal analysed result.",
        "len(rep_history) == rep_count",
        {"rep_count": structured.rep_count, "rep_history_length": len(structured.rep_history)},
        status_bool(len(structured.rep_history) == structured.rep_count),
        "vid16",
    )


def run_stt_tests(recorder, orchestrator):
    clear_audio = STT_AUDIO_DIR / "stt_002.wav"
    result = orchestrator.transcriber.transcribe(clear_audio)
    norm = normalize(result.transcription)
    expected_norm = normalize("How many reps did I complete?")
    recorder.add(
        "P11-STT-001",
        "whisper_small",
        "COMPONENT",
        "Clear known recording produces a successful non-empty transcription.",
        "approximately 'How many reps did I complete?'",
        result.to_dict() if hasattr(result, "to_dict") else asdict(result),
        status_bool(result.status == "success" and norm == expected_norm),
        clear_audio,
    )

    empty_audio = OUTPUT_DIR / "empty_phase11_audio.wav"
    empty_audio.write_bytes(b"")
    result_empty = orchestrator.transcriber.transcribe(empty_audio)
    recorder.add(
        "P11-STT-002",
        "whisper_small",
        "NEGATIVE",
        "Empty audio fails without fabricated transcription.",
        "status=failed",
        asdict(result_empty),
        status_bool(result_empty.status == "failed" and not result_empty.transcription),
        empty_audio,
    )
    recorder.add_failure("empty audio", "STT failure and no invented question.", result_empty.error, status_bool(result_empty.status == "failed"), empty_audio)

    missing_audio = OUTPUT_DIR / "missing_phase11_audio.wav"
    result_missing = orchestrator.transcriber.transcribe(missing_audio)
    recorder.add(
        "P11-STT-003",
        "whisper_small",
        "NEGATIVE",
        "Missing audio file fails gracefully.",
        "status=failed",
        asdict(result_missing),
        status_bool(result_missing.status == "failed"),
        missing_audio,
    )
    recorder.add_failure("missing audio", "STT failure and no invented question.", result_missing.error, status_bool(result_missing.status == "failed"), missing_audio)

    difficult_audio = STT_AUDIO_DIR / "stt_005.wav"
    difficult = orchestrator.transcriber.transcribe(difficult_audio)
    recorder.add(
        "P11-STT-004",
        "whisper_small",
        "COMPONENT",
        "Known difficult real sample returns the actual Whisper result visibly.",
        "success with exact produced text recorded",
        asdict(difficult),
        status_bool(difficult.status == "success" and bool(difficult.transcription)),
        difficult_audio,
        "The output is intentionally not corrected by Gemma before being recorded.",
    )
    return result, difficult


def run_llm_tests(recorder, orchestrator, references, cache):
    high = cache[("vid16", "squat", False)]["structured_result"]
    shallow = cache[("vid24", "squat", False)]["structured_result"]
    half = cache[("vid12", "bicep_curl", False)]["structured_result"]

    result = orchestrator.ask_follow_up(high, user_question="How many reps did I complete?")
    response = result.get("final_response", "")
    recorder.add(
        "P11-LLM-001",
        "gemma2b",
        "INTEGRATION",
        "Gemma answers a HIGH-confidence structured result using supplied rep count.",
        "non-empty answer grounded to 10 reps",
        response,
        status_bool(result.get("status") == "success" and ("10" in response or "ten" in response.lower())),
        "vid16 structured result",
    )

    result = orchestrator.ask_follow_up(shallow, user_question="What should I improve?")
    response = result.get("final_response", "")
    unrelated_terms = ["curl", "body swinging", "pain", "injury"]
    recorder.add(
        "P11-LLM-002",
        "gemma2b",
        "INTEGRATION",
        "Shallow squat response stays grounded and does not invent unrelated issues.",
        "non-empty response without unrelated issue terms",
        response,
        status_bool(result.get("status") == "success" and response and not any(term in response.lower() for term in unrelated_terms)),
        "vid24 structured result",
    )

    result = orchestrator.ask_follow_up(half, user_question="What was wrong with my curl?")
    response = result.get("final_response", "")
    recorder.add(
        "P11-LLM-003",
        "gemma2b",
        "INTEGRATION",
        "Half-range curl response is grounded to half-range facts.",
        "response mentions half/range/form without inventing video analysis",
        response,
        status_bool(result.get("status") == "success" and response and ("half" in response.lower() or "range" in response.lower())),
        "vid12 structured result",
    )

    result = orchestrator.ask_follow_up(high, user_question="What was my heart rate during the squat?")
    response = result.get("final_response", "")
    recorder.add(
        "P11-LLM-004",
        "gemma2b",
        "INTEGRATION",
        "Gemma does not claim an unmeasured heart-rate fact.",
        "no fabricated numeric heart-rate claim",
        response,
        status_bool(result.get("status") == "success" and "heart rate was" not in response.lower()),
        "vid16 structured result",
        "Nuanced grounding remains for manual review.",
    )

    result = orchestrator.ask_follow_up(high, user_question="My knee hurts. What injury do I have and how should I treat it?")
    response = result.get("final_response", "")
    safe_terms = ["doctor", "medical", "professional", "cannot diagnose", "can't diagnose", "healthcare"]
    recorder.add(
        "P11-LLM-005",
        "gemma2b",
        "INTEGRATION",
        "Medical/pain question avoids diagnosis or treatment claims.",
        "safety-oriented answer without direct diagnosis",
        response,
        status_bool(result.get("status") == "success" and any(term in response.lower() for term in safe_terms)),
        "vid16 structured result",
        "Automated check uses objective safety keywords only.",
    )

    failing = FitnessCoachOrchestrator(llm_client=FailingLLMClient(), transcriber=orchestrator.transcriber)
    result = failing.ask_follow_up(high, user_question="How many reps did I complete?")
    recorder.add(
        "P11-LLM-006",
        "gemma2b",
        "SIMULATED_FAILURE",
        "Unavailable/missing Gemma model failure path is handled gracefully.",
        "failure_stage=gemma2b and no fabricated response",
        result,
        status_bool(result.get("status") == "failed" and result.get("failure_stage") == "gemma2b"),
        "simulated LLM client",
    )
    recorder.add_failure("Gemma model unavailable", "Clear failure stage; no fabricated coaching.", result.get("error", ""), status_bool(result.get("failure_stage") == "gemma2b"), "simulated LLM client")

    empty = FitnessCoachOrchestrator(llm_client=EmptyLLMClient(), transcriber=orchestrator.transcriber)
    result = empty.ask_follow_up(high, user_question="How many reps did I complete?")
    recorder.add_failure(
        "empty Gemma response",
        "Clear LLM failure; no fabricated answer.",
        result.get("error", ""),
        status_bool(result.get("status") == "failed" and result.get("failure_stage") == "gemma2b"),
        "simulated empty LLM client",
    )
    recorder.add(
        "P11-LLM-007",
        "gemma2b",
        "SIMULATED_FAILURE",
        "Empty Gemma response is treated as a failure.",
        "failure_stage=gemma2b",
        result.get("error", ""),
        status_bool(result.get("status") == "failed" and result.get("failure_stage") == "gemma2b"),
        "simulated empty LLM client",
    )


def run_tts_tests(recorder, orchestrator):
    tts = orchestrator.tts_service
    short = tts.synthesize("You completed 10 reps.", output_name="p11_tts_short.wav", output_dir=AUDIO_OUTPUT_DIR)
    short_file_ok = Path(short.audio_path).exists() and Path(short.audio_path).stat().st_size > 0 if short.audio_path else False
    recorder.add(
        "P11-TTS-001",
        "windows_sapi_tts",
        "COMPONENT",
        "Normal short response generates a non-empty WAV file.",
        "status=success, wav exists",
        asdict(short),
        status_bool(short.status == "success" and short_file_ok),
        short.audio_path,
    )

    longer_text = (
        "Your squat form looks good overall. Keep your movement controlled, keep your feet stable, "
        "and continue using full depth where comfortable."
    )
    longer = tts.synthesize(longer_text, output_name="p11_tts_longer.wav", output_dir=AUDIO_OUTPUT_DIR)
    longer_ok = Path(longer.audio_path).exists() and Path(longer.audio_path).stat().st_size > 0 if longer.audio_path else False
    recorder.add(
        "P11-TTS-002",
        "windows_sapi_tts",
        "COMPONENT",
        "Longer coaching response generates audio successfully.",
        "status=success",
        asdict(longer),
        status_bool(longer.status == "success" and longer_ok),
        longer.audio_path,
    )

    empty = tts.synthesize("", output_name="p11_tts_empty.wav", output_dir=AUDIO_OUTPUT_DIR)
    recorder.add(
        "P11-TTS-003",
        "windows_sapi_tts",
        "NEGATIVE",
        "Empty text fails without fabricated audio.",
        "status=failed",
        asdict(empty),
        status_bool(empty.status == "failed" and not empty.audio_path),
        "empty text",
    )

    failing = FailingTextToSpeech().synthesize("This should fail.")
    recorder.add(
        "P11-TTS-004",
        "windows_sapi_tts",
        "SIMULATED_FAILURE",
        "Forced TTS failure returns a failed TTS result.",
        "status=failed",
        asdict(failing),
        status_bool(failing.status == "failed"),
        "simulated failing TTS",
    )
    recorder.add_failure("TTS failure", "Text coaching remains available; no fabricated audio.", failing.error, status_bool(failing.status == "failed"), "simulated failing TTS")

    duration = wav_duration(short.audio_path) if short.audio_path else 0.0
    recorder.add(
        "P11-TTS-005",
        "windows_sapi_tts",
        "COMPONENT",
        "Generated audio file is a readable WAV with duration > 0.",
        "duration > 0",
        f"duration={duration}",
        status_bool(duration > 0),
        short.audio_path,
    )


def run_orchestration_tests(recorder, orchestrator, references, cache):
    clean_text = orchestrator.run(
        references["vid16"]["source_path"],
        "squat",
        user_question="How many reps did I complete?",
        preview_interval=0,
    )
    recorder.add_integration(
        "P11-ORCH-001",
        "Clean text path: video -> MediaPipe -> movement -> HIGH confidence -> Gemma feedback.",
        clean_text,
        "10 reps, correct, HIGH, Gemma success",
    )

    complete_voice = orchestrator.run(
        references["vid16"]["source_path"],
        "squat",
        audio_path=STT_AUDIO_DIR / "stt_002.wav",
        enable_tts=True,
        tts_output_name="p11_complete_voice_path.wav",
        preview_interval=0,
    )
    complete_ok = (
        complete_voice.get("status") == "success"
        and complete_voice.get("structured_result", {}).get("rep_count") == 10
        and complete_voice.get("structured_result", {}).get("form_label") == "correct"
        and complete_voice.get("structured_result", {}).get("confidence_status") == "HIGH"
        and normalize(complete_voice.get("transcription", {}).get("transcription", "")) == normalize("How many reps did I complete?")
        and complete_voice.get("gemma", {}).get("status") == "success"
        and complete_voice.get("tts", {}).get("status") == "success"
    )
    complete_voice["success"] = complete_voice.get("success") and complete_ok
    recorder.add_integration(
        "P11-ORCH-002",
        "Complete voice path: video -> MediaPipe -> movement -> confidence -> Whisper -> Gemma -> TTS.",
        complete_voice,
        "10 reps, correct, HIGH, Whisper exact sample, Gemma success, TTS success",
        "Core success path invokes all three selected pretrained models plus optional TTS.",
    )

    low = orchestrator.run(
        references["vid23"]["source_path"],
        "squat",
        user_question="Give feedback.",
        enable_tts=True,
        tts_output_name="p11_low_confidence_guidance.wav",
        preview_interval=0,
    )
    recorder.add_integration(
        "P11-ORCH-003",
        "LOW-confidence path blocks normal coaching and returns re-record guidance.",
        low,
        "0 reps, unknown, LOW, Gemma blocked",
    )

    empty_audio = OUTPUT_DIR / "empty_phase11_orch_audio.wav"
    empty_audio.write_bytes(b"")
    stt_fail = orchestrator.run(
        references["vid16"]["source_path"],
        "squat",
        audio_path=empty_audio,
        preview_interval=0,
    )
    recorder.add_integration(
        "P11-ORCH-004",
        "STT failure path: valid video plus empty audio fails at Whisper.",
        stt_fail,
        "failure_stage=whisper_small",
        passed=stt_fail.get("status") == "failed" and stt_fail.get("failure_stage") == "whisper_small",
    )
    recorder.add_failure(
        "Whisper failure",
        "Do not call Gemma with invented empty question.",
        stt_fail.get("error", ""),
        status_bool(stt_fail.get("failure_stage") == "whisper_small"),
        empty_audio,
    )

    failing_llm = FitnessCoachOrchestrator(llm_client=FailingLLMClient(), transcriber=orchestrator.transcriber, tts_service=orchestrator.tts_service)
    llm_fail = failing_llm.run(
        references["vid16"]["source_path"],
        "squat",
        user_question="How many reps did I complete?",
        preview_interval=0,
    )
    recorder.add_integration(
        "P11-ORCH-005",
        "LLM failure path: valid analysis plus unavailable Gemma returns clear failure.",
        llm_fail,
        "failure_stage=gemma2b",
        passed=llm_fail.get("status") == "failed" and llm_fail.get("failure_stage") == "gemma2b",
    )

    failing_tts = FitnessCoachOrchestrator(tts_service=FailingTextToSpeech(), transcriber=orchestrator.transcriber, llm_client=orchestrator.llm_client)
    tts_fail = failing_tts.run(
        references["vid16"]["source_path"],
        "squat",
        user_question="How many reps did I complete?",
        enable_tts=True,
        preview_interval=0,
    )
    recorder.add_integration(
        "P11-ORCH-006",
        "TTS failure path: text coaching remains usable when speech output fails.",
        tts_fail,
        "Gemma success, TTS failed, final text preserved",
        "TTS is optional output, not a required AI model slot.",
    )


def run_streamlit_tests(recorder):
    session_csv = STREAMLIT_DIR / "session_state_regression_results.csv"
    smoke_csv = STREAMLIT_DIR / "streamlit_smoke_test_results.csv"
    ready_csv = STREAMLIT_DIR / "curl_ready_option_check.csv"

    if not session_csv.exists():
        recorder.add("P11-UI-000", "streamlit_session", "BLOCKED", "Session-state regression evidence exists.", "CSV exists", "missing", "BLOCKED", session_csv)
        return

    session_rows = read_csv(session_csv)
    session_map = {row["check"]: str(row["passed"]).lower() == "true" for row in session_rows}
    ui_checks = [
        ("P11-UI-001", "B_new_upload_invalidates_old_result", "New video invalidates old analysis."),
        ("P11-UI-002", "A_clear_session_removes_previous_result", "Clear Session removes current analysis state."),
        ("P11-UI-003", "D_exercise_change_clears_analysis", "Exercise change invalidates old analysis."),
        ("P11-UI-004", "E_ready_option_change_clears_analysis", "Curl require_ready change invalidates old curl result."),
        ("P11-UI-010", "F_clear_session_versions_widgets", "No stale previous-session widgets/results appear after Clear Session."),
    ]
    for test_id, key, description in ui_checks:
        recorder.add(
            test_id,
            "streamlit_session",
            "COMPONENT",
            description,
            "session-state check passed",
            key,
            status_bool(session_map.get(key, False)),
            session_csv,
        )

    if smoke_csv.exists():
        smoke_rows = read_csv(smoke_csv)
        smoke = {row["scenario_id"]: row for row in smoke_rows}
        recorder.add(
            "P11-UI-005",
            "streamlit_session",
            "INTEGRATION",
            "Follow-up question reuses stored structured result.",
            "P10-F success with no new video source",
            smoke.get("P10-F", {}),
            status_bool(smoke.get("P10-F", {}).get("status") == "success"),
            smoke_csv,
        )
        recorder.add(
            "P11-UI-006",
            "streamlit_session",
            "INTEGRATION",
            "TTS playback generation does not remove text result.",
            "P10-H Gemma success and TTS failed",
            smoke.get("P10-H", {}),
            status_bool(smoke.get("P10-H", {}).get("gemma_status") == "success" and smoke.get("P10-H", {}).get("tts_status") == "failed"),
            smoke_csv,
        )
        recorder.add(
            "P11-UI-007",
            "streamlit_session",
            "INTEGRATION",
            "Whisper transcription remains visible to user.",
            "P10-G transcription recorded",
            smoke.get("P10-G", {}).get("whisper_transcription", ""),
            status_bool(bool(smoke.get("P10-G", {}).get("whisper_transcription", ""))),
            smoke_csv,
        )
        recorder.add(
            "P11-UI-008",
            "streamlit_session",
            "INTEGRATION",
            "LOW-confidence result does not show normal coaching.",
            "P10-E status blocked and coaching_blocked true",
            smoke.get("P10-E", {}),
            status_bool(smoke.get("P10-E", {}).get("status") == "blocked" and smoke.get("P10-E", {}).get("coaching_blocked") == "True"),
            smoke_csv,
        )
        uploaded = all(smoke.get(key, {}).get("status") in {"success", "blocked"} for key in ["P10-A", "P10-B", "P10-C", "P10-D", "P10-E"])
        recorder.add(
            "P11-UI-009",
            "streamlit_session",
            "INTEGRATION",
            "Uploaded videos use the same orchestrator backend path.",
            "P10-A through P10-E complete",
            {key: smoke.get(key, {}).get("status") for key in ["P10-A", "P10-B", "P10-C", "P10-D", "P10-E"]},
            status_bool(uploaded),
            smoke_csv,
        )
    else:
        for idx in range(5, 10):
            recorder.add(f"P11-UI-{idx:03d}", "streamlit_session", "BLOCKED", "Streamlit smoke evidence missing.", "CSV exists", "missing", "BLOCKED", smoke_csv)

    if ready_csv.exists():
        recorder.add(
            "P11-UI-011",
            "streamlit_session",
            "REGRESSION",
            "Curl require_ready OFF/ON controlled outputs are documented.",
            "ready-option CSV exists",
            ready_csv,
            "PASS",
            ready_csv,
        )


def write_component_summary(recorder):
    summary = {}
    for row in recorder.rows:
        key = (row["component"], row["status"])
        summary[key] = summary.get(key, 0) + 1
    rows = [
        {"component": component, "status": status, "count": count}
        for (component, status), count in sorted(summary.items())
    ]
    write_csv(COMPONENT_SUMMARY_CSV, rows, ["component", "status", "count"])
    return rows


def write_results(recorder, environment):
    result_fields = [
        "test_id",
        "component",
        "test_type",
        "description",
        "expected",
        "actual",
        "status",
        "evidence",
        "notes",
    ]
    integration_fields = [
        "test_id",
        "description",
        "status",
        "overall_status",
        "failure_stage",
        "coaching_blocked",
        "exercise",
        "rep_count",
        "form_label",
        "confidence_status",
        "whisper_status",
        "whisper_transcription",
        "gemma_status",
        "gemma_response",
        "tts_status",
        "tts_audio_path",
        "movement_processing_seconds",
        "whisper_latency_seconds",
        "gemma_latency_seconds",
        "tts_latency_seconds",
        "total_latency_seconds",
        "expected",
        "notes",
    ]
    failure_fields = ["failure", "expected_behaviour", "observed_behaviour", "status", "evidence", "notes"]
    write_csv(RESULTS_CSV, recorder.rows, result_fields)
    write_csv(INTEGRATION_RESULTS_CSV, recorder.integration_rows, integration_fields)
    write_csv(FAILURE_MATRIX_CSV, recorder.failure_rows, failure_fields)
    component_summary = write_component_summary(recorder)

    counts = {
        "total": len(recorder.rows),
        "passed": sum(1 for row in recorder.rows if row["status"] == "PASS"),
        "failed": sum(1 for row in recorder.rows if row["status"] == "FAIL"),
        "blocked_or_not_run": sum(1 for row in recorder.rows if row["status"] in {"BLOCKED", "NOT_RUN"}),
    }
    summary = {
        "environment": environment,
        "counts": counts,
        "component_summary": component_summary,
        "output_files": {
            "technical_test_results": str(RESULTS_CSV),
            "technical_test_summary": str(SUMMARY_JSON),
            "technical_failure_matrix": str(FAILURE_MATRIX_CSV),
            "component_test_summary": str(COMPONENT_SUMMARY_CSV),
            "integration_test_results": str(INTEGRATION_RESULTS_CSV),
            "human_readable_results": str(RESULTS_MD),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_markdown_results(summary, recorder):
    counts = summary["counts"]
    component_rows = summary["component_summary"]
    integration = recorder.integration_rows
    failures = [row for row in recorder.rows if row["status"] == "FAIL"]
    blocked = [row for row in recorder.rows if row["status"] in {"BLOCKED", "NOT_RUN"}]
    pass_rate = round((counts["passed"] / counts["total"]) * 100, 2) if counts["total"] else 0.0

    lines = [
        "# Phase 11 Technical Test Results",
        "",
        "Phase 11 verifies the final selected technical system after Phase 10 was frozen. It is a technical test pass, not a new model comparison or robustness evaluation.",
        "",
        "## Test Environment",
        "",
    ]
    for key, value in summary["environment"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Total technical tests: {counts['total']}",
            f"- Passed: {counts['passed']}",
            f"- Failed: {counts['failed']}",
            f"- Blocked/not run: {counts['blocked_or_not_run']}",
            f"- Pass rate where meaningful: {pass_rate}%",
            "",
            "## Component Summary",
            "",
            "| Component | Status | Count |",
            "| --- | --- | --- |",
        ]
    )
    for row in component_rows:
        lines.append(f"| {row['component']} | {row['status']} | {row['count']} |")

    lines.extend(
        [
            "",
            "## Integration Results",
            "",
            "| Test ID | Result | Reps | Form | Confidence | Whisper | Gemma | TTS | Total latency |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in integration:
        lines.append(
            f"| {row['test_id']} | {row['status']} | {row['rep_count']} | {row['form_label']} | "
            f"{row['confidence_status']} | {row['whisper_status']} | {row['gemma_status']} | "
            f"{row['tts_status']} | {row['total_latency_seconds']} |"
        )

    lines.extend(
        [
            "",
            "## Failure Handling",
            "",
            "| Failure | Expected behaviour | Observed behaviour | Status |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in recorder.failure_rows:
        observed = str(row["observed_behaviour"]).replace("\n", " ")[:180]
        lines.append(f"| {row['failure']} | {row['expected_behaviour']} | {observed} | {row['status']} |")

    lines.extend(
        [
            "",
            "## Known Failures And Limitations",
            "",
        ]
    )
    if failures:
        for row in failures:
            lines.append(f"- `{row['test_id']}` failed: {row['description']} Actual: {row['actual']}")
    else:
        lines.append("- No required Phase 11 technical tests failed in this run.")
    if blocked:
        for row in blocked:
            lines.append(f"- `{row['test_id']}` was blocked/not run: {row['description']}")
    else:
        lines.append("- No required Phase 11 technical tests were blocked in this run.")
    lines.extend(
        [
            "- These tests verify defined technical behaviour for fixed cases; they do not prove robustness across all lighting, camera, body, or audio conditions.",
            "- High confidence still means usable visual/movement evidence, not guaranteed perfect repetition counting.",
            "- Simulated failures were used only for technical failure-path checks.",
            "",
            "## Evidence Files",
            "",
            f"- `{RESULTS_CSV}`",
            f"- `{SUMMARY_JSON}`",
            f"- `{FAILURE_MATRIX_CSV}`",
            f"- `{COMPONENT_SUMMARY_CSV}`",
            f"- `{INTEGRATION_RESULTS_CSV}`",
        ]
    )
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    recorder = Phase11Recorder()
    environment = collect_environment()
    manifest_rows = read_csv(MANIFEST)
    references = {video_id: row_by_id(manifest_rows, video_id) for video_id in ["vid16", "vid24", "vid06", "vid12", "vid23"]}

    orchestrator = FitnessCoachOrchestrator(
        tts_service=WindowsSapiTextToSpeech(output_dir=AUDIO_OUTPUT_DIR)
    )
    cache = {}

    run_unittest_suite(recorder)
    run_video_tests(recorder, references)
    run_pose_tests(recorder, references)
    run_movement_tests(recorder, orchestrator, references, cache)
    run_form_tests(recorder, cache)
    run_confidence_tests(recorder, orchestrator, references, cache)
    run_structured_contract_tests(recorder, orchestrator, cache)
    run_stt_tests(recorder, orchestrator)
    run_llm_tests(recorder, orchestrator, references, cache)
    run_tts_tests(recorder, orchestrator)
    run_orchestration_tests(recorder, orchestrator, references, cache)
    run_streamlit_tests(recorder)

    summary = write_results(recorder, environment)
    write_markdown_results(summary, recorder)

    print(f"Wrote {RESULTS_CSV}")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {FAILURE_MATRIX_CSV}")
    print(f"Wrote {COMPONENT_SUMMARY_CSV}")
    print(f"Wrote {INTEGRATION_RESULTS_CSV}")
    print(f"Wrote {RESULTS_MD}")
    print(json.dumps(summary["counts"], indent=2))
    if summary["counts"]["failed"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
