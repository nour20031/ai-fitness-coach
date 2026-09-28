import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, request

import cv2
import whisper

from src.audio import TextToSpeechResult, WindowsSapiTextToSpeech
from src.pose.bicep_curl_analyzer import BicepCurlAnalyzer
from src.pose.confidence_validator import ConfidenceValidator
from src.pose.pose_detector import PoseDetector
from src.pose.squat_analyzer import SquatAnalyzer


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SYSTEM_PROMPT = ROOT_DIR / "dataset" / "llm" / "system_prompt.txt"
WHISPER_MODEL_DIR = ROOT_DIR / "models" / "whisper"

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


@dataclass
class StructuredMovementResult:
    exercise: str
    rep_count: int
    form_label: str
    detected_issues: list[str]
    measured_features: dict[str, Any]
    confidence_status: str
    confidence_metrics: dict[str, Any]
    confidence_reasons: list[str]
    confidence_message: str
    rep_history: list[dict[str, Any]] = field(default_factory=list)
    thresholds: dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscriptionResult:
    status: str
    transcription: str = ""
    latency_seconds: float = 0.0
    error: str = ""
    model: str = "whisper_small"


@dataclass
class LLMResult:
    status: str
    response: str = ""
    latency_seconds: float = 0.0
    error: str = ""
    model: str = "gemma2:2b"
    prompt: str = ""


def _create_analyzer(exercise: str, require_ready: bool = False):
    if exercise == "squat":
        return SquatAnalyzer(verbose=False)
    if exercise == "bicep_curl":
        return BicepCurlAnalyzer(require_ready=require_ready, verbose=False)
    raise ValueError(f"Unsupported exercise: {exercise}")


def _extract_issue_counts(summary: dict[str, Any]) -> dict[str, int]:
    issue_counts = summary.get("issue_counts") or {}
    if issue_counts:
        return issue_counts

    for rep in summary.get("rep_history", []):
        for issue in rep.get("issues", []):
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    return issue_counts


def _structured_features(exercise: str, summary: dict[str, Any]) -> dict[str, Any]:
    rep_history = summary.get("rep_history", [])
    if exercise == "squat":
        knee_angles = [
            rep["min_knee_angle"]
            for rep in rep_history
            if "min_knee_angle" in rep
        ]
        return {
            "minimum_knee_angle": min(knee_angles) if knee_angles else None,
            "per_rep_min_knee_angles": knee_angles,
        }

    if exercise == "bicep_curl":
        min_elbow = [
            rep["min_elbow_angle"]
            for rep in rep_history
            if "min_elbow_angle" in rep
        ]
        max_elbow = [
            rep["max_elbow_angle"]
            for rep in rep_history
            if "max_elbow_angle" in rep
        ]
        shoulder_swing = [
            rep["max_shoulder_deviation"]
            for rep in rep_history
            if "max_shoulder_deviation" in rep
        ]
        return {
            "minimum_elbow_angle": min(min_elbow) if min_elbow else None,
            "maximum_elbow_angle": max(max_elbow) if max_elbow else None,
            "maximum_shoulder_deviation": max(shoulder_swing) if shoulder_swing else None,
            "per_rep_min_elbow_angles": min_elbow,
            "per_rep_max_elbow_angles": max_elbow,
        }

    return {}


class WhisperSmallTranscriber:
    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            WHISPER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            self._model = whisper.load_model("small", download_root=str(WHISPER_MODEL_DIR))
        return self._model

    def transcribe(self, audio_path: str | Path | None) -> TranscriptionResult:
        if not audio_path:
            return TranscriptionResult(status="not_requested")

        path = Path(audio_path)
        if not path.exists() or path.stat().st_size == 0:
            return TranscriptionResult(
                status="failed",
                error=f"Audio file is missing or empty: {path}",
            )

        start = time.perf_counter()
        try:
            result = self._load_model().transcribe(
                str(path),
                language="en",
                fp16=False,
                verbose=False,
            )
        except Exception as exc:
            return TranscriptionResult(
                status="failed",
                latency_seconds=round(time.perf_counter() - start, 4),
                error=f"Whisper transcription failed: {exc}",
            )

        transcription = (result.get("text") or "").strip()
        latency = round(time.perf_counter() - start, 4)
        if not transcription:
            return TranscriptionResult(
                status="failed",
                latency_seconds=latency,
                error="Whisper returned an empty transcription.",
            )
        return TranscriptionResult(
            status="success",
            transcription=transcription,
            latency_seconds=latency,
        )


class GemmaCoachClient:
    def __init__(
        self,
        model_tag: str = "gemma2:2b",
        system_prompt_path: str | Path = DEFAULT_SYSTEM_PROMPT,
        ollama_url: str = "http://127.0.0.1:11434/api/generate",
    ):
        self.model_tag = model_tag
        self.ollama_url = ollama_url
        self.system_prompt = Path(system_prompt_path).read_text(encoding="utf-8")

    def build_prompt(
        self,
        structured_result: StructuredMovementResult,
        user_question: str,
    ) -> str:
        facts = asdict(structured_result)
        return (
            f"{self.system_prompt}\n\n"
            "Structured movement-analysis facts:\n"
            f"{json.dumps(facts, indent=2, sort_keys=True)}\n\n"
            f"User question: {user_question}\n\n"
            "Answer:"
        )

    def generate(
        self,
        structured_result: StructuredMovementResult,
        user_question: str,
    ) -> LLMResult:
        prompt = self.build_prompt(structured_result, user_question)
        payload = {
            "model": self.model_tag,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 220,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.ollama_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start = time.perf_counter()
        try:
            with request.urlopen(req, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            return LLMResult(
                status="failed",
                latency_seconds=round(time.perf_counter() - start, 4),
                error=f"Ollama/Gemma request failed: {exc}",
                model=self.model_tag,
                prompt=prompt,
            )

        text = (result.get("response") or "").strip()
        latency = round(time.perf_counter() - start, 4)
        if not text:
            return LLMResult(
                status="failed",
                latency_seconds=latency,
                error="Gemma returned an empty response.",
                model=self.model_tag,
                prompt=prompt,
            )
        return LLMResult(
            status="success",
            response=text,
            latency_seconds=latency,
            model=self.model_tag,
            prompt=prompt,
        )


class FitnessCoachOrchestrator:
    def __init__(
        self,
        llm_model_tag: str = "gemma2:2b",
        transcriber: WhisperSmallTranscriber | None = None,
        llm_client: GemmaCoachClient | None = None,
        tts_service: WindowsSapiTextToSpeech | None = None,
    ):
        self.transcriber = transcriber or WhisperSmallTranscriber()
        self.llm_client = llm_client or GemmaCoachClient(model_tag=llm_model_tag)
        self.tts_service = tts_service or WindowsSapiTextToSpeech()

    def check_llm_ready(self) -> dict[str, Any]:
        req = request.Request(
            "http://127.0.0.1:11434/api/tags",
            method="GET",
        )
        try:
            with request.urlopen(req, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            return {
                "ready": False,
                "error": f"Ollama service is unavailable: {exc}",
                "model": self.llm_client.model_tag,
            }

        models = payload.get("models", [])
        names = {model.get("name", "") for model in models}
        if self.llm_client.model_tag not in names:
            return {
                "ready": False,
                "error": f"Required model is not installed: {self.llm_client.model_tag}",
                "model": self.llm_client.model_tag,
                "available_models": sorted(names),
            }
        return {
            "ready": True,
            "model": self.llm_client.model_tag,
            "available_models": sorted(names),
        }

    def analyze_video(
        self,
        video_path: str | Path,
        exercise: str,
        frame_callback=None,
        preview_interval: int = 8,
        require_ready: bool = False,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        path = Path(video_path)
        if not path.exists():
            return {
                "status": "failed",
                "error": f"Video file does not exist: {path}",
                "processing_time_seconds": 0.0,
            }

        detector = PoseDetector()
        analyzer = _create_analyzer(exercise, require_ready=require_ready)
        confidence_validator = ConfidenceValidator(exercise)
        capture = cv2.VideoCapture(str(path))

        if not capture.isOpened():
            return {
                "status": "failed",
                "error": f"Could not open video: {path}",
                "processing_time_seconds": round(time.perf_counter() - start, 4),
            }

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        processed_frames = 0
        pose_frames = 0
        last_clean_frame = None
        last_landmarks = {}
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            processed_frames += 1
            clean_frame = frame.copy()
            detector.find_pose(frame, draw=False)
            landmarks = detector.get_landmarks(frame)
            last_clean_frame = clean_frame
            last_landmarks = landmarks
            confidence_validator.update(detector, landmarks)
            if landmarks:
                pose_frames += 1
                analyzer.analyze(detector, landmarks, frame.copy())

            if frame_callback and preview_interval > 0 and processed_frames % preview_interval == 0:
                preview_payload = self._build_frame_preview_payload(
                    detector=detector,
                    frame=clean_frame,
                    landmarks=landmarks,
                    analyzer=analyzer,
                    exercise=exercise,
                    frame_index=processed_frames,
                    total_frames=total_frames,
                )
                frame_callback(preview_payload)

        capture.release()
        if hasattr(analyzer, "finalize"):
            analyzer.finalize()

        if frame_callback and last_clean_frame is not None:
            final_preview_payload = self._build_frame_preview_payload(
                detector=detector,
                frame=last_clean_frame,
                landmarks=last_landmarks,
                analyzer=analyzer,
                exercise=exercise,
                frame_index=processed_frames,
                total_frames=total_frames,
            )
            final_preview_payload["is_final_preview"] = True
            frame_callback(final_preview_payload)

        summary = analyzer.get_session_summary()
        confidence = confidence_validator.result(summary.get("total_reps", 0))
        issue_counts = _extract_issue_counts(summary)
        detected_issues = sorted(issue_counts.keys())
        structured = StructuredMovementResult(
            exercise=exercise,
            rep_count=int(summary.get("total_reps", 0)),
            form_label=summary.get("predicted_form_label", "unknown"),
            detected_issues=detected_issues,
            measured_features=_structured_features(exercise, summary),
            confidence_status=confidence["confidence_status"],
            confidence_metrics={
                "usable_frame_rate": confidence.get("usable_frame_rate"),
                "required_joint_availability": confidence.get("required_joint_availability"),
                "angle_calculability_rate": confidence.get("angle_calculability_rate"),
                "dropout_rate": confidence.get("dropout_rate"),
                "movement_signal_range": confidence.get("movement_signal_range"),
                "processed_frames": processed_frames,
                "pose_detected_frames": pose_frames,
            },
            confidence_reasons=confidence.get("reason_codes", []),
            confidence_message=confidence.get("user_message", ""),
            rep_history=summary.get("rep_history", []),
            thresholds=summary.get("thresholds", {}),
        )

        return {
            "status": "success",
            "summary": summary,
            "confidence": confidence,
            "structured_result": structured,
            "processing_time_seconds": round(time.perf_counter() - start, 4),
        }

    def _build_frame_preview_payload(
        self,
        detector,
        frame,
        landmarks,
        analyzer,
        exercise,
        frame_index,
        total_frames,
    ) -> dict[str, Any]:
        annotated = frame.copy()
        if getattr(detector, "results", None) and detector.results.pose_landmarks:
            detector.mp_draw.draw_landmarks(
                annotated,
                detector.results.pose_landmarks,
                detector.mp_pose.POSE_CONNECTIONS,
            )

        summary = analyzer.get_session_summary()
        angle_label, angle_value = self._preview_angle(detector, landmarks, exercise)
        return {
            "frame": annotated,
            "frame_index": frame_index,
            "total_frames": total_frames,
            "rep_count": int(summary.get("total_reps", 0)),
            "angle_label": angle_label,
            "angle_value": angle_value,
        }

    @staticmethod
    def _best_visible_angle(detector, landmarks, left_points, right_points):
        left_visibility = sum(landmarks.get(point, {}).get("visibility", 0) for point in left_points)
        right_visibility = sum(landmarks.get(point, {}).get("visibility", 0) for point in right_points)
        points = left_points if left_visibility >= right_visibility else right_points
        return detector.calculate_angle(landmarks, *points)

    def _preview_angle(self, detector, landmarks, exercise):
        if not landmarks:
            return "", None
        if exercise == "squat":
            return "Knee angle", self._best_visible_angle(
                detector,
                landmarks,
                (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
                (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
            )
        if exercise == "bicep_curl":
            return "Elbow angle", self._best_visible_angle(
                detector,
                landmarks,
                (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
                (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
            )
        return "", None

    def run(
        self,
        video_path: str | Path,
        exercise: str,
        user_question: str | None = None,
        audio_path: str | Path | None = None,
        enable_tts: bool = False,
        tts_output_name: str | None = None,
        frame_callback=None,
        preview_interval: int = 8,
        require_ready: bool = False,
    ) -> dict[str, Any]:
        total_start = time.perf_counter()
        trace: dict[str, Any] = {
            "video_path": str(video_path),
            "exercise": exercise,
            "stage_outputs": {},
        }

        movement = self.analyze_video(
            video_path,
            exercise,
            frame_callback=frame_callback,
            preview_interval=preview_interval,
            require_ready=require_ready,
        )
        trace["stage_outputs"]["video_pose_movement_confidence"] = movement
        if movement["status"] != "success":
            return self._failure_result("video_movement", movement.get("error", ""), trace, total_start)

        structured: StructuredMovementResult = movement["structured_result"]

        transcription = TranscriptionResult(status="not_requested")
        final_question = (user_question or "").strip()
        if audio_path:
            transcription = self.transcriber.transcribe(audio_path)
            trace["stage_outputs"]["whisper_small"] = asdict(transcription)
            if transcription.status != "success":
                return self._failure_result(
                    "whisper_small",
                    transcription.error,
                    trace,
                    total_start,
                    movement,
                    transcription,
                )
            final_question = transcription.transcription

        if not final_question:
            final_question = "Give me concise coaching feedback based on this result."

        if structured.confidence_status == "LOW":
            low_response = structured.confidence_message
            if final_question:
                low_response = (
                    f"{structured.confidence_message} "
                    "The system should not give confident form coaching from this attempt."
                )
            tts = self._speak_response(low_response, enable_tts, tts_output_name)
            trace["stage_outputs"]["text_to_speech"] = asdict(tts)
            trace["stage_outputs"]["gemma2b"] = {
                "status": "blocked",
                "reason": "LOW_CONFIDENCE",
                "llm_called": False,
            }
            return {
                "status": "blocked",
                "success": True,
                "failure_stage": "",
                "coaching_blocked": True,
                "structured_result": asdict(structured),
                "transcription": asdict(transcription),
                "gemma": asdict(LLMResult(status="blocked", response=low_response)),
                "tts": asdict(tts),
                "final_response": low_response,
                "latency": self._latency(movement, transcription, None, total_start, tts),
                "trace": trace,
            }

        llm = self.llm_client.generate(structured, final_question)
        trace["stage_outputs"]["gemma2b"] = asdict(llm)
        if llm.status != "success":
            return self._failure_result("gemma2b", llm.error, trace, total_start, movement, transcription, llm)

        tts = self._speak_response(llm.response, enable_tts, tts_output_name)
        trace["stage_outputs"]["text_to_speech"] = asdict(tts)
        return {
            "status": "success",
            "success": True,
            "failure_stage": "",
            "coaching_blocked": False,
            "structured_result": asdict(structured),
            "transcription": asdict(transcription),
            "gemma": asdict(llm),
            "tts": asdict(tts),
            "final_response": llm.response,
            "latency": self._latency(movement, transcription, llm, total_start, tts),
            "trace": trace,
        }

    def ask_follow_up(
        self,
        structured_result: StructuredMovementResult | dict[str, Any],
        user_question: str | None = None,
        audio_path: str | Path | None = None,
        enable_tts: bool = False,
        tts_output_name: str | None = None,
    ) -> dict[str, Any]:
        total_start = time.perf_counter()
        structured = self._coerce_structured_result(structured_result)
        trace: dict[str, Any] = {
            "exercise": structured.exercise,
            "stage_outputs": {
                "structured_result_reused": asdict(structured),
            },
        }

        transcription = TranscriptionResult(status="not_requested")
        final_question = (user_question or "").strip()
        if audio_path:
            transcription = self.transcriber.transcribe(audio_path)
            trace["stage_outputs"]["whisper_small"] = asdict(transcription)
            if transcription.status != "success":
                return self._failure_result(
                    "whisper_small",
                    transcription.error,
                    trace,
                    total_start,
                    movement={"structured_result": structured, "processing_time_seconds": 0.0},
                    transcription=transcription,
                )
            final_question = transcription.transcription

        if not final_question:
            return self._failure_result(
                "question",
                "No question was provided.",
                trace,
                total_start,
                movement={"structured_result": structured, "processing_time_seconds": 0.0},
                transcription=transcription,
            )

        if structured.confidence_status == "LOW":
            low_response = (
                f"{structured.confidence_message} "
                "The system should not give confident form coaching from this attempt."
            )
            tts = self._speak_response(low_response, enable_tts, tts_output_name)
            trace["stage_outputs"]["gemma2b"] = {
                "status": "blocked",
                "reason": "LOW_CONFIDENCE",
                "llm_called": False,
            }
            trace["stage_outputs"]["text_to_speech"] = asdict(tts)
            return {
                "status": "blocked",
                "success": True,
                "failure_stage": "",
                "coaching_blocked": True,
                "structured_result": asdict(structured),
                "transcription": asdict(transcription),
                "gemma": asdict(LLMResult(status="blocked", response=low_response)),
                "tts": asdict(tts),
                "final_response": low_response,
                "latency": self._latency(
                    {"processing_time_seconds": 0.0},
                    transcription,
                    None,
                    total_start,
                    tts,
                ),
                "trace": trace,
            }

        llm = self.llm_client.generate(structured, final_question)
        trace["stage_outputs"]["gemma2b"] = asdict(llm)
        if llm.status != "success":
            return self._failure_result(
                "gemma2b",
                llm.error,
                trace,
                total_start,
                movement={"structured_result": structured, "processing_time_seconds": 0.0},
                transcription=transcription,
                llm=llm,
            )

        tts = self._speak_response(llm.response, enable_tts, tts_output_name)
        trace["stage_outputs"]["text_to_speech"] = asdict(tts)
        return {
            "status": "success",
            "success": True,
            "failure_stage": "",
            "coaching_blocked": False,
            "structured_result": asdict(structured),
            "transcription": asdict(transcription),
            "gemma": asdict(llm),
            "tts": asdict(tts),
            "final_response": llm.response,
            "latency": self._latency(
                {"processing_time_seconds": 0.0},
                transcription,
                llm,
                total_start,
                tts,
            ),
            "trace": trace,
        }

    def speak_text(
        self,
        text: str,
        output_name: str | None = None,
    ) -> TextToSpeechResult:
        return self._speak_response(text, True, output_name)

    def create_pose_preview_frame(self, video_path: str | Path, max_frames: int = 90):
        path = Path(video_path)
        if not path.exists():
            return None

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return None

        detector = PoseDetector()
        frame_index = 0
        preview_frame = None
        while frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            annotated = detector.find_pose(frame, draw=True)
            landmarks = detector.get_landmarks(annotated)
            if landmarks:
                preview_frame = annotated
                break
            preview_frame = annotated

        capture.release()
        return preview_frame

    @staticmethod
    def _coerce_structured_result(
        structured_result: StructuredMovementResult | dict[str, Any],
    ) -> StructuredMovementResult:
        if isinstance(structured_result, StructuredMovementResult):
            return structured_result
        return StructuredMovementResult(**structured_result)

    def _speak_response(
        self,
        final_response: str,
        enable_tts: bool,
        output_name: str | None = None,
    ) -> TextToSpeechResult:
        if not enable_tts:
            return TextToSpeechResult(status="not_requested")
        return self.tts_service.synthesize(final_response, output_name=output_name)

    def _latency(self, movement, transcription, llm, total_start, tts: TextToSpeechResult | None = None):
        return {
            "movement_processing_seconds": movement.get("processing_time_seconds", 0.0) if movement else 0.0,
            "whisper_latency_seconds": transcription.latency_seconds if transcription else 0.0,
            "gemma_latency_seconds": llm.latency_seconds if llm else 0.0,
            "tts_latency_seconds": tts.latency_seconds if tts else 0.0,
            "total_latency_seconds": round(time.perf_counter() - total_start, 4),
        }

    def _failure_result(
        self,
        stage: str,
        error_message: str,
        trace: dict[str, Any],
        total_start: float,
        movement: dict[str, Any] | None = None,
        transcription: TranscriptionResult | None = None,
        llm: LLMResult | None = None,
    ) -> dict[str, Any]:
        return {
            "status": "failed",
            "success": False,
            "failure_stage": stage,
            "error": error_message,
            "coaching_blocked": True,
            "structured_result": asdict(movement["structured_result"]) if movement and movement.get("structured_result") else {},
            "transcription": asdict(transcription) if transcription else {},
            "gemma": asdict(llm) if llm else {},
            "tts": asdict(TextToSpeechResult(status="not_requested")),
            "final_response": error_message,
            "latency": self._latency(movement or {}, transcription, llm, total_start),
            "trace": trace,
        }
