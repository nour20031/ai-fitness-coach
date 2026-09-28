import tempfile
import unittest
from pathlib import Path

from app.app import (
    clear_video_state,
    commit_active_video,
    has_current_analysis,
    reset_analysis_state,
)
from src.orchestration.fitness_coach_orchestrator import StructuredMovementResult


def _base_state():
    return {
        "video_path": "",
        "video_name": "",
        "video_metadata": None,
        "analysis_result": None,
        "pose_preview_frame": None,
        "pose_preview_payload": None,
        "pose_preview_video_id": "",
        "latest_question_result": None,
        "latest_question_text": "",
        "latest_tts_audio": "",
        "latest_tts_error": "",
        "active_video_id": "",
        "analysed_video_id": "",
        "video_source_signature": "",
        "video_origin": "",
        "session_generation": 0,
    }


def _temp_video(prefix):
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix=prefix)
    handle.write(b"temporary phase 11 session-state video")
    handle.close()
    return handle.name


def _mark_analysed(state):
    state["analysis_result"] = {"structured_result": {"rep_count": 10}}
    state["pose_preview_frame"] = "old-frame"
    state["pose_preview_payload"] = {"rep_count": 10}
    state["pose_preview_video_id"] = state["active_video_id"]
    state["latest_question_result"] = {"final_response": "old answer"}
    state["latest_question_text"] = "old question"
    state["latest_tts_audio"] = ""
    state["analysed_video_id"] = state["active_video_id"]


class Phase11ContractTests(unittest.TestCase):
    def test_structured_movement_result_contract_fields(self):
        result = StructuredMovementResult(
            exercise="squat",
            rep_count=10,
            form_label="correct",
            detected_issues=[],
            measured_features={"minimum_knee_angle": 75.0},
            confidence_status="HIGH",
            confidence_metrics={"usable_frame_rate": 0.99},
            confidence_reasons=[],
            confidence_message="Usable movement evidence was available.",
            rep_history=[{"rep": 1, "issues": []}],
            thresholds={"deep_squat_threshold": 115},
        )

        self.assertEqual(result.exercise, "squat")
        self.assertEqual(result.rep_count, 10)
        self.assertIsInstance(result.detected_issues, list)
        self.assertIsInstance(result.measured_features, dict)
        self.assertIsInstance(result.confidence_metrics, dict)
        self.assertIsInstance(result.rep_history, list)
        self.assertIsInstance(result.thresholds, dict)

    def test_new_video_invalidates_previous_analysis(self):
        state = _base_state()
        first = _temp_video("p11_first_")
        second = _temp_video("p11_second_")
        try:
            commit_active_video(first, "first.mp4", "upload", "upload:first", state)
            first_id = state["active_video_id"]
            _mark_analysed(state)
            self.assertTrue(has_current_analysis(state))

            commit_active_video(second, "second.mp4", "upload", "upload:second", state)
            self.assertNotEqual(first_id, state["active_video_id"])
            self.assertFalse(has_current_analysis(state))
            self.assertIsNone(state["analysis_result"])
        finally:
            clear_video_state(state, remove_video=True)
            Path(first).unlink(missing_ok=True)
            Path(second).unlink(missing_ok=True)

    def test_reset_analysis_state_keeps_current_video_but_clears_result(self):
        state = _base_state()
        video = _temp_video("p11_reset_")
        try:
            commit_active_video(video, "curl.mp4", "upload", "upload:curl", state)
            _mark_analysed(state)
            reset_analysis_state(state)

            self.assertTrue(state["video_path"])
            self.assertFalse(has_current_analysis(state))
            self.assertIsNone(state["analysis_result"])
            self.assertEqual(state["latest_question_text"], "")
        finally:
            clear_video_state(state, remove_video=True)
            Path(video).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
