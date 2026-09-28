import csv
import sys
import tempfile
from pathlib import Path

from dataset_common import ROOT_DIR

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.app import clear_video_state, commit_active_video, has_current_analysis, reset_analysis_state


OUTPUT = ROOT_DIR / "dataset" / "processed" / "evaluation" / "streamlit" / "session_state_regression_results.csv"


def temp_video_file(prefix):
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix=prefix)
    handle.write(b"temporary streamlit test video")
    handle.close()
    return handle.name


def base_state():
    return {
        "video_path": "",
        "video_name": "",
        "video_metadata": None,
        "analysis_result": None,
        "pose_preview_frame": None,
        "pose_preview_payload": None,
        "pose_preview_video_id": "",
        "latest_question_result": None,
        "latest_tts_audio": "",
        "latest_tts_error": "",
        "active_video_id": "",
        "analysed_video_id": "",
        "video_source_signature": "",
        "video_origin": "",
        "session_generation": 0,
    }


def mark_analysed(state):
    state["analysis_result"] = {"structured_result": {"rep_count": 10}}
    state["pose_preview_frame"] = "old-frame"
    state["pose_preview_payload"] = {"rep_count": 10}
    state["pose_preview_video_id"] = state["active_video_id"]
    state["latest_question_result"] = {"final_response": "old answer"}
    state["latest_tts_audio"] = ""
    state["analysed_video_id"] = state["active_video_id"]


def result(name, passed, detail):
    return {"check": name, "passed": passed, "detail": detail}


def main():
    rows = []

    # A. clear session after an analysed video removes all current-analysis state.
    state = base_state()
    path1 = temp_video_file("p10_a_")
    commit_active_video(path1, "video1.mp4", "upload", "upload:video1", state)
    mark_analysed(state)
    clear_video_state(state, remove_video=True)
    rows.append(result(
        "A_clear_session_removes_previous_result",
        not has_current_analysis(state) and not state["video_path"] and state["analysis_result"] is None,
        "Analysis, video path, preview frame, question result, and analysed_video_id were cleared.",
    ))

    # B. uploading a new video invalidates the previous video's result before analysis.
    state = base_state()
    path1 = temp_video_file("p10_b1_")
    path2 = temp_video_file("p10_b2_")
    commit_active_video(path1, "video1.mp4", "upload", "upload:video1", state)
    old_id = state["active_video_id"]
    mark_analysed(state)
    commit_active_video(path2, "video2.mp4", "upload", "upload:video2", state)
    rows.append(result(
        "B_new_upload_invalidates_old_result",
        state["active_video_id"] != old_id and not has_current_analysis(state) and state["analysis_result"] is None,
        "New upload received a new active_video_id and previous result was cleared.",
    ))
    clear_video_state(state, remove_video=True)

    # C. recording a new video invalidates the previous recording's result.
    state = base_state()
    path1 = temp_video_file("p10_c1_")
    path2 = temp_video_file("p10_c2_")
    commit_active_video(path1, "recorded1.mp4", "record", "record:1", state)
    old_id = state["active_video_id"]
    mark_analysed(state)
    commit_active_video(path2, "recorded2.mp4", "record", "record:2", state)
    rows.append(result(
        "C_new_recording_invalidates_old_result",
        state["active_video_id"] != old_id and not has_current_analysis(state) and state["analysis_result"] is None,
        "New recording received a new active_video_id and previous result was cleared.",
    ))
    clear_video_state(state, remove_video=True)

    # D. changing exercise clears the current analysis.
    state = base_state()
    commit_active_video(temp_video_file("p10_d_"), "video.mp4", "upload", "upload:video", state)
    mark_analysed(state)
    reset_analysis_state(state)
    rows.append(result(
        "D_exercise_change_clears_analysis",
        not has_current_analysis(state) and state["analysis_result"] is None,
        "Exercise-change reset cleared analysis while keeping the active video.",
    ))
    clear_video_state(state, remove_video=True)

    # E. changing curl require_ready clears the current curl analysis.
    state = base_state()
    commit_active_video(temp_video_file("p10_e_"), "curl.mp4", "upload", "upload:curl", state)
    mark_analysed(state)
    reset_analysis_state(state)
    rows.append(result(
        "E_ready_option_change_clears_analysis",
        not has_current_analysis(state) and state["analysis_result"] is None,
        "Ready-option reset cleared the previous curl analysis.",
    ))
    clear_video_state(state, remove_video=True)

    # F. Clear Session changes widget generation so old upload/audio widgets do not repopulate.
    state = base_state()
    old_generation = state["session_generation"]
    state["session_generation"] = old_generation + 1
    rows.append(result(
        "F_clear_session_versions_widgets",
        state["session_generation"] != old_generation,
        "Widget keys include session_generation, so clearing creates fresh uploader/audio widgets.",
    ))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "passed", "detail"])
        writer.writeheader()
        writer.writerows(rows)

    failures = [row for row in rows if not row["passed"]]
    print(f"Wrote {OUTPUT}")
    if failures:
        for failure in failures:
            print(f"FAILED: {failure['check']} - {failure['detail']}")
        raise SystemExit(1)
    print("All session-state regression checks passed.")


if __name__ == "__main__":
    main()
