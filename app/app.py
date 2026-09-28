import os
import sys
import tempfile
import uuid
import inspect
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.orchestration import FitnessCoachOrchestrator


EXERCISES = {
    "Squat": {
        "key": "squat",
        "help": "Keep hips, knees, and ankles visible.",
    },
    "Bicep Curl": {
        "key": "bicep_curl",
        "help": "Keep shoulders, elbows, and wrists visible.",
    },
}

SUPPORTED_VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "m4v"]
SUPPORTED_AUDIO_TYPES = ["wav", "mp3", "m4a", "ogg"]
PREVIEW_INTERVAL = 8
ORCHESTRATOR_CACHE_VERSION = "phase10_final_ui_require_ready_v2"


@st.cache_resource
def get_orchestrator(cache_version=ORCHESTRATOR_CACHE_VERSION):
    return FitnessCoachOrchestrator()


def call_with_supported_kwargs(func, *args, **kwargs):
    """Call cached services safely after Streamlit hot reloads older objects."""
    signature = inspect.signature(func)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return func(*args, **supported)


def init_session_state():
    defaults = {
        "video_path": "",
        "video_name": "",
        "video_metadata": None,
        "analysis_result": None,
        "pose_preview_frame": None,
        "pose_preview_payload": None,
        "latest_question_result": None,
        "latest_tts_audio": "",
        "latest_tts_error": "",
        "curl_require_ready": False,
        "last_exercise_key": "",
        "last_curl_require_ready": False,
        "session_generation": 0,
        "active_video_id": "",
        "analysed_video_id": "",
        "pose_preview_video_id": "",
        "video_source_signature": "",
        "video_origin": "",
        "correct_mirrored_webcam": True,
        "latest_question_text": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def inject_css():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.05rem;
            padding-bottom: 1.5rem;
            background: #f8fafc;
        }
        .stApp {
            background: #f8fafc;
        }
        section[data-testid="stSidebar"] {
            background: #f1f5f9;
            border-right: 1px solid #e5e7eb;
        }
        .small-note {
            color: #64748b;
            font-size: 0.88rem;
            margin-top: 0.1rem;
        }
        .status-card {
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 0.75rem;
            background: #ffffff;
        }
        .soft-card {
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 0.85rem;
            background: #ffffff;
        }
        .tight-title {
            margin-bottom: 0.15rem;
            color: #111827;
        }
        .chip-row {
            display: flex;
            gap: 0.45rem;
            flex-wrap: wrap;
            margin: 0.6rem 0 0.25rem 0;
        }
        .chip {
            border: 1px solid #ccfbf1;
            background: #f0fdfa;
            color: #0f766e;
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .input-card, .coach-card, .detail-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1rem;
        }
        .coach-card {
            min-height: 350px;
        }
        .card-title {
            color: #111827;
            font-weight: 700;
            font-size: 1.02rem;
            margin-bottom: 0.55rem;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.5rem 0 0.75rem 0;
        }
        .metric-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-left: 4px solid #0f766e;
            border-radius: 10px;
            padding: 0.8rem 0.9rem;
        }
        .metric-card.success { border-left-color: #16a34a; }
        .metric-card.warning { border-left-color: #d97706; }
        .metric-card.neutral { border-left-color: #1f2937; }
        .metric-value {
            color: #111827;
            font-size: 1.85rem;
            line-height: 1.15;
            font-weight: 600;
        }
        .metric-label {
            color: #64748b;
            font-size: 0.83rem;
            margin-top: 0.2rem;
        }
        .confidence-card {
            border-radius: 10px;
            padding: 0.75rem 0.9rem;
            margin: 0.55rem 0 0.7rem 0;
            border: 1px solid #bbf7d0;
            background: #f0fdf4;
            color: #166534;
        }
        .confidence-card.low {
            border-color: #fed7aa;
            background: #fffbeb;
            color: #92400e;
        }
        .mini-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.6rem;
            margin: 0.55rem 0;
        }
        .pipeline-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.7rem;
            margin-bottom: 0.9rem;
        }
        .flow-row {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 0.75rem;
            color: #334155;
            margin: 0.5rem 0;
        }
        .qa-card {
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            background: #ffffff;
            padding: 0.75rem 0.85rem;
            margin-top: 0.75rem;
        }
        .qa-role {
            color: #64748b;
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }
        .rep-summary {
            color: #475569;
            font-size: 0.92rem;
            margin: -0.2rem 0 0.5rem 0;
        }
        .stButton > button[kind="primary"] {
            background: #1f2937;
            border-color: #1f2937;
            color: white;
            border-radius: 8px;
            font-weight: 650;
        }
        .stButton > button[kind="primary"]:hover {
            background: #0f766e;
            border-color: #0f766e;
            color: white;
        }
        div[data-baseweb="tab-list"] button[aria-selected="true"] {
            color: #0f766e;
        }
        div[data-baseweb="tab-highlight"] {
            background-color: #0f766e;
        }
        div[data-testid="stAlert"] {
            border-radius: 10px;
        }
        @media (max-width: 1100px) {
            .metric-grid, .pipeline-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def make_temp_path(suffix):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.close()
    return temp_file.name


def remove_temp_file(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _state(state=None):
    return st.session_state if state is None else state


def _set_state_value(state, key, value):
    state[key] = value


def reset_analysis_state(state=None):
    state = _state(state)
    for key, value in {
        "video_metadata": None,
        "analysis_result": None,
        "pose_preview_frame": None,
        "pose_preview_payload": None,
        "pose_preview_video_id": "",
        "latest_question_result": None,
        "latest_question_text": "",
        "latest_tts_error": "",
        "analysed_video_id": "",
    }.items():
        _set_state_value(state, key, value)

    latest_tts = state.get("latest_tts_audio", "")
    if latest_tts and Path(latest_tts).name.startswith("streamlit_"):
        remove_temp_file(latest_tts)
    _set_state_value(state, "latest_tts_audio", "")


def clear_video_state(state=None, remove_video=True):
    state = _state(state)
    if remove_video:
        remove_temp_file(state.get("video_path", ""))
    reset_analysis_state(state)
    for key, value in {
        "video_path": "",
        "video_name": "",
        "video_source_signature": "",
        "video_origin": "",
        "active_video_id": "",
    }.items():
        _set_state_value(state, key, value)


def commit_active_video(path, name, origin, source_signature="", state=None):
    state = _state(state)
    old_path = state.get("video_path", "")
    if old_path and old_path != path:
        remove_temp_file(old_path)
    reset_analysis_state(state)
    _set_state_value(state, "video_path", str(path))
    _set_state_value(state, "video_name", name)
    _set_state_value(state, "video_origin", origin)
    _set_state_value(state, "video_source_signature", source_signature or f"{origin}:{name}:{uuid.uuid4().hex}")
    _set_state_value(state, "active_video_id", uuid.uuid4().hex)
    _set_state_value(state, "analysed_video_id", "")


def has_current_analysis(state=None):
    state = _state(state)
    return bool(
        state.get("analysis_result")
        and state.get("active_video_id")
        and state.get("analysed_video_id") == state.get("active_video_id")
    )


def reset_full_session():
    generation = int(st.session_state.get("session_generation", 0)) + 1
    clear_video_state(st.session_state, remove_video=True)
    keep = {
        "session_generation": generation,
        "curl_require_ready": False,
        "last_curl_require_ready": False,
        "last_exercise_key": "",
        "correct_mirrored_webcam": True,
    }
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()
    for key, value in keep.items():
        st.session_state[key] = value


def save_uploaded_file(uploaded_file, fallback_suffix):
    suffix = Path(uploaded_file.name).suffix or fallback_suffix
    temp_path = make_temp_path(suffix)
    with open(temp_path, "wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return temp_path


def validate_video_file(path):
    if not path or not Path(path).exists():
        return False, "Please upload a video before analysing."

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return False, "Your video could not be opened. Please try another recording."

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    capture.release()

    if frame_count <= 0 or width <= 0 or height <= 0:
        return False, "Your video could not be read. Please try another recording."

    return True, {
        "frame_count": frame_count,
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "duration_seconds": round(frame_count / fps, 2) if fps else 0.0,
    }


def bgr_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def fit_frame_for_display(frame, max_width=760, max_height=420, fill=(39, 24, 17)):
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = cv2.copyMakeBorder(
        resized,
        (max_height - new_height) // 2,
        max_height - new_height - (max_height - new_height) // 2,
        (max_width - new_width) // 2,
        max_width - new_width - (max_width - new_width) // 2,
        cv2.BORDER_CONSTANT,
        value=fill,
    )
    return canvas


def draw_overlay_panel(frame, payload=None):
    if not payload:
        return frame

    lines = []
    total = payload.get("total_frames") or 0
    frame_index = payload.get("frame_index") or 0
    if total:
        lines.append(f"Frame {frame_index} / {total}")
    elif frame_index:
        lines.append(f"Frame {frame_index}")
    lines.append(f"Reps {payload.get('rep_count', 0)}")
    angle = payload.get("angle_value")
    label = (payload.get("angle_label") or "Angle").replace(" angle", "")
    if angle is not None:
        lines.append(f"{label} {angle:.1f} deg")

    overlay = frame.copy()
    panel_x, panel_y = 16, 16
    line_height = 24
    panel_width = 210
    panel_height = 22 + line_height * len(lines)
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (20, 24, 32),
        -1,
    )
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    y = panel_y + 28
    for index, line in enumerate(lines):
        color = (255, 255, 255) if index != 1 else (130, 255, 170)
        cv2.putText(
            frame,
            line,
            (panel_x + 14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
        y += line_height
    return frame


def prepare_display_frame(frame, payload=None, max_width=760, max_height=420):
    display = fit_frame_for_display(frame, max_width=max_width, max_height=max_height)
    return draw_overlay_panel(display, payload)


def first_video_frame(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def record_webcam_clip(
    output_path,
    seconds=20,
    camera_index=0,
    preview_placeholder=None,
    progress_bar=None,
    correct_mirror=True,
):
    capture = cv2.VideoCapture(camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    if not capture.isOpened():
        raise RuntimeError("Could not open the local webcam.")

    fps = 20
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    import time
    start_time = time.time()
    frame_count = 0
    while time.time() - start_time < seconds:
        ok, frame = capture.read()
        if not ok:
            break
        if correct_mirror:
            frame = cv2.flip(frame, 1)
        writer.write(frame)
        frame_count += 1
        if preview_placeholder and frame_count % 4 == 0:
            display = prepare_display_frame(frame, max_width=760, max_height=420)
            preview_placeholder.image(
                bgr_to_rgb(display),
                channels="RGB",
                use_container_width=False,
                caption="Recording preview",
            )
        if progress_bar:
            progress_bar.progress(min((time.time() - start_time) / seconds, 1.0))

    capture.release()
    writer.release()
    if progress_bar:
        progress_bar.progress(1.0)
    if frame_count == 0:
        raise RuntimeError("No frames were recorded from the webcam.")
    return frame_count


def human_label(value):
    labels = {
        "bicep_curl": "Bicep Curl",
        "half_range": "Half Range",
        "correct": "Correct",
        "shallow": "Shallow",
        "unknown": "Unknown",
        "body_swinging": "Body Swinging",
        "half_range_of_motion": "Half Range",
        "not_deep_enough": "Shallow",
        "forward_lean": "Forward Lean",
    }
    return labels.get(str(value), str(value).replace("_", " ").title())


def reason_label(code):
    labels = {
        "NO_REPS_DETECTED": "No valid repetitions were detected.",
        "INSUFFICIENT_MOVEMENT_SIGNAL": "The movement signal was too small to analyse confidently.",
        "LOW_POSE_DETECTION_RATE": "The body pose was not detected in enough frames.",
        "LOW_REQUIRED_JOINT_AVAILABILITY": "Required body landmarks were not visible enough.",
        "LOW_ANGLE_CALCULABILITY": "The required joint angles could not be calculated reliably.",
        "HIGH_TRACKING_DROPOUT": "Pose tracking dropped out too often.",
    }
    return labels.get(code, str(code).replace("_", " ").title())


def render_video_or_visual(placeholder):
    frame = st.session_state.pose_preview_frame
    if (
        frame is not None
        and st.session_state.pose_preview_video_id
        and st.session_state.pose_preview_video_id == st.session_state.active_video_id
    ):
        display = prepare_display_frame(
            frame,
            st.session_state.pose_preview_payload,
            max_width=560,
            max_height=350,
        )
        placeholder.image(
            bgr_to_rgb(display),
            channels="RGB",
            use_container_width=False,
            caption="Pose analysis visual from the uploaded video",
        )
    elif st.session_state.video_path:
        preview = first_video_frame(st.session_state.video_path)
        if preview is not None:
            display = prepare_display_frame(preview, max_width=760, max_height=420)
            placeholder.image(
                bgr_to_rgb(display),
                channels="RGB",
                use_container_width=False,
                caption="Uploaded video preview",
            )
        else:
            placeholder.video(st.session_state.video_path)
    else:
        placeholder.info("Upload a video to preview it here.")


def build_rep_table(rep_history):
    rows = []
    for rep in rep_history or []:
        issues = rep.get("issues", [])
        row = {
            "Rep": rep.get("rep", ""),
            "Issue": ", ".join(human_label(issue) for issue in issues) if issues else "No Issue",
        }
        if "min_knee_angle" in rep:
            row["Min Knee Angle"] = format_angle(rep.get("min_knee_angle"))
        if "min_elbow_angle" in rep:
            row["Min Elbow Angle"] = format_angle(rep.get("min_elbow_angle"))
            row["Max Elbow Angle"] = format_angle(rep.get("max_elbow_angle"))
        if "max_shoulder_deviation" in rep:
            row["Shoulder Movement"] = rep.get("max_shoulder_deviation")
        rows.append(row)
    return pd.DataFrame(rows)


def format_angle(value):
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.1f}°"
    except (TypeError, ValueError):
        return str(value)


def rep_breakdown_summary(rep_history):
    total = len(rep_history or [])
    clear = sum(1 for rep in rep_history or [] if not rep.get("issues"))
    attention = total - clear
    if total == 0:
        return ""
    if attention == 0:
        return f"{total} repetitions analysed · {clear} with no detected issues"
    return f"{total} repetitions analysed · {clear} clear · {attention} need attention"


def result_from_analysis_only(movement_result, llm_error):
    if movement_result.get("status") != "success":
        return {
            "status": "failed",
            "success": False,
            "failure_stage": "video_movement",
            "error": movement_result.get("error", ""),
            "coaching_blocked": True,
            "structured_result": {},
            "transcription": {},
            "gemma": {},
            "tts": {},
            "final_response": movement_result.get("error", ""),
            "latency": {
                "movement_processing_seconds": movement_result.get("processing_time_seconds", 0.0),
                "whisper_latency_seconds": 0.0,
                "gemma_latency_seconds": 0.0,
                "tts_latency_seconds": 0.0,
                "total_latency_seconds": movement_result.get("processing_time_seconds", 0.0),
            },
        }

    structured = movement_result["structured_result"]
    return {
        "status": "movement_only",
        "success": True,
        "failure_stage": "",
        "error": llm_error,
        "coaching_blocked": structured.confidence_status == "LOW",
        "structured_result": structured.__dict__,
        "transcription": {"status": "not_requested"},
        "gemma": {"status": "not_requested"},
        "tts": {"status": "not_requested"},
        "final_response": "",
        "latency": {
            "movement_processing_seconds": movement_result.get("processing_time_seconds", 0.0),
            "whisper_latency_seconds": 0.0,
            "gemma_latency_seconds": 0.0,
            "tts_latency_seconds": 0.0,
            "total_latency_seconds": movement_result.get("processing_time_seconds", 0.0),
        },
    }


def run_analysis(video_path, exercise_key, visual_placeholder, progress_bar, progress_text, require_ready=False):
    orchestrator = get_orchestrator()
    ready = orchestrator.check_llm_ready()
    st.session_state.pose_preview_frame = None
    st.session_state.pose_preview_payload = None
    st.session_state.pose_preview_video_id = ""
    analysis_video_id = st.session_state.active_video_id

    def on_frame(payload):
        if analysis_video_id != st.session_state.active_video_id:
            return
        frame = payload["frame"]
        st.session_state.pose_preview_frame = frame
        st.session_state.pose_preview_payload = payload
        st.session_state.pose_preview_video_id = analysis_video_id
        display = prepare_display_frame(frame, payload, max_width=760, max_height=420)
        visual_placeholder.image(
            bgr_to_rgb(display),
            channels="RGB",
            use_container_width=False,
            caption="Live pose analysis",
        )
        total = payload.get("total_frames") or 0
        frame_index = payload.get("frame_index") or 0
        if total:
            progress_bar.progress(min(frame_index / total, 1.0))
            angle = payload.get("angle_value")
            angle_text = f" | {payload.get('angle_label')}: {angle:.1f} deg" if angle is not None else ""
            progress_text.caption(
                f"Frame {frame_index} / {total} | Reps: {payload.get('rep_count', 0)}{angle_text}"
            )

    with st.spinner("Analysing exercise..."):
        if ready["ready"]:
            result = call_with_supported_kwargs(
                orchestrator.run,
                video_path=video_path,
                exercise=exercise_key,
                user_question="Give concise coaching feedback based on this result.",
                frame_callback=on_frame,
                preview_interval=PREVIEW_INTERVAL,
                require_ready=require_ready,
            )
        else:
            movement = call_with_supported_kwargs(
                orchestrator.analyze_video,
                video_path,
                exercise_key,
                frame_callback=on_frame,
                preview_interval=PREVIEW_INTERVAL,
                require_ready=require_ready,
            )
            result = result_from_analysis_only(movement, ready.get("error", ""))

    progress_bar.progress(1.0)
    progress_text.caption("Analysis complete.")
    if analysis_video_id == st.session_state.active_video_id:
        st.session_state.analysis_result = result
        st.session_state.analysed_video_id = analysis_video_id
        st.session_state.latest_question_result = None
        st.session_state.latest_question_text = ""
        st.session_state.latest_tts_audio = ""
        st.session_state.latest_tts_error = ""


def show_status_sidebar(ready):
    st.sidebar.markdown("### AI Fitness Coach")
    st.sidebar.caption("Exercise-form feedback with visual, voice, and language AI.")
    st.sidebar.divider()
    exercise_name = st.sidebar.selectbox("Exercise", list(EXERCISES.keys()))
    st.sidebar.caption(EXERCISES[exercise_name]["help"])
    st.sidebar.divider()
    st.sidebar.markdown("**System**")
    st.sidebar.caption("✓ Pose ready")
    st.sidebar.caption("✓ Speech ready")
    st.sidebar.caption("✓ AI Coach ready" if ready["ready"] else "⚠ AI Coach unavailable")
    st.sidebar.caption("✓ Voice output ready")
    with st.sidebar.expander("System details"):
        st.write("**Pose:** MediaPipe Pose / BlazePose")
        st.write("**Speech-to-Text:** Whisper small")
        st.write("**Language:** Gemma 2 2B via Ollama")
        st.write("**Speech Output:** Windows SAPI")
        if not ready["ready"]:
            st.warning(ready.get("error", ""))
    st.sidebar.divider()
    if st.sidebar.button("Clear Session"):
        reset_full_session()
        st.rerun()
    return exercise_name, EXERCISES[exercise_name]["key"]


def show_validation_card(valid, metadata_or_error, exercise_name, exercise_key, analyse_disabled):
    st.markdown("#### Input / Validation")
    st.write(f"**Exercise:** {exercise_name}")
    if exercise_key == "bicep_curl":
        st.info("Bicep Curl recording tip: start with the arm extended, keep the elbow visible, then begin curling.")
        st.session_state.curl_require_ready = st.checkbox(
            "Start counting after extended-arm ready position",
            value=st.session_state.curl_require_ready,
            help="Recommended when the recording begins with the elbow clearly extended. Leave off for dataset-style clips that may start mid-rep.",
        )
    if valid:
        metadata = metadata_or_error
        st.markdown(
            "<div class='confidence-card'><strong>✓ Video ready</strong></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="mini-grid">
              <div class="soft-card"><div class="small-note">Duration</div><strong>{metadata['duration_seconds']} s</strong></div>
              <div class="soft-card"><div class="small-note">FPS</div><strong>{metadata['fps']}</strong></div>
              <div class="soft-card"><div class="small-note">Resolution</div><strong>{metadata['width']} × {metadata['height']}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning(metadata_or_error)
        st.caption("Upload is the reliable fallback. Record uses the local webcam on this computer.")
    return st.button("Analyse Exercise", type="primary", disabled=analyse_disabled or not valid, use_container_width=True)


def show_metrics(result):
    structured = result.get("structured_result") or {}
    latency = result.get("latency") or {}
    form_label = human_label(structured.get("form_label", "unknown"))
    confidence = structured.get("confidence_status", "")
    form_class = "success" if form_label == "Correct" else "warning" if form_label in {"Shallow", "Half Range"} else "neutral"
    confidence_class = "success" if confidence == "HIGH" else "warning" if confidence == "LOW" else "neutral"
    st.markdown(
        f"""
        <div class="metric-grid">
          <div class="metric-card neutral"><div class="metric-value">{structured.get('rep_count', 0)}</div><div class="metric-label">Repetitions</div></div>
          <div class="metric-card {form_class}"><div class="metric-value">{form_label}</div><div class="metric-label">Form</div></div>
          <div class="metric-card {confidence_class}"><div class="metric-value">{confidence}</div><div class="metric-label">Confidence</div></div>
          <div class="metric-card neutral"><div class="metric-value">{latency.get('movement_processing_seconds', 0.0):.1f}s</div><div class="metric-label">Analysis Time</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_compact(result):
    structured = result.get("structured_result") or {}
    status = structured.get("confidence_status")
    if status == "HIGH":
        st.markdown(
            """
            <div class="confidence-card">
              <strong>✓ Analysis confidence: HIGH</strong>
              <div class="small-note">Enough usable visual and movement information was available for analysis. This does not guarantee perfect repetition counting.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif status == "LOW":
        message = structured.get("confidence_message") or "Please record again with the full movement clearly visible."
        if (
            structured.get("exercise") == "bicep_curl"
            and st.session_state.curl_require_ready
            and int(structured.get("rep_count", 0) or 0) == 0
        ):
            message = (
                "No repetition cycle was detected after using the extended-arm starting-position option. "
                "Try recording again and begin with your arm clearly extended."
            )
        st.markdown(
            f"""
            <div class="confidence-card low">
              <strong>⚠ Analysis confidence: LOW</strong>
              <div style="margin-top:0.35rem;">{message}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        reasons = structured.get("confidence_reasons") or []
        if reasons:
            st.caption(" ".join(reason_label(reason) for reason in reasons))


def speak_button(text, key):
    if not text:
        return
    if st.button("🔊 Listen", key=f"speak_{key}", use_container_width=False):
        result = get_orchestrator().speak_text(
            text,
            output_name=f"streamlit_{key}_{uuid.uuid4().hex}.wav",
        )
        if result.status == "success":
            st.session_state.latest_tts_audio = result.audio_path
            st.session_state.latest_tts_error = ""
        else:
            st.session_state.latest_tts_audio = ""
            st.session_state.latest_tts_error = result.error or "Spoken feedback could not be generated."

    if st.session_state.latest_tts_audio and Path(st.session_state.latest_tts_audio).exists():
        st.audio(st.session_state.latest_tts_audio)
    elif st.session_state.latest_tts_error:
        st.warning("Spoken feedback could not be generated. The text response is still available.")
        with st.expander("TTS technical details"):
            st.write(st.session_state.latest_tts_error)


def show_result_dashboard():
    if not has_current_analysis():
        return
    result = st.session_state.analysis_result
    if result.get("status") == "failed" and not result.get("structured_result"):
        st.error("Your video could not be analysed. Please try another recording.")
        with st.expander("Technical details"):
            st.write(result.get("error", ""))
        return

    st.markdown("#### Analysis Result")
    show_metrics(result)
    confidence_compact(result)

    pose_col, coach_col = st.columns([0.52, 0.48], vertical_alignment="top")
    with pose_col:
        with st.container(border=True):
            st.markdown("<div class='card-title'>Pose Analysis</div>", unsafe_allow_html=True)
            render_video_or_visual(st.empty())

    with coach_col:
        with st.container(border=True):
            st.markdown("<div class='card-title'>✦ AI Coaching</div>", unsafe_allow_html=True)
            structured = result.get("structured_result") or {}
            confidence = structured.get("confidence_status")
            gemma = result.get("gemma") or {}
            response = ""
            if confidence == "LOW":
                response = result.get("final_response") or structured.get("confidence_message", "")
            elif gemma.get("status") == "success":
                response = result.get("final_response", "")
            elif result.get("error"):
                st.warning("Local coaching model unavailable. Please ensure Ollama is running and gemma2:2b is installed.")
                with st.expander("Technical details"):
                    st.write(result.get("error", ""))
            st.write(response or "Coaching feedback is not available.")
            speak_button(response, "initial")


def show_rep_breakdown():
    if not has_current_analysis():
        return
    result = st.session_state.analysis_result or {}
    structured = result.get("structured_result") or {}
    rep_table = build_rep_table(structured.get("rep_history") or [])
    if rep_table.empty:
        return
    st.markdown("#### Rep Breakdown")
    st.markdown(
        f"<div class='rep-summary'>{rep_breakdown_summary(structured.get('rep_history') or [])}</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        rep_table,
        use_container_width=True,
        hide_index=True,
        height=250,
    )


def ask_text_question(question):
    if not has_current_analysis():
        st.warning("Analyse the current video before asking a follow-up question.")
        return
    structured = (st.session_state.analysis_result or {}).get("structured_result") or {}
    ready = get_orchestrator().check_llm_ready()
    if not ready["ready"] and structured.get("confidence_status") != "LOW":
        st.error("Local coaching model unavailable. Please ensure Ollama is running and gemma2:2b is installed.")
        return
    with st.spinner("Asking coach..."):
        st.session_state.latest_question_result = get_orchestrator().ask_follow_up(
            structured,
            user_question=question,
        )
        st.session_state.latest_question_text = question
        st.session_state.latest_tts_audio = ""
        st.session_state.latest_tts_error = ""


def ask_voice_question(audio_file):
    if not has_current_analysis():
        st.warning("Analyse the current video before asking a voice question.")
        return
    if not audio_file:
        st.warning("Record or upload an audio question first.")
        return
    structured = (st.session_state.analysis_result or {}).get("structured_result") or {}
    ready = get_orchestrator().check_llm_ready()
    if not ready["ready"] and structured.get("confidence_status") != "LOW":
        st.error("Local coaching model unavailable. Please ensure Ollama is running and gemma2:2b is installed.")
        return
    audio_path = save_uploaded_file(audio_file, ".wav")
    try:
        with st.spinner("Transcribing voice question..."):
            st.session_state.latest_question_result = get_orchestrator().ask_follow_up(
                structured,
                audio_path=audio_path,
            )
            transcription = st.session_state.latest_question_result.get("transcription") or {}
            st.session_state.latest_question_text = transcription.get("transcription", "")
            st.session_state.latest_tts_audio = ""
            st.session_state.latest_tts_error = ""
    finally:
        remove_temp_file(audio_path)


def show_question_result():
    if not has_current_analysis():
        return
    result = st.session_state.latest_question_result
    if not result:
        return
    transcription = result.get("transcription") or {}
    if result.get("failure_stage") == "whisper_small":
        st.error("Speech could not be transcribed. Please record again or type your question.")
        with st.expander("Technical details"):
            st.write(result.get("error", ""))
        return
    if result.get("failure_stage") == "gemma2b":
        st.error("The AI coaching model is currently unavailable.")
        with st.expander("Technical details"):
            st.write(result.get("error", ""))
        return
    if transcription.get("status") == "success":
        st.markdown(
            f"""
            <div class="qa-card">
              <div class="qa-role">Transcription</div>
              <div>“{transcription.get('transcription', '')}”</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif st.session_state.latest_question_text:
        st.markdown(
            f"""
            <div class="qa-card">
              <div class="qa-role">You</div>
              <div>“{st.session_state.latest_question_text}”</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    response = result.get("final_response", "")
    st.markdown(
        f"""
        <div class="qa-card">
          <div class="qa-role">AI Coach</div>
          <div>{response}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    speak_button(response, "question")


def show_ask_tab():
    text_col, voice_col = st.columns(2, vertical_alignment="top")
    with text_col:
        st.markdown("##### Text Question")
        q_col, ask_col = st.columns([0.82, 0.18], vertical_alignment="bottom")
        with q_col:
            question = st.text_input("Ask a follow-up", placeholder="How were my squats?", label_visibility="collapsed")
        with ask_col:
            if st.button("Ask", disabled=not question.strip(), key="ask_text"):
                ask_text_question(question.strip())

    with voice_col:
        st.markdown("##### Voice Question")
        generation = st.session_state.session_generation
        audio_recording = st.audio_input("🎙 Record question", key=f"record_voice_{generation}") if hasattr(st, "audio_input") else None
        audio_upload = st.file_uploader("📁 Upload audio", type=SUPPORTED_AUDIO_TYPES, key=f"voice_upload_{generation}")
        selected_audio = audio_recording or audio_upload
        if selected_audio:
            st.audio(selected_audio)
        if st.button("Ask voice", disabled=selected_audio is None, key="ask_voice"):
            ask_voice_question(selected_audio)

    show_question_result()


def show_details_tab():
    result = st.session_state.analysis_result or {}
    structured = result.get("structured_result") or {}
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Detected Issues")
        issues = structured.get("detected_issues") or []
        st.write(", ".join(human_label(issue) for issue in issues) if issues else "None")
        with st.expander("Measured Features", expanded=True):
            st.json(structured.get("measured_features") or {})
    with col2:
        with st.expander("Confidence Metrics", expanded=True):
            st.json(structured.get("confidence_metrics") or {})
        st.markdown("##### Confidence Reasons")
        reasons = structured.get("confidence_reasons") or []
        st.write([reason_label(reason) for reason in reasons] if reasons else "None")

    with st.expander("Thresholds"):
        st.json(structured.get("thresholds") or {})


def show_pipeline_tab():
    result = st.session_state.analysis_result or {}
    latency = result.get("latency") or {}
    st.markdown(
        """
        <div class="pipeline-grid">
          <div class="soft-card"><strong>Visual AI</strong><div class="small-note">MediaPipe Pose / BlazePose</div></div>
          <div class="soft-card"><strong>Audio AI</strong><div class="small-note">Whisper small</div></div>
          <div class="soft-card"><strong>Language AI</strong><div class="small-note">Gemma 2 2B</div></div>
          <div class="soft-card"><strong>Voice Output</strong><div class="small-note">Windows SAPI</div></div>
        </div>
        <div class="flow-row"><strong>Video</strong> → MediaPipe → Movement Analysis → Confidence Gate → Gemma → Feedback → TTS</div>
        <div class="flow-row"><strong>Voice</strong> → Whisper → Gemma</div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Component latencies", expanded=True):
        st.json(latency)


def show_tabs():
    if not has_current_analysis():
        return
    ask_tab, details_tab, pipeline_tab = st.tabs(["Ask Coach", "Analysis Details", "AI Pipeline"])
    with ask_tab:
        show_ask_tab()
    with details_tab:
        show_details_tab()
    with pipeline_tab:
        show_pipeline_tab()


def main():
    st.set_page_config(
        page_title="AI-Orchestrated Fitness Coach",
        page_icon="AI",
        layout="wide",
    )
    init_session_state()
    inject_css()

    ready = get_orchestrator().check_llm_ready()
    exercise_name, exercise_key = show_status_sidebar(ready)
    if st.session_state.last_exercise_key and st.session_state.last_exercise_key != exercise_key:
        reset_analysis_state()
    st.session_state.last_exercise_key = exercise_key

    st.markdown("<h2 class='tight-title'>AI-Orchestrated Fitness Coach</h2>", unsafe_allow_html=True)
    st.markdown("<div class='small-note'>Pose analysis · Voice interaction · Grounded AI coaching</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="chip-row">
          <span class="chip">Visual AI</span>
          <span class="chip">Voice AI</span>
          <span class="chip">Language AI</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("This prototype provides exercise-form feedback and is not a medical diagnostic tool.")

    top_left, top_right = st.columns([0.6, 0.4], vertical_alignment="top")
    with top_left:
        st.markdown("#### Video / Live Analysis")
        visual_placeholder = st.empty()

    with top_right:
        with st.container(border=True):
            upload_tab, record_tab = st.tabs(["Upload Video", "Record Video"])
            with upload_tab:
                uploaded_video = st.file_uploader(
                    "Upload exercise video",
                    type=SUPPORTED_VIDEO_TYPES,
                    label_visibility="collapsed",
                    key=f"video_upload_{st.session_state.session_generation}",
                )
                if uploaded_video:
                    upload_size = getattr(uploaded_video, "size", None)
                    if upload_size is None:
                        upload_size = len(uploaded_video.getbuffer())
                    upload_signature = f"upload:{uploaded_video.name}:{upload_size}"
                    if upload_signature != st.session_state.video_source_signature:
                        new_path = save_uploaded_file(uploaded_video, ".mp4")
                        commit_active_video(
                            new_path,
                            uploaded_video.name,
                            "upload",
                            source_signature=upload_signature,
                        )
                        st.rerun()

            with record_tab:
                rec_col1, rec_col2 = st.columns(2)
                seconds = rec_col1.slider("Seconds", min_value=5, max_value=60, value=20, step=5)
                camera_index = rec_col2.number_input("Camera", min_value=0, max_value=5, value=0)
                st.session_state.correct_mirrored_webcam = st.checkbox(
                    "Correct mirrored webcam",
                    value=st.session_state.correct_mirrored_webcam,
                    help="Applies only to new webcam recordings. Uploaded videos are not flipped.",
                )
                if st.button("Record Video", use_container_width=True):
                    recorded_path = make_temp_path(".mp4")
                    rec_progress = st.progress(0)
                    try:
                        with st.spinner("Recording from local webcam..."):
                            record_webcam_clip(
                                recorded_path,
                                seconds=int(seconds),
                                camera_index=int(camera_index),
                                preview_placeholder=visual_placeholder,
                                progress_bar=rec_progress,
                                correct_mirror=bool(st.session_state.correct_mirrored_webcam),
                            )
                        recording_name = f"recorded_{uuid.uuid4().hex[:8]}.mp4"
                        commit_active_video(
                            recorded_path,
                            recording_name,
                            "record",
                            source_signature=f"record:{recording_name}",
                        )
                        st.success("Recording saved. You can analyse it now.")
                        st.rerun()
                    except RuntimeError as error:
                        remove_temp_file(recorded_path)
                        st.error(str(error))

            valid = False
            metadata_or_error = "Upload a video to start."
            if st.session_state.video_path:
                valid, metadata_or_error = validate_video_file(st.session_state.video_path)
                st.session_state.video_metadata = metadata_or_error if valid else None

            analyse_clicked = show_validation_card(
                valid=valid,
                metadata_or_error=metadata_or_error,
                exercise_name=exercise_name,
                exercise_key=exercise_key,
                analyse_disabled=not st.session_state.video_path,
            )
            current_require_ready = bool(st.session_state.curl_require_ready) if exercise_key == "bicep_curl" else False
            if (
                exercise_key == "bicep_curl"
                and st.session_state.last_curl_require_ready != current_require_ready
                and has_current_analysis()
            ):
                reset_analysis_state()
                st.info("Ready-position option changed. Please analyse the video again.")
            st.session_state.last_curl_require_ready = current_require_ready
            progress_bar = st.progress(0)
            progress_text = st.empty()

    render_video_or_visual(visual_placeholder)

    if analyse_clicked:
        require_ready = bool(st.session_state.curl_require_ready) if exercise_key == "bicep_curl" else False
        run_analysis(
            st.session_state.video_path,
            exercise_key,
            visual_placeholder,
            progress_bar,
            progress_text,
            require_ready=require_ready,
        )

    if has_current_analysis():
        show_result_dashboard()
        show_rep_breakdown()
        show_tabs()


if __name__ == "__main__":
    main()
