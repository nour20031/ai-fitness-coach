# Streamlit Implementation

Phase 10 integrates the validated backend into the final user-facing Streamlit app. The Streamlit app is a client of the orchestration layer; it does not duplicate pose-estimation loops, movement thresholds, Whisper loading, Ollama HTTP calls, or Windows SAPI TTS logic.

## Entry Point

`app/app.py`

Run with:

```powershell
.\venv\Scripts\python.exe -m streamlit run app\app.py
```

## Final UI Flow

1. Select exercise: Squat or Bicep Curl.
2. Upload an exercise video or record a short local webcam clip.
3. Validate that the video exists and can be read.
4. Analyse the video through `FitnessCoachOrchestrator` while showing live annotated pose frames.
5. Display repetition count, form label, confidence status, and the final analysed pose visual.
6. Display grounded Gemma coaching if confidence is HIGH.
7. Display re-record guidance if confidence is LOW.
8. Allow typed follow-up questions without rerunning video analysis.
9. Allow recorded or uploaded voice questions.
10. Show the Whisper transcription before the AI answer.
11. Generate optional spoken feedback through Windows SAPI TTS.

Upload remains the most stable video input path. The app also restores a local webcam recording path using OpenCV `VideoCapture`; the recorded clip is saved as a temporary MP4 and then passed through the same `FitnessCoachOrchestrator` pipeline as an uploaded video. No separate webcam analysis algorithm is used. Streamlit 1.58.0 supports audio recording through `st.audio_input`, but it does not provide direct browser video recording as a reliable built-in widget.

## Compact Layout

The polished Phase 10 UI uses a compact desktop-first layout:

- Sidebar: exercise selection, short guidance, compact system status, optional technical system details, clear session.
- Main header: project title, `Pose analysis · Voice interaction · Grounded AI coaching`, subtle AI-domain chips, and concise safety note.
- Before analysis: two columns with video preview/live analysis on the left and a single upload/record/validation/action card on the right.
- After analysis: semantic metric cards followed by side-by-side Pose Analysis and AI Coaching cards.
- Rep breakdown: summary line plus compact scrollable table.
- Secondary content is placed in tabs: conversational Ask Coach, compact Analysis Details, and polished AI Pipeline cards.

## Final UI Visual Polish

The final visual polish uses a restrained academic theme:

- primary dark `#1F2937`
- teal accent `#0F766E`
- success green `#16A34A`
- warning amber `#D97706`
- page background `#F8FAFC`
- card background `#FFFFFF`
- border `#E5E7EB`
- primary text `#111827`
- secondary text `#64748B`

The default red Streamlit tab/button accent is overridden with the teal/navy theme. Result cards use subtle state borders rather than strong full-card fills. The HIGH confidence message is compact, while LOW confidence uses a light amber treatment. The AI Coaching section is presented as a card with a concise `Listen` button, while TTS behaviour remains unchanged.

## Session State

The app stores:

- uploaded temporary video path
- selected video name
- `active_video_id`
- `analysed_video_id`
- `session_generation`
- analysis result
- pose-preview frame
- latest question result
- latest TTS audio path
- latest TTS error
- bicep-curl ready-position option

This prevents the full video analysis from rerunning when the user asks a follow-up question or generates spoken feedback.

The app uses a central `reset_analysis_state()` helper whenever a new video is uploaded, a new recording is committed, the exercise changes, or the bicep ready-position mode changes. This clears only analysis-dependent state: result, pose preview, question answer, TTS output, and metadata. `Clear Session` additionally removes the active temporary video and increments `session_generation`, which is included in uploader/audio widget keys so Streamlit creates fresh widgets instead of restoring the previous upload.

Each video input receives a new `active_video_id`. When analysis completes, `analysed_video_id` is set to the current active ID. Results, rep breakdown, final pose preview, AI coaching, and question tabs are displayed only when `analysed_video_id == active_video_id`. This prevents stale results from appearing after Clear Session, a new upload, a new recording, an exercise change, or a bicep ready-mode change.

## Orchestration Call

The main analysis button calls:

`FitnessCoachOrchestrator.run(...)`

Follow-up questions call:

`FitnessCoachOrchestrator.ask_follow_up(...)`

TTS playback calls:

`FitnessCoachOrchestrator.speak_text(...)`

The app also uses:

`FitnessCoachOrchestrator.check_llm_ready(...)`

to show whether Ollama and `gemma2:2b` are available before coaching feedback is requested.

## Live Pose Visual

The app displays actual pose-analysis frames from the selected video while processing. This is implemented through an optional backend callback:

`FitnessCoachOrchestrator.analyze_video(..., frame_callback=..., preview_interval=8)`

The callback is invoked from the existing MediaPipe analysis loop. It draws landmarks/skeleton connections on a clean copy of the original raw frame using the already calculated MediaPipe landmarks. The analyser still receives its own frame copy for validated processing, while Streamlit receives a clean display-only copy.

- current frame index
- current repetition count
- knee angle for squat when available
- elbow angle for bicep curl when available

Display frames are fitted into a bounded dashboard canvas using `fit_frame_for_display(...)`, preserving aspect ratio and adding letterboxing/padding when needed. This prevents portrait videos from becoming extremely tall without reducing the MediaPipe inference frame. The live-analysis display is limited to about 760 x 420 px, while the final pose-analysis preview is limited to about 560 x 350 px. The letterbox background uses neutral charcoal rather than a saturated colour.

The overlay is drawn once in Streamlit as a compact top-left panel. Old analyser/debug text is not reused for the display callback, which prevents duplicate labels such as `Frame`, `Reps`, `Knee`, `Elbow`, `Arm`, and `Leg` from overlapping. The preview updates every 8 processed frames. All frames are still analysed by the backend; only the Streamlit display is throttled. This keeps the validated movement-analysis logic unchanged. After analysis, the app keeps the latest clean annotated frame visible as the pose-analysis result.

## Webcam Recording Orientation

The Record Video path includes a bicep/squat recording option:

`Correct mirrored webcam`

It defaults on because the local webcam preview was horizontally mirrored. When enabled, OpenCV applies `cv2.flip(frame, 1)` to webcam frames before both preview display and MP4 writing. This correction is applied only to newly recorded webcam clips. Uploaded videos, research dataset videos, and benchmark evidence files are not flipped or modified.

## Bicep Ready-Position Option

When Bicep Curl is selected, the app shows the instruction:

`Bicep Curl recording tip: start with the arm extended, keep the elbow visible, then begin curling.`

It also exposes a bicep-only checkbox:

`Start counting after extended-arm ready position`

This option passes the existing `require_ready` behaviour into `FitnessCoachOrchestrator.run(...)` / `analyze_video(...)` and then into `BicepCurlAnalyzer`. It does not create new thresholds or a second rep-counting algorithm. The default remains off because the regression check showed that enabling it changed the known `vid12` half-range curl result from 10 reps to 9 reps. Changing this option invalidates the stored analysis result so the user must analyse again with the new interpretation mode.

## Rep Breakdown

The rep-by-rep table is shown directly after the Pose Analysis and AI Coaching area, before the Ask Coach / Analysis Details / AI Pipeline tabs. It uses human-readable issue labels such as `Body Swinging`, `Half Range`, `Shallow`, and `No Issue`, while preserving the underlying stored issue codes. A summary line is calculated only from existing `rep_history`, for example `10 repetitions analysed · 10 with no detected issues`. The table height is capped at about 250 px so longer recordings scroll inside the table instead of stretching the full page.

## Validation Cards

Video duration, FPS, and resolution are shown using compact HTML cards rather than Streamlit metrics. This keeps values such as `576 x 1024` fully visible in the right-hand validation column.

## Confidence Presentation

HIGH confidence shows the movement result and permits normal Gemma coaching.

LOW confidence is visually highlighted, normal coaching is blocked, and the user sees re-record guidance. Technical reason codes are translated into simple user-facing explanations, with detailed metrics available in an expander.

## Questions

Typed questions reuse the stored structured movement result. The video is not processed again.

Voice questions use:

`audio -> Whisper small -> visible transcription -> Gemma + structured facts -> grounded answer`

The app displays the transcription so that STT errors are visible to the user.

The Ask Coach tab uses a compact two-column layout: text question on the left and voice question on the right. Responses are presented conversationally as `You` / `Transcription` / `AI Coach` cards while keeping Whisper transcription visible.

## AI Pipeline Presentation

The AI Pipeline tab presents four compact cards:

- Visual AI: MediaPipe Pose / BlazePose
- Audio AI: Whisper small
- Language AI: Gemma 2 2B
- Voice Output: Windows SAPI

It also shows simple text flows: `Video -> MediaPipe -> Movement Analysis -> Confidence Gate -> Gemma -> Feedback -> TTS` and `Voice -> Whisper -> Gemma`. Windows SAPI is presented as an output component, not as one of the three required pretrained models.

## TTS Playback

TTS is optional. The app provides a compact `Listen` button after coaching responses. If TTS succeeds, Streamlit displays the WAV using `st.audio`. If TTS fails, the text response remains visible and the error is shown only in an optional technical expander.

## Temporary Files

Uploaded video/audio files are written to temporary paths because the backend expects file paths. Ordinary user uploads are not copied into the research dataset.

## Error Handling

The app handles:

- missing or unreadable video
- unsupported exercise state
- LOW confidence
- missing or failed audio transcription
- empty or failed Whisper transcription
- Ollama or `gemma2:2b` unavailable
- Gemma response failure
- TTS failure

Technical details are kept behind expanders where possible.

## Phase 10 Smoke Evidence

Evidence files:

- `dataset/processed/evaluation/streamlit/streamlit_smoke_test_results.csv`
- `dataset/processed/evaluation/streamlit/streamlit_smoke_test_examples.md`
- `dataset/processed/evaluation/streamlit/session_state_regression_results.csv`

Smoke-test highlights:

- Correct squat: `vid16`, 10 reps, correct, HIGH confidence
- Shallow squat: `vid24`, 10 reps, shallow, HIGH confidence
- Correct bicep curl: `vid06`, 10 reps, correct, HIGH confidence
- Half-range curl: `vid12`, 10 reps, half_range, HIGH confidence
- LOW confidence: `vid23`, coaching blocked
- Text follow-up: answer generated from stored structured result
- Voice follow-up: Whisper transcription shown as "How many reps did I complete?"
- TTS playback: spoken answer generated successfully
- TTS failure: text answer remained available
- Session-state checks: Clear Session, new upload, new recording, exercise change, and ready-option change all cleared stale analysis state.

The same smoke tests were rerun after the UI polish. The reference analysis results were unchanged:

- `vid16`: 10 reps, correct, HIGH
- `vid24`: 10 reps, shallow, HIGH
- `vid06`: 10 reps, correct, HIGH
- `vid12`: 10 reps, half_range, HIGH
- `vid23`: 0 reps, unknown, LOW, coaching blocked
