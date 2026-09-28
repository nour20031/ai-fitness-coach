# Phase 13 Error Propagation Protocol

Phase 13 evaluates how upstream errors move through the frozen AI-orchestrated fitness-coaching system. It does not change models, thresholds, verified labels, prompts, confidence rules, or Streamlit behaviour.

## Scope

Two propagation chains are evaluated using representative evidence from existing phases:

1. Visual chain: degraded video evidence -> MediaPipe Pose / BlazePose -> movement analysis -> confidence gate -> structured result -> Gemma coaching.
2. Speech chain: user voice question -> Whisper small transcription -> Gemma answer.

## Evidence Sources

- Phase 6 Whisper small benchmark: `dataset/processed/evaluation/stt/whisper_small_results.csv`.
- Phase 8 orchestration traces and app behaviour.
- Phase 11 technical testing evidence.
- Phase 12 robustness/failure results: `dataset/processed/evaluation/robustness/robustness_video_results.csv`.

## Visual Method

Ten representative visual cases are selected from Phase 12. The set includes two clean baseline controls, two occlusion propagated-error cases, two occlusion blocked cases, two partial-body blocked cases, one crop case, and one fast-movement case.

For HIGH-confidence cases, the frozen Gemma 2 2B client is called using structured facts derived from the frozen Phase 12 CSV row. For LOW-confidence cases, normal coaching is not generated; the existing confidence message is recorded as the final user-facing output.

## Speech Method

Ten existing Whisper small recordings are selected from the Phase 6 benchmark. For each recording, Gemma is called twice using the same structured movement result from the clean `vid16_baseline` squat case:

1. reference/intended question;
2. actual Whisper transcription.

The comparison isolates the downstream effect of transcription error.

## Classifications

- `NO_PROPAGATION`: upstream result remains correct enough and final response remains appropriate.
- `STOPPED`: confidence or failure handling prevents normal downstream coaching.
- `PROPAGATED_TRANSPARENTLY`: an error reaches later stages but remains visible or traceable.
- `PROPAGATED_TO_INCORRECT_OUTPUT`: an upstream error reaches Gemma and produces user-facing feedback based on incorrect or changed facts.

## UI Transcription Visibility

The frozen Streamlit code transcribes a voice question inside `ask_voice_question()` and calls `ask_follow_up()` immediately. The transcription is displayed later by `show_question_result()`. Therefore, the current app makes the transcription visible after processing, but not before Gemma generates the answer, and the user cannot edit it before submission in the same interaction.
