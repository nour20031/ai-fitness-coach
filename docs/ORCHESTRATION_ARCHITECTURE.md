# Phase 8 Orchestration Architecture

Phase 8 integrates the three selected pretrained models into one traceable coaching pipeline.

## Selected Pretrained Models

| Role | Selected model | Data space |
| --- | --- | --- |
| Model 1 | MediaPipe Pose / BlazePose | Visual/video pose estimation |
| Model 2 | Whisper small | Audio speech-to-text |
| Model 3 | `gemma2:2b` via Ollama | Text/language feedback |

Ollama is the runtime for Gemma. The deterministic movement-analysis rules and confidence gate are not pretrained models.

## Final Architecture

```text
                   EXERCISE VIDEO
                          |
               MediaPipe Pose / BlazePose
                 PRETRAINED MODEL 1
                          |
               landmarks + visibility
                          |
                  Movement Analysis
                          |
             reps + issues + measurements
                          |
                Confidence Validation
                          |
                 structured result
                          |
                     gemma2:2b
                 PRETRAINED MODEL 3
                          |
                grounded text feedback
                          ^
                          |
                transcribed question
                          ^
                   Whisper small
                 PRETRAINED MODEL 2
                          ^
                 USER VOICE QUESTION
```

Visual domain feeds MediaPipe. Audio domain feeds Whisper. Text and structured facts feed Gemma.

## Structured Result Schema

The orchestration layer passes a single structured movement result into Gemma:

```json
{
  "exercise": "squat",
  "rep_count": 8,
  "form_label": "shallow",
  "detected_issues": ["not_deep_enough"],
  "measured_features": {
    "minimum_knee_angle": 128.0,
    "per_rep_min_knee_angles": [130.2, 128.0]
  },
  "confidence_status": "HIGH",
  "confidence_metrics": {
    "usable_frame_rate": 0.99,
    "required_joint_availability": 0.95,
    "angle_calculability_rate": 0.95,
    "dropout_rate": 0.0,
    "movement_signal_range": 60.0,
    "processed_frames": 600,
    "pose_detected_frames": 598
  },
  "confidence_reasons": [],
  "confidence_message": "Analysis confidence is high. Repetition count and form feedback can be shown.",
  "rep_history": [],
  "thresholds": {}
}
```

Only measurements already produced by the movement-analysis layer are included. No hidden or invented measurements are added for the language model.

## Orchestration Service

Implementation file:

```text
src/orchestration/fitness_coach_orchestrator.py
```

The orchestrator:

- analyses the exercise video with MediaPipe Pose / BlazePose
- runs the selected squat or bicep-curl movement analyser
- applies the confidence gate
- converts the result into the structured schema
- optionally transcribes a user voice question using Whisper small
- builds a grounded Gemma prompt using structured facts plus the user question
- calls `gemma2:2b` through Ollama
- returns a final response, stage latencies, failure status, and trace information

## Grounding Rule

Gemma never receives raw video. It receives only:

- exercise name
- detected repetition count
- form label
- detected issues
- measured features
- confidence result
- user question or Whisper transcription

Gemma must not claim it watched the video, invent additional issues, diagnose injury, infer pain causes, or override upstream movement results.

## Low-Confidence Behaviour

If the confidence gate returns `LOW`, confident coaching is blocked. The orchestrator returns the confidence-layer re-record guidance instead of asking Gemma to produce form coaching.

Example:

```text
Analysis confidence is low because no valid repetitions were detected. Please record again with the full movement visible.
```

This stops weak visual evidence from becoming overconfident coaching feedback.

## Failure Handling

The orchestrator returns a clear failure stage if:

- the video path is missing or unreadable
- no useful movement can be analysed
- Whisper receives missing or empty audio
- Whisper returns empty text
- Ollama is unavailable
- the requested Gemma model is unavailable
- Gemma returns an empty response

Failed components are not replaced with fabricated outputs.

## Error Propagation Trace

Every orchestration result preserves enough information to trace:

```text
video
-> pose and landmarks
-> movement result
-> confidence decision
-> Whisper transcription when used
-> Gemma prompt/response
```

This allows later evaluation to explain failures such as:

- pose error -> incorrect movement measurement -> incorrect structured fact -> incorrect explanation
- low confidence -> coaching blocked -> error propagation stopped
- STT error -> incorrect user question -> potentially irrelevant Gemma response
