# Phase 10 Streamlit Integration Smoke Tests

These are UI/backend integration smoke checks only. They are not the formal Phase 11 technical evaluation.

LLM readiness: `{"ready": true, "model": "gemma2:2b", "available_models": ["gemma2:2b", "llama3.2:3b", "mistral:7b-instruct"]}`

## P10-A - Correct squat uploaded-video path

- Status: `success`
- Confidence: `HIGH`
- Reps: `10`
- Form: `correct`
- Coaching blocked: `False`
- Whisper transcription: ``
- Gemma status: `success`
- TTS status: `not_requested`
- Pose preview available: `True`

Final response:

Your squat form looks good! You're mostly maintaining a good knee angle throughout the movement.  Keep working on maintaining that angle consistently.

## P10-B - Shallow squat uploaded-video path

- Status: `success`
- Confidence: `HIGH`
- Reps: `10`
- Form: `shallow`
- Coaching blocked: `False`
- Whisper transcription: ``
- Gemma status: `success`
- TTS status: `not_requested`
- Pose preview available: `True`

Final response:

Your squats seem shallow. You're not going deep enough.  Focus on getting your knees to a good depth.

## P10-C - Correct bicep-curl uploaded-video path

- Status: `success`
- Confidence: `HIGH`
- Reps: `10`
- Form: `correct`
- Coaching blocked: `False`
- Whisper transcription: ``
- Gemma status: `success`
- TTS status: `not_requested`
- Pose preview available: `True`

Final response:

Your bicep curls look good! You're hitting the correct form consistently, with good elbow angles and shoulder deviation.  Keep practicing and you'll continue to improve.

## P10-D - Half-range bicep-curl uploaded-video path

- Status: `success`
- Confidence: `HIGH`
- Reps: `10`
- Form: `half_range`
- Coaching blocked: `False`
- Whisper transcription: ``
- Gemma status: `success`
- TTS status: `not_requested`
- Pose preview available: `True`

Final response:

You're showing good form for the bicep curl, but you're hitting the half-range of motion limit in several reps.  Focus on maintaining a full range of motion throughout the exercise.

## P10-E - LOW-confidence re-record guidance path

- Status: `blocked`
- Confidence: `LOW`
- Reps: `0`
- Form: `unknown`
- Coaching blocked: `True`
- Whisper transcription: ``
- Gemma status: `blocked`
- TTS status: `not_requested`
- Pose preview available: `True`

Final response:

Analysis confidence is low because no valid repetitions were detected. Please record again with the full movement visible. The system should not give confident form coaching from this attempt.

## P10-F - Text follow-up reuses stored structured result

- Status: `success`
- Confidence: `HIGH`
- Reps: `10`
- Form: `correct`
- Coaching blocked: `False`
- Whisper transcription: ``
- Gemma status: `success`
- TTS status: `not_requested`
- Pose preview available: `True`

Final response:

You completed 10 reps.

## P10-G - Voice question plus visible Whisper transcription plus TTS

- Status: `success`
- Confidence: `HIGH`
- Reps: `10`
- Form: `correct`
- Coaching blocked: `False`
- Whisper transcription: `How many reps did I complete?`
- Gemma status: `success`
- TTS status: `success`
- Pose preview available: `True`

Final response:

You completed 10 reps.

## P10-H - TTS failure preserves text answer

- Status: `success`
- Confidence: `HIGH`
- Reps: `10`
- Form: `correct`
- Coaching blocked: `False`
- Whisper transcription: ``
- Gemma status: `success`
- TTS status: `failed`
- Pose preview available: `True`

Final response:

You completed 10 reps.
