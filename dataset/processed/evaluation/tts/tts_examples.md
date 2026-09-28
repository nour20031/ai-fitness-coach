# Phase 9 TTS Examples

These examples demonstrate Text-to-Speech as an additional output layer after the final text response. TTS does not alter pose, movement-analysis, confidence, Whisper, or Gemma outputs.

## P9-A - Clean video plus Whisper question plus Gemma answer plus TTS

- Status: `success`
- Confidence: `HIGH`
- Coaching blocked: `False`
- Detected reps: `10`
- Form label: `correct`
- Whisper transcription: `How many reps did I complete?`
- Gemma status: `success`
- TTS status: `success`
- TTS audio: `./dataset\processed\evaluation\tts\audio\integration_clean_e2.wav`
- Total latency: `30.7723` seconds

Final response:

You completed 10 reps.

## P9-B - LOW-confidence re-record guidance spoken by TTS

- Status: `blocked`
- Confidence: `LOW`
- Coaching blocked: `True`
- Detected reps: `0`
- Form label: `unknown`
- Whisper transcription: ``
- Gemma status: `blocked`
- TTS status: `success`
- TTS audio: `./dataset\processed\evaluation\tts\audio\integration_low_confidence.wav`
- Total latency: `22.9597` seconds

Final response:

Analysis confidence is low because no valid repetitions were detected. Please record again with the full movement visible. The system should not give confident form coaching from this attempt.

## P9-C - TTS failure does not remove text coaching

- Status: `success`
- Confidence: `HIGH`
- Coaching blocked: `False`
- Detected reps: `10`
- Form label: `correct`
- Whisper transcription: ``
- Gemma status: `success`
- TTS status: `failed`
- TTS audio: ``
- Total latency: `25.443` seconds

Final response:

You completed 10 reps.
