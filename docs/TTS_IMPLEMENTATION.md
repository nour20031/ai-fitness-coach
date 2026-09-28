# Text-to-Speech Implementation

Phase 9 adds Text-to-Speech as an output-presentation component after the final coaching response has already been generated. It is not one of the three mandatory pretrained models. The three core pretrained models remain MediaPipe Pose / BlazePose, Whisper small, and `gemma2:2b`.

The mandatory pretrained-model roles remain:

1. MediaPipe Pose / BlazePose for visual pose estimation
2. Whisper small for speech-to-text
3. `gemma2:2b` for grounded language feedback

Windows SAPI TTS is an additional output/interface component only.

## Selected TTS Solution

The implemented TTS component uses Windows SAPI through PowerShell COM automation. This was selected because it is local, offline, lightweight, and already available on the Windows laptop without installing a large additional model.

Implementation file:

`src/audio/text_to_speech.py`

Service class:

`WindowsSapiTextToSpeech`

The service accepts final response text and returns a structured result containing:

- `status`
- `audio_path`
- `latency_seconds`
- `engine`
- `voice`
- `rate`
- `audio_format`
- `audio_duration_seconds`
- `error`

Generated audio is stored under:

`dataset/processed/evaluation/tts/audio/`

## Architecture

```text
Exercise video
-> MediaPipe Pose / BlazePose
-> movement analysis
-> confidence gate
-> structured facts
-> gemma2:2b
-> final text response
-> Windows SAPI TTS
-> spoken coaching response
```

For voice interaction:

```text
User voice
-> Whisper small
-> transcribed question
-> gemma2:2b + structured movement facts
-> final text response
-> Windows SAPI TTS
-> spoken answer
```

TTS speaks the final response only. It does not alter the movement result, confidence status, Whisper transcription, or Gemma response text.

## Low-Confidence Handling

If the confidence gate returns `LOW`, the system does not generate normal form coaching. It returns the existing re-record guidance. TTS is allowed to speak that guidance because it is only reading the already-approved final message.

Example:

```text
Analysis confidence is low because no valid repetitions were detected. Please record again with the full movement visible.
```

## Failure Handling

TTS is optional. If audio generation fails, the overall coaching text remains available. The orchestrator records:

- `tts.status = failed`
- `tts.error`
- no fabricated audio path

This means a text coaching interaction can still succeed even when spoken output fails.

## Integration Point

The orchestrator in `src/orchestration/fitness_coach_orchestrator.py` now supports:

- `enable_tts`
- `tts_output_name`
- `tts` result block
- `tts_latency_seconds`

TTS is disabled by default so earlier Phase 8 tests keep their existing behaviour unless speech output is explicitly requested.
