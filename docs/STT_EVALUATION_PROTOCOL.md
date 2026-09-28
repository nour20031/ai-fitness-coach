# Speech-to-Text Evaluation Protocol

Phase 6 compares Whisper variants for the speech-to-text component of the fitness coach. This is the second required pretrained-model slot in the final orchestration.

## Candidate Models

The planned comparison uses the same local Whisper implementation for all variants:

| Candidate | Purpose |
| --- | --- |
| Whisper tiny | Fastest and smallest baseline |
| Whisper base | Middle option for accuracy/runtime trade-off |
| Whisper small | Higher-capacity option, likely slower |

Table 1. Whisper variants planned for Phase 6.

## Fixed Test Set

The fixed test set is defined in:

```text
dataset/stt/stt_test_manifest.csv
```

It contains 24 short English fitness-coach questions across four categories:

- repetition question
- form question
- improvement question
- confidence question

The benchmark must use real recorded human speech as the main evaluation data. Synthetic TTS audio must not be used as the main benchmark.

## Recording Procedure

Record the test set with:

```powershell
.\venv\Scripts\python.exe scripts\record_stt_test_set.py
```

Recording requirements:

- Use the same microphone for all 24 recordings.
- Record in a quiet room.
- Read each prompt clearly in English.
- Do not change the reference text after recording.
- Save each file to the path already specified in the manifest.

## Benchmark Procedure

After all 24 WAV files exist, run:

```powershell
.\venv\Scripts\python.exe scripts\run_whisper_benchmark.py
```

The script will compare:

- Whisper tiny
- Whisper base
- Whisper small

and save results under:

```text
dataset/processed/evaluation/stt/
```

## Metrics

For every audio file and model variant, record:

- reference transcription
- predicted transcription
- word error rate
- exact-match result
- failed/empty transcription
- latency
- audio duration
- real-time factor

Summary metrics:

- average WER
- median WER
- exact-match rate
- failed transcription count
- mean latency
- mean real-time factor

## Current Status

The Phase 6 benchmark has been completed on 24 real speech recordings. Results and model selection are documented in `docs/STT_EVALUATION_RESULTS.md` and `dataset/processed/evaluation/stt/whisper_model_selection.md`.
