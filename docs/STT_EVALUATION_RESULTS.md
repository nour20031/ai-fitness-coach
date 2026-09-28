# Speech-to-Text Evaluation Results

Phase 6 compared three Whisper variants on the fixed 24-recording speech test set. All recordings were real human speech recorded from the prompts in `dataset/stt/stt_test_manifest.csv`.

## Benchmark Setup

| Item | Value |
| --- | --- |
| STT model family | Whisper |
| Variants | tiny, base, small |
| Library version | 20250625 |
| Device | CPU |
| Test recordings | 24 |
| Audio format | WAV |
| Output folder | `dataset/processed/evaluation/stt/` |

Table 1. Speech-to-text benchmark setup.

## Main Results

| Model | Size MB | Average WER | Median WER | Exact Match Rate | Failed Transcriptions | Mean Latency | Mean RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Whisper tiny | 72.1 | 0.2848 | 0.2429 | 0.2083 | 0 | 0.4065 s | 0.0813 |
| Whisper base | 138.5 | 0.2383 | 0.1834 | 0.1250 | 0 | 0.7212 s | 0.1442 |
| Whisper small | 461.2 | 0.1921 | 0.1429 | 0.2917 | 0 | 2.2729 s | 0.4546 |

Table 2. Whisper variant comparison on the fixed STT test set.

## Selected Variant

The selected model is **Whisper small**.

Whisper small was selected using an accuracy-prioritised selection strategy. Average Word Error Rate was the primary criterion because transcription errors can propagate into the later language-model stage. Failed transcriptions and latency were treated as secondary criteria. Whisper small achieved the lowest average and median WER and the highest exact-match rate, with zero failed transcriptions. Although it was slower and larger than tiny and base, its mean latency of 2.2729 seconds is acceptable for short spoken questions in the prototype.

This is a task-specific selection, not a claim that Whisper small is universally superior. Whisper tiny is substantially faster and smaller, and Whisper base provides an intermediate resource requirement. Small was chosen because transcription quality was prioritised for this fitness-coaching application.

The benchmark recordings are approximately 5 seconds long. Whisper small had a mean real-time factor of 0.4546, so it still processed the benchmark audio faster than real time on the tested CPU environment. Its latency should therefore be interpreted as acceptable for short user questions rather than simply "fast."

## Error Examples

| Model | Audio | Reference | Prediction | WER |
| --- | --- | --- | --- | ---: |
| tiny | stt_010 | Should I record the video again? | Thank you for watching! | 1.0000 |
| tiny | stt_022 | Did my elbows bend enough in the curl? | LPU, bend the enough in this curl. | 0.6250 |
| base | stt_022 | Did my elbows bend enough in the curl? | Help you bend enough in this curve. | 0.6250 |
| base | stt_005 | Why was my curl marked as half range? | Why was my current market as | 0.5000 |
| small | stt_005 | Why was my curl marked as half range? | Why was Michael market as | 0.6250 |
| small | stt_023 | What should I fix before my next set? | Watch the effects before my next set. | 0.5000 |

Table 3. Representative transcription errors.

## Interpretation

The results show a clear accuracy/runtime trade-off. Tiny is appropriate when speed and small model size are the main priorities, but its errors include fitness-specific word substitutions such as "curl" becoming "current" or "curls" becoming "curse." Base reduces average WER over tiny while keeping a moderate resource requirement, but it does not consistently improve exact matches. Small gives the best overall transcription quality, which is more important for the coaching assistant because misunderstood questions can lead to irrelevant feedback. Even the selected Whisper small model still makes domain-specific transcription errors for terms such as "curl", "half range", and "confidence", so speech input should remain editable or confirmable in the final app.

## Limitations

- The STT comparison used 24 real human-recorded English questions under a controlled project environment.
- All three Whisper variants received the exact same recordings, which provides a fair paired comparison between variants.
- Speaker and acoustic diversity are limited. Results may differ with other speakers, accents, microphones, background noise, or longer utterances.
- The benchmark used CPU only.
- Exact-match rate is strict; minor wording changes can fail exact match even when the meaning remains understandable.
- Some terms important to the project, such as "curl", "half range", and "confidence", still caused errors and should be checked in the final app.

## Phase Status

Selected STT model: Whisper small.

Selection basis: accuracy-prioritised trade-off using WER first, then transcription failures and latency.

Main limitation: controlled 24-recording test set with limited speaker and acoustic diversity.

Next phase: Phase 7 - Compare Language Models for Coaching Feedback.
