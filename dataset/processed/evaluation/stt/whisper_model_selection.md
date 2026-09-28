# Whisper Model Selection

Selected variant: `whisper_small`

The selection uses an accuracy-prioritised strategy on the fixed recorded STT test set. Average Word Error Rate is the primary criterion because transcription errors can propagate into the later language-model coaching stage. Failed transcriptions and latency are secondary criteria.

| Model | Size MB | Average WER | Median WER | Exact Match Rate | Failed | Mean Latency | RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| whisper_tiny | 72.1 | 0.2848 | 0.2429 | 0.2083 | 0 | 0.4065 | 0.0813 |
| whisper_base | 138.5 | 0.2383 | 0.1834 | 0.125 | 0 | 0.7212 | 0.1442 |
| whisper_small | 461.2 | 0.1921 | 0.1429 | 0.2917 | 0 | 2.2729 | 0.4546 |

Whisper small was selected because it achieved the lowest average WER, lowest median WER, highest exact-match rate, and zero failed transcriptions. This is a task-specific trade-off for the fitness-coaching prototype, not evidence that Whisper small is universally superior. Whisper tiny is substantially faster and smaller, and Whisper base provides an intermediate resource requirement. Small was chosen because transcription quality was prioritised for short coaching questions.

The recordings are approximately 5 seconds long. Whisper small's mean RTF of 0.4546 means it processed the benchmark audio faster than real time on the tested CPU environment, although its mean latency of 2.2729 seconds should be described as acceptable rather than fast.

Main limitation: the comparison used 24 real human-recorded English questions under a controlled project environment. All three models received the exact same recordings, but speaker and acoustic diversity are limited, so results may differ with other speakers, accents, microphones, background noise, or longer utterances.

Next phase: Phase 7 - Compare Language Models for Coaching Feedback.
