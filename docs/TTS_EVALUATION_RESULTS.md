# Text-to-Speech Evaluation Results

Phase 9 evaluates Text-to-Speech as an additional interface component. It is not a fourth required pretrained model and it does not change the three-model orchestration evidence from Phase 8.

## TTS Test Set

The fixed TTS test set contains 8 representative final responses in:

`dataset/tts/tts_test_cases.json`

The cases include:

- repetition-count feedback
- correct squat feedback
- shallow squat feedback
- correct bicep-curl feedback
- half-range curl feedback
- improvement guidance
- low-confidence re-record guidance
- a longer coaching response

## Generation Results

Caption: Phase 9 Windows SAPI TTS generation results on 8 representative coaching responses.

| Metric | Result |
| --- | --- |
| Engine | Windows SAPI |
| Voice | Microsoft David Desktop - English (United States) |
| Speech rate | 0 |
| Output format | WAV |
| Test cases | 8 |
| Successful generations | 8 |
| Failed generations | 0 |
| Mean generation latency | 0.3566 seconds |
| Minimum generation latency | 0.3463 seconds |
| Maximum generation latency | 0.3854 seconds |
| Mean audio duration | 7.8336 seconds |
| Minimum audio duration | 2.2242 seconds |
| Maximum audio duration | 13.3666 seconds |

Evidence file:

`dataset/processed/evaluation/tts/tts_test_results.csv`

Generated audio directory:

`dataset/processed/evaluation/tts/audio/`

## Integration Results

Caption: Phase 9 TTS integration tests showing successful spoken output, low-confidence spoken guidance, and TTS failure handling.

| Scenario | Result |
| --- | --- |
| Clean end-to-end spoken path | `vid16` + `stt_002.wav` produced 10 correct squat reps, Whisper transcribed "How many reps did I complete?", Gemma responded "You completed 10 reps.", and TTS generated `integration_clean_e2.wav`. |
| Low-confidence spoken guidance | `vid23` remained LOW confidence with 0 reps and `unknown` form; normal coaching was blocked and TTS spoke the re-record guidance. |
| TTS failure handling | A forced TTS failure left the text response available: "You completed 10 reps." The interaction status remained successful and `tts.status` was `failed`. |

Clean end-to-end latency for one representative Phase 9 local integration run:

- Movement / pose / confidence: 23.3560 seconds
- Whisper small: 4.1830 seconds
- Gemma: 2.8663 seconds
- TTS: 0.3647 seconds
- Total: 30.7723 seconds

Low-confidence spoken-guidance latency for one representative Phase 9 local integration run:

- Movement / pose / confidence: 22.5886 seconds
- Whisper small: 0.0 seconds
- Gemma: 0.0 seconds because coaching was blocked
- TTS: 0.3703 seconds
- Total: 22.9597 seconds

The reported end-to-end latency values are from individual Phase 9 local integration runs. They are not the final average system latency. Repeated technical performance testing is deferred to the later system-evaluation phase.

Evidence files:

- `dataset/processed/evaluation/tts/tts_integration_results.csv`
- `dataset/processed/evaluation/tts/tts_examples.md`

## Manual Intelligibility and Naturalness Review

The completed manual review sheet is stored at:

`dataset/processed/evaluation/tts/tts_manual_review.csv`

The scoring rubric was:

- Intelligibility: 2 = clearly understandable, 1 = understandable with minor problems, 0 = difficult to understand
- Naturalness: 2 = reasonably natural, 1 = robotic but understandable, 0 = distracting or poor

Caption: Researcher manual review of Phase 9 generated TTS samples.

| Metric | Result |
| --- | ---: |
| Samples manually reviewed | 8 |
| Mean intelligibility | 2.00 / 2 |
| Mean naturalness | 2.00 / 2 |
| Minimum intelligibility | 2 |
| Maximum intelligibility | 2 |
| Minimum naturalness | 2 |
| Maximum naturalness | 2 |

All eight generated samples were manually listened to by the researcher. Every sample received an intelligibility score of 2/2 and a naturalness score of 2/2 under the predefined rubric. This indicates that the selected Windows SAPI voice was considered clearly understandable and sufficiently natural for the prototype. These results are limited to one researcher, one local voice, and a small fixed set of coaching responses.

These subjective scores were provided by the researcher and were not generated automatically or judged by another LLM.

## Limitations

The TTS check is a suitability test, not a major model-comparison phase. It uses one local Windows SAPI voice and 8 fixed representative responses. Subjective audio scoring was performed by one researcher on the local Windows environment. Windows SAPI is Windows-specific, and no comparison with neural TTS was necessary because TTS is an optional presentation layer. Full user-level perception testing is still required to evaluate whether users find the spoken feedback clear, helpful, and comfortable during the final Streamlit interaction.
