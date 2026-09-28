# STT Evaluation Status

Phase 6 STT benchmarking has been completed on the 24 real recorded WAV files in `dataset/stt/audio/`.

Key outputs:

- `whisper_tiny_results.csv`
- `whisper_base_results.csv`
- `whisper_small_results.csv`
- `whisper_model_comparison.csv`
- `stt_group_summary.csv`
- `whisper_model_selection.md`

Selected variant:

```text
whisper_small
```

Selection basis:

```text
accuracy-prioritised trade-off using WER first, then transcription failures and latency
```

Final benchmark summary:

| Model | Size MB | Average WER | Median WER | Exact Match Rate | Failed | Mean Latency | Mean RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| whisper_tiny | 72.1 | 0.2848 | 0.2429 | 0.2083 | 0 | 0.4065 s | 0.0813 |
| whisper_base | 138.5 | 0.2383 | 0.1834 | 0.1250 | 0 | 0.7212 s | 0.1442 |
| whisper_small | 461.2 | 0.1921 | 0.1429 | 0.2917 | 0 | 2.2729 s | 0.4546 |

Whisper small was selected because transcription quality was prioritised for this application. Tiny is substantially faster and smaller, and base is an intermediate option, but small produced the strongest WER and exact-match results on the fixed 24-recording benchmark. The recordings are approximately 5 seconds long, so small's RTF of 0.4546 still means faster-than-real-time processing on the tested CPU environment.

Main limitation: this controlled test set has limited speaker and acoustic diversity. Results may differ with other speakers, accents, microphones, background noise, or longer utterances.

Next phase:

```text
Phase 7 - Compare Language Models for Coaching Feedback
```
