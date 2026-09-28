# Phase 8 Orchestration Results

Phase 8 is an integration demonstration, not a full system-performance evaluation. It verifies that the three selected pretrained models can be connected through a traceable orchestration layer.

## Evidence Files

- `dataset/processed/evaluation/orchestration/orchestration_test_results.csv`
- `dataset/processed/evaluation/orchestration/orchestration_examples.md`
- `dataset/processed/evaluation/orchestration/orchestration_trace_examples.json`

## Scenario Summary

| Scenario | Purpose | Status | Confidence | Reps | Form label | Whisper | Gemma | Total latency |
| --- | --- | --- | --- | ---: | --- | --- | --- | ---: |
| A | Correct squat + text request | success | HIGH | 10 | correct | not requested | success | 28.2255s |
| B | Shallow squat + text request | success | HIGH | 10 | shallow | not requested | success | 22.8664s |
| C | Correct bicep curl + text request | success | HIGH | 10 | correct | not requested | success | 22.2716s |
| D | Half-range curl + text request | success | HIGH | 10 | half_range | not requested | success | 17.1575s |
| E | Upstream movement-error propagation | success | HIGH | 1 | half_range | success | success | 20.6209s |
| E2 | Clean three-model success | success | HIGH | 10 | correct | success | success | 26.4068s |
| F | LOW-confidence blocking | blocked | LOW | 0 | unknown | not requested | blocked | 22.5310s |
| G | Whisper empty-audio failure | failed | HIGH | 10 | half_range | failed | not called | 18.2203s |
| H | Gemma unavailable failure | failed | HIGH | 10 | half_range | not requested | failed | 18.1957s |
| I | STT error propagation | success | HIGH | 1 | half_range | success | success | 17.9296s |

Table 1. Phase 8 orchestration integration results.

## Clean Three-Model Success

Scenario E2 demonstrates a clean successful orchestration path:

```text
exercise video
-> MediaPipe Pose / BlazePose
-> movement analysis
-> confidence gate
-> Whisper small voice transcription
-> gemma2:2b grounded response
```

Input video: `vid16`.

Voice question audio: `stt_002.wav`.

Whisper transcription:

```text
How many reps did I complete?
```

Structured facts:

```json
{
  "exercise": "squat",
  "rep_count": 10,
  "form_label": "correct",
  "detected_issues": [],
  "confidence_status": "HIGH"
}
```

Gemma response:

```text
You completed 10 reps.
```

Component latency:

| Component | Latency |
| --- | ---: |
| MediaPipe + movement + confidence | 24.1547s |
| Whisper small | 1.7276s |
| Gemma | 0.5232s |
| Total | 26.4068s |

Table 2. Clean three-model orchestration latency for one local run.

The 26.4068-second value represents one Phase 8 integration run on the tested local CPU environment. It should not be interpreted as the final average system latency. Full performance testing, including repeated runs and possible model-loading or cold-start effects, is deferred to a later technical-evaluation phase.

## Upstream Movement-Error Propagation

Scenario E remains important error-propagation evidence. It used `vid09`, where the movement-analysis layer detected:

```json
{
  "exercise": "bicep_curl",
  "rep_count": 1,
  "form_label": "half_range",
  "detected_issues": ["half_range_of_motion"],
  "confidence_status": "HIGH"
}
```

Whisper correctly transcribed:

```text
How many reps did I complete?
```

Gemma answered:

```text
You completed 1 rep.
```

Gemma can remain fully grounded to the supplied structured facts while the final coaching response is still incorrect if an upstream movement-analysis error has already entered the pipeline. This demonstrates that language-model grounding prevents new hallucinations but does not correct upstream pose or repetition-count errors.

Scenario E component latency in this run was:

| Component | Latency |
| --- | ---: |
| MediaPipe + movement + confidence | 15.3134s |
| Whisper small | 4.6446s |
| Gemma | 0.6618s |
| Total | 20.6209s |

Table 3. Upstream movement-error propagation example latency for one local run.

## Low-Confidence Block

Scenario F used `vid23`, the known low-confidence shallow squat case. The confidence gate returned:

```json
{
  "confidence_status": "LOW",
  "confidence_reasons": [
    "NO_REPS_DETECTED",
    "INSUFFICIENT_MOVEMENT_SIGNAL"
  ]
}
```

The orchestrator blocked confident coaching and returned:

```text
Analysis confidence is low because no valid repetitions were detected. Please record again with the full movement visible. The system should not give confident form coaching from this attempt.
```

This demonstrates:

```text
weak visual/movement evidence
-> confidence gate
-> coaching blocked
-> error propagation stopped
```

## Failure Handling

Scenario G tested empty-audio handling. Video analysis succeeded, but Whisper received no valid audio signal. The orchestrator returned:

```text
Audio file is missing or empty
```

Gemma was not given a fabricated question.

Scenario H tested Gemma/Ollama failure handling by requesting a missing model tag. The orchestrator returned:

```text
Ollama/Gemma request failed: HTTP Error 404: Not Found
```

No fabricated coaching response was returned.

## STT Error Propagation

Scenario I demonstrates speech-to-text error propagation.

The intended prompt was:

```text
Why was my curl marked as half range?
```

Whisper transcribed:

```text
Why was Michael market as
```

Gemma then answered the distorted transcription and referred to "Michael". This shows:

```text
STT error
-> altered user question
-> potentially irrelevant LLM response
```

## Phase 8 Conclusion

Phase 8 demonstrates real orchestration of three selected pretrained models operating across three different data spaces:

- Visual: MediaPipe Pose / BlazePose
- Audio: Whisper small
- Language: `gemma2:2b`

The movement-analysis rules, confidence gate, and Ollama runtime are supporting components. They are not counted as additional pretrained models.

## Limitations

- Phase 8 proves integration, not final system accuracy.
- Full robustness, usability, and end-to-end system evaluation remain later phases.
- The full-video MediaPipe step dominates latency on CPU.
- The latency values are single-run integration measurements, not final average system performance.
- The LLM still depends on upstream movement facts and cannot correct pose or rep-counting errors.
- LOW-confidence cases currently return fixed guidance rather than a rich conversational explanation.

## Next Step

Phase 9 - Add Text-to-Speech after the three-model text orchestration has been proven.
