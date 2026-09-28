# Phase 13 Error Propagation Results

Phase 13 used representative evidence from existing frozen evaluations. It did not modify selected models, movement thresholds, confidence thresholds, verified labels, prompts, or Streamlit behaviour.

## Summary

| Pipeline | Cases | No propagation | Stopped | Propagated transparently | Propagated to incorrect output |
|---|---:|---:|---:|---:|---:|
| visual | 10 | 3 | 4 | 0 | 3 |
| audio | 10 | 2 | 0 | 5 | 3 |

Table 1. Representative Phase 13 propagation outcomes.

## Visual Error Propagation

The visual subset included 10 cases. Clean controls `VIS-01` and `VIS-02` remained correct enough. Four LOW-confidence cases were stopped before normal Gemma coaching: two occlusion cases and two partial-body cases.

The strongest propagated visual error was `VIS-03` (`vid16`, `occlusion`). The baseline result was 10 reps, while the degraded condition produced 21 reps and form `shallow` with HIGH confidence. Gemma received these structured facts and generated a response from them. This is an upstream visual/movement error, not an LLM hallucination.

The strongest blocked visual example was `VIS-05` (`vid13`, `occlusion`). The degraded condition produced 0 reps and LOW confidence with reasons `LOW_REQUIRED_JOINT_AVAILABILITY;LOW_ANGLE_CALCULABILITY;NO_REPS_DETECTED`. Normal coaching was blocked.

## Speech Error Propagation

The speech subset included 10 Whisper small recordings. Exact or intent-preserved transcriptions had limited downstream effect. Larger transcription errors changed the language-model input and could change the final answer.

The strongest STT propagated-error example was `AUD-09` (`stt_005`). The intended question was "Why was my curl marked as half range?", but Whisper transcribed "Why was Michael market as". Gemma answered the transcription it received, so the final response could become inappropriate for the original spoken question.

## Confidence Gate Finding

The confidence gate successfully stopped some low-quality visual inputs, especially cases with no detected repetitions or poor required-joint/angle availability. However, HIGH confidence did not guarantee semantic correctness. Some degraded videos still produced usable-looking evidence and HIGH confidence while the movement result was wrong.

## Gemma Grounding Finding

In visual propagated-error cases, Gemma generally followed the structured movement facts it received. When those facts were wrong, the final answer could still be wrong overall even though Gemma was grounded. This distinguishes upstream-origin error from LLM-origin hallucination.

## Transcription Visibility Finding

The current Streamlit app displays successful Whisper transcriptions after the voice interaction has been processed. This improves transparency because the user can see what text was used, but it does not prevent propagation during that interaction because the transcription is not shown for correction before Gemma is called.

## Limitations

This is a representative error-propagation study, not a new full benchmark. It uses 10 visual cases and 10 speech cases selected from frozen evidence. Intent and answer-change classifications include researcher judgement and should be reported as qualitative analysis rather than population-level statistics.
