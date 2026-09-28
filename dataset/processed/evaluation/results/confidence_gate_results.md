# Phase 5 Confidence Gate Results

The confidence gate uses pose/data-quality evidence and movement-analysis evidence. The implemented flow is: video -> MediaPipe Pose / BlazePose -> landmarks + visibility -> movement analysis -> reps + angles + movement signal -> confidence / validation gate -> final coaching feedback only if confidence is HIGH. Low-confidence videos are blocked from confident feedback and receive re-record guidance.

## Summary

- Videos evaluated: 23
- High confidence: 22
- Low confidence: 1

## Reason Codes

| reason_code | count |
| --- | --- |
| INSUFFICIENT_MOVEMENT_SIGNAL | 1 |
| NO_REPS_DETECTED | 1 |

Table 1. Low-confidence reason-code counts.

## Difficult-Video Check

| source_video_id | exercise | form_label | camera_view | detected_reps | confidence_status | movement_signal_range | reason_codes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vid01 | bicep_curl | correct | front | 2 | HIGH | 136.28 |  |
| vid03 | bicep_curl | correct | front | 15 | HIGH | 145.95 |  |
| vid09 | bicep_curl | half_range | front | 1 | HIGH | 30.35 |  |
| vid10 | bicep_curl | half_range | front | 5 | HIGH | 60.97 |  |
| vid15 | squat | correct | front | 2 | HIGH | 60.41 |  |
| vid23 | squat | shallow | front | 0 | LOW | 6.6 | NO_REPS_DETECTED;INSUFFICIENT_MOVEMENT_SIGNAL |

Table 2. Confidence-gate behaviour on known difficult Phase 4 videos.

High confidence does not mean high accuracy. `vid01`, `vid09`, `vid10`, and `vid15` passed the confidence gate because their visual and movement evidence was usable enough for analysis, but their repetition counts were still imperfect. These are movement-analysis accuracy limitations rather than low-confidence input failures.

## Blocked Videos

| source_video_id | exercise | form_label | camera_view | detected_reps | confidence_status | reason_codes | user_message |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vid23 | squat | shallow | front | 0 | LOW | NO_REPS_DETECTED;INSUFFICIENT_MOVEMENT_SIGNAL | Analysis confidence is low because no valid repetitions were detected. Please record again with the full movement visible. |

Table 3. Videos blocked from confident feedback.
