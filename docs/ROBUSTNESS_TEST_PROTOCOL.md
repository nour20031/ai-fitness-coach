# Phase 12 Robustness Test Protocol

Phase 12 tests how the frozen final visual pipeline behaves when exercise-video input is degraded. It does not tune models, thresholds, labels, or app behaviour.

## Paired Design

The same 23 original source videos are analysed in baseline form and under deterministic derived robustness conditions. Each robustness variant inherits the source video ID, anonymous participant ID, exercise, verified label, camera view, and expected repetition count from its original recording.

Robustness variants are derived from existing recordings and do not represent additional participants or independent observations.

## Conditions And Parameters

| Condition | Parameters |
| --- | --- |
| baseline | original source video; no transform |
| darker | cv2.convertScaleAbs(alpha=0.60, beta=-10) |
| brighter | cv2.convertScaleAbs(alpha=1.18, beta=25) |
| rotation | 5 degree centre rotation, borderMode=BORDER_REPLICATE; reused Phase 3 preliminary rotation level |
| crop | centre crop 5 percent margins on all sides, resized back to original dimensions; reused Phase 3 crop/zoom level |
| partial_body | exercise-specific visibility stress: squat masks bottom 22 percent; bicep_curl masks right 26 percent |
| poor_framing | translate frame +16 percent width and +8 percent height with black border |
| fast_movement | temporal compression proxy: keep every second frame at original fps, approximately half duration |
| occlusion | deterministic black rectangle over relevant joints: squat central lower body; bicep_curl upper/side arm area |

## Metrics

- Pose/data quality: processed frames, pose detection coverage, required-joint availability, angle calculability, tracking dropout, usable frame rate, and movement signal range.
- Movement analysis: expected reps, detected reps, absolute rep-count error, exact-count rate, within +/-1 rate, expected label, predicted label, and form-label match.
- Confidence behaviour: HIGH/LOW status, confidence reason codes, and whether coaching would be blocked.
- Failure categories: POSE_NOT_DETECTED, LOW_REQUIRED_JOINT_AVAILABILITY, LOW_ANGLE_CALCULABILITY, HIGH_TRACKING_DROPOUT, NO_REPS_DETECTED, REP_COUNT_ERROR_GT_1, FORM_MISMATCH, LOW_CONFIDENCE, COACHING_BLOCKED.

## Synthetic Condition Limitation

Controlled transformations are proxies. A digitally rotated video is not identical to a physically tilted camera, temporal compression is not identical to a participant naturally moving faster, and artificial occlusion is not equivalent to every real-world obstruction. These variants are paired stress-test inputs, not new dataset samples.
