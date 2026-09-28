# Pose Model Selection

Selected model: `MediaPipe Pose / BlazePose`
Alternative model not selected: `MoveNet`

The selection is based on the measured Phase 3 benchmark metrics, not on which model was already integrated.

## Overall Evidence

| Metric | MediaPipe | MoveNet |
| --- | ---: | ---: |
| rep_count_mae | 2.8261 | 3.1739 |
| exact_count_rate | 0.2609 | 0.2174 |
| within_plus_minus_1_rate | 0.3913 | 0.3913 |
| form_label_match_rate | 0.8696 | 0.8696 |
| mean_required_joint_availability | 0.6448 | 0.9324 |
| mean_angle_calculability | 0.9952 | 0.9958 |
| failure_rate | 0.6957 | 0.7826 |
| mean_effective_fps | 27.3205 | 34.7589 |

## Decision Notes

- MediaPipe score across decision criteria: 3.
- MoveNet score across decision criteria: 3.
- MediaPipe task-performance wins: 3.
- MoveNet task-performance wins: 0.
- MediaPipe Pose / BlazePose was selected for the final movement-analysis pipeline because it achieved lower repetition-count error and a higher exact-count rate on the primary benchmark. These application-level measures were prioritised because accurate repetition analysis is central to the fitness-coaching objective. MoveNet demonstrated substantially higher required-joint availability and processing speed, but these advantages did not translate into better overall repetition-count or form-label performance on this project dataset. The selection is therefore a task-specific trade-off rather than evidence that MediaPipe is universally superior.
- Strengths and weaknesses should be interpreted with the per-video failure cases and group summaries.
- Next phase: Phase 4 — Finalise Movement Analysis using the selected MediaPipe Pose / BlazePose model.
