# Phase 4 Movement Analysis Results

MediaPipe Pose / BlazePose is fixed as the pose model. These results evaluate the explainable rule-based movement-analysis layer for squat and bicep curl on the frozen 23 original source videos.

## Overall Results

| video_count | expected_reps | detected_reps | rep_count_mae | exact_count_rate | within_plus_minus_1_rate | form_label_match_rate | failure_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 23 | 230 | 205 | 2.5652 | 0.3478 | 0.5652 | 0.913 | 0.4783 |

Table 1. Overall movement-analysis performance on the frozen source-video benchmark.

## Group Results

| group_value | video_count | rep_count_mae | exact_count_rate | within_plus_minus_1_rate | form_label_match_rate |
| --- | --- | --- | --- | --- | --- |
| bicep_curl | 14 | 2.6429 | 0.2857 | 0.5 | 0.9286 |
| squat | 9 | 2.4444 | 0.4444 | 0.6667 | 0.8889 |

Table 2. Movement-analysis performance by exercise.

| group_value | video_count | rep_count_mae | exact_count_rate | within_plus_minus_1_rate | form_label_match_rate |
| --- | --- | --- | --- | --- | --- |
| front | 13 | 4.1538 | 0.0769 | 0.3846 | 0.9231 |
| side | 10 | 0.5 | 0.7 | 0.8 | 0.9 |

Table 3. Movement-analysis performance by camera view.

| group_value | video_count | rep_count_mae | exact_count_rate | within_plus_minus_1_rate | form_label_match_rate |
| --- | --- | --- | --- | --- | --- |
| correct | 15 | 2.0667 | 0.3333 | 0.6 | 0.9333 |
| half_range | 6 | 3.0 | 0.3333 | 0.5 | 1.0 |
| shallow | 2 | 5.0 | 0.5 | 0.5 | 0.5 |

Table 4. Movement-analysis performance by verified form label.

## Failure Cases

| source_video_id | exercise | form_label | camera_view | expected_reps | detected_reps | absolute_rep_count_error | predicted_form_label | failure_categories | likely_failure_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vid01 | bicep_curl | correct | front | 10 | 2 | 8 | correct | REP_COUNT_ERROR | curl_rep_state_logic_or_partial_arm_visibility |
| vid03 | bicep_curl | correct | front | 10 | 15 | 5 | correct | REP_COUNT_ERROR | curl_rep_state_logic_or_partial_arm_visibility |
| vid04 | bicep_curl | correct | front | 10 | 12 | 2 | correct | REP_COUNT_ERROR | curl_rep_state_logic_or_partial_arm_visibility |
| vid05 | bicep_curl | correct | side | 10 | 10 | 0 | half_range | FORM_LABEL_MISMATCH | form_threshold_or_movement_range |
| vid07 | bicep_curl | correct | side | 10 | 12 | 2 | correct | REP_COUNT_ERROR | curl_rep_state_logic_or_partial_arm_visibility |
| vid09 | bicep_curl | half_range | front | 10 | 1 | 9 | half_range | REP_COUNT_ERROR | curl_rep_state_logic_or_partial_arm_visibility |
| vid10 | bicep_curl | half_range | front | 10 | 5 | 5 | half_range | REP_COUNT_ERROR | curl_rep_state_logic_or_partial_arm_visibility |
| vid11 | bicep_curl | half_range | front | 10 | 13 | 3 | half_range | REP_COUNT_ERROR | curl_rep_state_logic_or_partial_arm_visibility |
| vid15 | squat | correct | front | 10 | 2 | 8 | correct | REP_COUNT_ERROR | squat_rep_state_logic_or_camera_angle |
| vid21 | squat | correct | side | 10 | 12 | 2 | correct | REP_COUNT_ERROR | squat_rep_state_logic_or_camera_angle |
| vid23 | squat | shallow | front | 10 | 0 | 10 | unknown | NO_REPS_DETECTED;REP_COUNT_ERROR;FORM_LABEL_MISMATCH | rep_state_logic_or_insufficient_visible_movement |

Table 5. Videos requiring review after the final Phase 4 movement-analysis evaluation.
