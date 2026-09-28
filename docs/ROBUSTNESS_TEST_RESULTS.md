# Phase 12 Robustness Test Results

These results use controlled derived variants of the frozen 23-video benchmark. Derived robustness videos are paired stress-test inputs and are not counted as additional participants or independent dataset samples.

## Baseline

| condition | video_count | mean_rep_count_mae | exact_count_rate | within_plus_minus_1_rate | form_label_match_rate | high_confidence_rate | low_confidence_rate | mean_pose_detection_coverage | mean_required_joint_availability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 23 | 2.5652 | 0.3478 | 0.5652 | 0.913 | 0.9565 | 0.0435 | 0.9944 | 0.9937 |

## Condition Summary

| condition | video_count | mean_rep_count_mae | delta_rep_mae_vs_baseline | exact_count_rate | delta_exact_count_rate_vs_baseline | within_plus_minus_1_rate | form_label_match_rate | high_confidence_rate | low_confidence_rate | coaching_block_rate | mean_pose_detection_coverage | mean_required_joint_availability | mean_angle_calculability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 23 | 2.5652 | 0.0 | 0.3478 | 0.0 | 0.5652 | 0.913 | 0.9565 | 0.0435 | 0.0435 | 0.9944 | 0.9937 | 0.9937 |
| darker | 23 | 2.5217 | -0.0435 | 0.3478 | 0.0 | 0.6087 | 0.913 | 0.9565 | 0.0435 | 0.0435 | 0.9943 | 0.9929 | 0.9929 |
| brighter | 23 | 2.4783 | -0.0869 | 0.3478 | 0.0 | 0.5652 | 0.913 | 0.9565 | 0.0435 | 0.0435 | 0.9944 | 0.9934 | 0.9934 |
| rotation | 23 | 2.5652 | 0.0 | 0.3478 | 0.0 | 0.6087 | 0.913 | 0.9565 | 0.0435 | 0.0435 | 0.9948 | 0.994 | 0.994 |
| crop | 23 | 2.5217 | -0.0435 | 0.3043 | -0.0435 | 0.6522 | 0.913 | 0.9565 | 0.0435 | 0.0435 | 0.9941 | 0.9934 | 0.9934 |
| partial_body | 23 | 5.0435 | 2.4783 | 0.2174 | -0.1304 | 0.3913 | 0.6522 | 0.6522 | 0.3478 | 0.3478 | 0.9902 | 0.6361 | 0.6361 |
| poor_framing | 23 | 2.5217 | -0.0435 | 0.3043 | -0.0435 | 0.6087 | 0.913 | 0.9565 | 0.0435 | 0.0435 | 0.991 | 0.9829 | 0.9829 |
| fast_movement | 23 | 2.6957 | 0.1305 | 0.3043 | -0.0435 | 0.5652 | 0.8696 | 0.913 | 0.087 | 0.087 | 0.9942 | 0.9922 | 0.9922 |
| occlusion | 23 | 5.3913 | 2.8261 | 0.0 | -0.3478 | 0.1739 | 0.5217 | 0.6957 | 0.3043 | 0.3043 | 0.9478 | 0.7212 | 0.7212 |

Largest degradation by rep-count MAE delta: `occlusion`.

## Exercise Summary

| group_value | condition | video_count | mean_rep_count_mae | within_plus_minus_1_rate | form_label_match_rate | high_confidence_rate | mean_required_joint_availability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bicep_curl | baseline | 14 | 2.6429 | 0.5 | 0.9286 | 1.0 | 0.9908 |
| bicep_curl | brighter | 14 | 2.5 | 0.5 | 0.9286 | 1.0 | 0.9908 |
| bicep_curl | crop | 14 | 2.5714 | 0.5714 | 0.9286 | 1.0 | 0.9899 |
| bicep_curl | darker | 14 | 2.6429 | 0.5 | 0.9286 | 1.0 | 0.9905 |
| bicep_curl | fast_movement | 14 | 2.9286 | 0.4286 | 0.8571 | 0.9286 | 0.9905 |
| bicep_curl | occlusion | 14 | 4.9286 | 0.1429 | 0.5 | 0.6429 | 0.6523 |
| bicep_curl | partial_body | 14 | 2.3571 | 0.6429 | 0.9286 | 1.0 | 0.9815 |
| bicep_curl | poor_framing | 14 | 2.5 | 0.5714 | 0.9286 | 1.0 | 0.9838 |
| bicep_curl | rotation | 14 | 2.7143 | 0.5 | 0.9286 | 1.0 | 0.9915 |
| squat | baseline | 9 | 2.4444 | 0.6667 | 0.8889 | 0.8889 | 0.9981 |
| squat | brighter | 9 | 2.4444 | 0.6667 | 0.8889 | 0.8889 | 0.9974 |
| squat | crop | 9 | 2.4444 | 0.7778 | 0.8889 | 0.8889 | 0.9987 |
| squat | darker | 9 | 2.3333 | 0.7778 | 0.8889 | 0.8889 | 0.9967 |
| squat | fast_movement | 9 | 2.3333 | 0.7778 | 0.8889 | 0.8889 | 0.9949 |
| squat | occlusion | 9 | 6.1111 | 0.2222 | 0.5556 | 0.7778 | 0.8284 |
| squat | partial_body | 9 | 9.2222 | 0.0 | 0.2222 | 0.1111 | 0.0987 |
| squat | poor_framing | 9 | 2.5556 | 0.6667 | 0.8889 | 0.8889 | 0.9815 |
| squat | rotation | 9 | 2.3333 | 0.7778 | 0.8889 | 0.8889 | 0.998 |

## View Summary

| group_value | condition | video_count | mean_rep_count_mae | within_plus_minus_1_rate | form_label_match_rate | high_confidence_rate | mean_required_joint_availability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| front | baseline | 13 | 4.1538 | 0.3846 | 0.9231 | 0.9231 | 0.9904 |
| front | brighter | 13 | 4.0 | 0.3846 | 0.9231 | 0.9231 | 0.9904 |
| front | crop | 13 | 4.1538 | 0.3846 | 0.9231 | 0.9231 | 0.9897 |
| front | darker | 13 | 4.1538 | 0.3846 | 0.9231 | 0.9231 | 0.99 |
| front | fast_movement | 13 | 4.3846 | 0.3077 | 0.8462 | 0.8462 | 0.9899 |
| front | occlusion | 13 | 5.0769 | 0.2308 | 0.6154 | 0.8462 | 0.8838 |
| front | partial_body | 13 | 6.2308 | 0.2308 | 0.6154 | 0.6154 | 0.5972 |
| front | poor_framing | 13 | 4.0769 | 0.3846 | 0.9231 | 0.9231 | 0.9758 |
| front | rotation | 13 | 4.2308 | 0.3846 | 0.9231 | 0.9231 | 0.9911 |
| side | baseline | 10 | 0.5 | 0.8 | 0.9 | 1.0 | 0.9979 |
| side | brighter | 10 | 0.5 | 0.8 | 0.9 | 1.0 | 0.9973 |
| side | crop | 10 | 0.4 | 1.0 | 0.9 | 1.0 | 0.9982 |
| side | darker | 10 | 0.4 | 0.9 | 0.9 | 1.0 | 0.9968 |
| side | fast_movement | 10 | 0.5 | 0.9 | 0.9 | 1.0 | 0.9951 |
| side | occlusion | 10 | 5.8 | 0.1 | 0.4 | 0.5 | 0.5098 |
| side | partial_body | 10 | 3.5 | 0.6 | 0.7 | 0.7 | 0.6866 |
| side | poor_framing | 10 | 0.5 | 0.9 | 0.9 | 1.0 | 0.992 |
| side | rotation | 10 | 0.4 | 0.9 | 0.9 | 1.0 | 0.9978 |

## Representative Failure Cases

| source_video_id | exercise | verified_label | camera_view | condition | baseline_detected_reps | condition_detected_reps | condition_abs_error | predicted_form_label | confidence_status | failure_categories |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vid16 | squat | correct | side | occlusion | 10 | 21 | 11 | shallow | HIGH | REP_COUNT_ERROR_GT_1;FORM_MISMATCH |
| vid17 | squat | correct | front | occlusion | 11 | 21 | 11 | shallow | HIGH | REP_COUNT_ERROR_GT_1;FORM_MISMATCH |
| vid14 | bicep_curl | half_range | side | occlusion | 11 | 0 | 10 | unknown | LOW | LOW_REQUIRED_JOINT_AVAILABILITY;LOW_ANGLE_CALCULABILITY;NO_REPS_DETECTED;REP_COUNT_ERROR_GT_1;FORM_MISMATCH;LOW_CONFIDENCE;COACHING_BLOCKED |
| vid15 | squat | correct | front | partial_body | 2 | 0 | 10 | unknown | LOW | LOW_REQUIRED_JOINT_AVAILABILITY;LOW_ANGLE_CALCULABILITY;NO_REPS_DETECTED;REP_COUNT_ERROR_GT_1;FORM_MISMATCH;LOW_CONFIDENCE;COACHING_BLOCKED |
| vid17 | squat | correct | front | partial_body | 11 | 0 | 10 | unknown | LOW | LOW_REQUIRED_JOINT_AVAILABILITY;LOW_ANGLE_CALCULABILITY;NO_REPS_DETECTED;REP_COUNT_ERROR_GT_1;FORM_MISMATCH;LOW_CONFIDENCE;COACHING_BLOCKED |
| vid19 | squat | correct | front | partial_body | 11 | 0 | 10 | unknown | LOW | LOW_REQUIRED_JOINT_AVAILABILITY;LOW_ANGLE_CALCULABILITY;NO_REPS_DETECTED;REP_COUNT_ERROR_GT_1;FORM_MISMATCH;LOW_CONFIDENCE;COACHING_BLOCKED |
| vid20 | squat | correct | front | partial_body | 10 | 0 | 10 | unknown | LOW | LOW_REQUIRED_JOINT_AVAILABILITY;LOW_ANGLE_CALCULABILITY;NO_REPS_DETECTED;REP_COUNT_ERROR_GT_1;FORM_MISMATCH;LOW_CONFIDENCE;COACHING_BLOCKED |
| vid23 | squat | shallow | front | partial_body | 0 | 0 | 10 | unknown | LOW | LOW_REQUIRED_JOINT_AVAILABILITY;LOW_ANGLE_CALCULABILITY;NO_REPS_DETECTED;REP_COUNT_ERROR_GT_1;FORM_MISMATCH;LOW_CONFIDENCE;COACHING_BLOCKED |
| vid16 | squat | correct | side | partial_body | 10 | 0 | 10 | unknown | LOW | LOW_REQUIRED_JOINT_AVAILABILITY;LOW_ANGLE_CALCULABILITY;NO_REPS_DETECTED;REP_COUNT_ERROR_GT_1;FORM_MISMATCH;LOW_CONFIDENCE;COACHING_BLOCKED |
| vid24 | squat | shallow | side | partial_body | 10 | 0 | 10 | unknown | LOW | LOW_REQUIRED_JOINT_AVAILABILITY;LOW_ANGLE_CALCULABILITY;NO_REPS_DETECTED;REP_COUNT_ERROR_GT_1;FORM_MISMATCH;LOW_CONFIDENCE;COACHING_BLOCKED |
| vid13 | bicep_curl | half_range | side | occlusion | 10 | 0 | 10 | unknown | LOW | LOW_REQUIRED_JOINT_AVAILABILITY;LOW_ANGLE_CALCULABILITY;NO_REPS_DETECTED;REP_COUNT_ERROR_GT_1;FORM_MISMATCH;LOW_CONFIDENCE;COACHING_BLOCKED |
| vid23 | squat | shallow | front | poor_framing | 0 | 0 | 10 | unknown | LOW | NO_REPS_DETECTED;REP_COUNT_ERROR_GT_1;FORM_MISMATCH;LOW_CONFIDENCE;COACHING_BLOCKED |

## Paired Statistics

| condition | video_count | mean_abs_error_difference | median_abs_error_difference | improved_count | unchanged_count | worsened_count | wilcoxon_statistic | wilcoxon_p_value | statistical_notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| darker | 23 | -0.0435 | 0.0 | 2 | 20 | 1 |  |  | Wilcoxon not run: No module named 'scipy' |
| brighter | 23 | -0.087 | 0.0 | 1 | 22 | 0 |  |  | Wilcoxon not run: No module named 'scipy' |
| rotation | 23 | 0.0 | 0.0 | 1 | 21 | 1 |  |  | Wilcoxon not run: No module named 'scipy' |
| crop | 23 | -0.0435 | 0.0 | 3 | 18 | 2 |  |  | Wilcoxon not run: No module named 'scipy' |
| partial_body | 23 | 2.4783 | 0.0 | 3 | 11 | 9 |  |  | Wilcoxon not run: No module named 'scipy' |
| poor_framing | 23 | -0.0435 | 0.0 | 3 | 18 | 2 |  |  | Wilcoxon not run: No module named 'scipy' |
| fast_movement | 23 | 0.1304 | 0.0 | 2 | 17 | 4 |  |  | Wilcoxon not run: No module named 'scipy' |
| occlusion | 23 | 2.8261 | 1.0 | 2 | 5 | 16 |  |  | Wilcoxon not run: No module named 'scipy' |

## Limitations

- Synthetic transformations are controlled proxies, not independently recorded real-world conditions.
- Robustness variants do not increase participant diversity and are not counted as new source videos.
- Results show sensitivity of the frozen visual/movement/confidence pipeline; they do not perform the full combined AI error-propagation study reserved for Phase 13.
- No models, movement thresholds, confidence thresholds, or labels were changed during Phase 12.
