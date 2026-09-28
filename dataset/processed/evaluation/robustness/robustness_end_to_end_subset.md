# Phase 12 Representative End-to-End Robustness Checks

This small subset checks that HIGH-confidence transformed inputs receive grounded Gemma feedback and LOW-confidence transformed inputs block normal coaching. It is not the full Phase 13 error-propagation study.

| Scenario | Robustness ID | Condition | Status | Confidence | Gemma | Passed |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_success | vid16_baseline | baseline | success | HIGH | success | True |
| darker_success | vid16_darker | darker | success | HIGH | success | True |
| crop_high_confidence_rep_error | vid01_crop | crop | success | HIGH | success | True |
| occlusion_low_confidence_block | vid13_occlusion | occlusion | blocked | LOW | blocked | True |
| fast_movement_timing_stress | vid16_fast_movement | fast_movement | success | HIGH | success | True |
