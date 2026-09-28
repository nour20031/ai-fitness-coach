# Pose Model Evaluation Protocol

This protocol defines the Phase 3 comparison between MediaPipe Pose and MoveNet. Phase 2 only establishes the framework; no MoveNet installation, benchmark execution, or model results are included here.

## Benchmark Dataset

The primary pose-model benchmark uses the 23 original volunteer source videos listed in:

`dataset/processed/evaluation/pose_benchmark_manifest.csv`

During Phase 3, two protected external source paths could not be decoded by OpenCV. The benchmark manifest therefore uses the canonical full-video copies under `dataset/raw/`, which were copied from the original volunteer videos during the dataset preparation stage. The benchmark still uses the same 23 original recordings, not repetition clips or augmented data.

The benchmark includes:

- 4 anonymous participants.
- Exercises: `squat` and `bicep_curl`.
- Verified form labels: `correct`, `shallow`, and `half_range`.
- Front and side camera views.
- Expected repetitions: 10 per source video, used as a validation reference.
- Frame-step: 3.

Original source videos are the primary benchmark. Augmented videos may later be used only for separate robustness testing. Augmented data must not count as new participants.

## Fair Comparison Rules

MediaPipe Pose and MoveNet must be evaluated using:

- the same 23 source videos
- the same frame-step value: `3`
- the same expected repetition counts
- the same verified source labels
- the same downstream angle formulas
- the same movement-analysis and repetition-counting logic
- the same machine and runtime measurement method
- the same failure-category definitions

The pose-only comparison can use all 23 videos because it is a paired technical benchmark. Any tuned downstream rules or learned classifiers must still use the Leave-One-Participant-Out strategy to avoid participant leakage.

## Common Required Joints

MediaPipe Pose and MoveNet have different landmark schemas, so the comparison must not be based on total landmark count. Instead, both models must be mapped into a shared required-joint interface:

- left shoulder
- right shoulder
- left elbow
- right elbow
- left wrist
- right wrist
- left hip
- right hip
- left knee
- right knee
- left ankle
- right ankle

These joints are sufficient for the current squat and bicep-curl movement analysis.

## Technical Metrics

For each model and source video, record:

- pose detection coverage
- required-joint availability
- angle calculability
- tracking dropout
- processing time
- milliseconds per frame
- effective FPS

## Downstream Metrics

For each model and source video, record:

- detected repetition count
- absolute repetition-count error
- mean absolute error across groups
- exact-count rate
- within +/-1 repetition rate
- form-label match where downstream form output is available
- failure rate

The verified form label comes from the source dataset. Rule-based outputs must not be used as ground truth.

## Failure Categories

Use the following categories consistently:

- `POSE_NOT_DETECTED`
- `REQUIRED_JOINT_MISSING`
- `LOW_LANDMARK_RELIABILITY`
- `ANGLE_NOT_CALCULABLE`
- `TRACKING_DROPOUT`
- `REP_COUNT_ERROR`
- `FORM_LABEL_MISMATCH`
- `PROCESSING_ERROR`

Multiple categories may apply to the same video.

## Output Files

Phase 2 creates:

- `dataset/processed/evaluation/pose_benchmark_manifest.csv`
- `dataset/processed/evaluation/pose_model_comparison_template.csv`

Phase 3 should later fill benchmark result files without changing the frozen Phase 1 segmentation outputs.

## Reporting Notes

Chapter 3 should describe this as the pose-model evaluation design: a paired benchmark using the same videos, joints, frame-step, labels, angle formulas, and downstream logic.

Chapter 5 should later report the actual results using the technical metrics, downstream metrics, failure categories, and paired comparison tables.
