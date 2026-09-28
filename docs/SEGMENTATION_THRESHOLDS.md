# Repetition Segmentation Threshold Notes

This note documents the current Phase 1B repetition-segmentation logic. These thresholds are for detecting movement cycles only; they are not ground-truth form labels and they are not used to decide whether a repetition is correct, shallow, or half-range.

## Signal Selection

- Front-view squats use the average of both knee-angle signals when both legs have usable visibility and movement range.
- Front-view bicep curls use the average of both elbow-angle signals when both arms have usable visibility and movement range.
- Side-view bicep curls and side-view squats select the side with the stronger combination of landmark visibility and observed movement range.

This was added because side-view recordings often show one limb clearly while the other side is partially occluded.

## Squat Cycle Detection

- Active threshold: knee angle below `150` degrees.
- Recovery threshold: knee angle above `155` degrees.
- Minimum signal drop: `18` degrees.

Reason: these thresholds detect a down-and-up knee-angle cycle without requiring a final judgement about squat quality. A shallow squat can still be a repetition, but videos with extremely small movement may still be flagged for review.

## Bicep Curl Cycle Detection

Front-view bicep curls use valley detection:

- Minimum prominence: `12` degrees.
- Recovery floor: elbow angle at or above `160` degrees.
- Valley ceiling: elbow angle at or below `166` degrees.
- Minimum valley spacing: `18` frames in the analysed frame sequence.

Reason: this preserved the previously verified front-view half-range cases, especially `vid09` and `vid10`.

Side-view bicep curls use adaptive cycle detection:

- Minimum signal range: `8` degrees.
- Minimum prominence floor: `8` degrees.
- Prominence fraction: `0.22` of the observed signal range.
- Minimum gap: `0.35` seconds.
- Minimum duration: `0.20` seconds.
- Maximum duration: `5.0` seconds.
- Peak search window: `4.0` seconds.

Reason: fixed full-extension thresholds can fail in side-view videos because one arm may be occluded or the visible elbow angle may not reach the same apparent extension as in front-view videos.

## Frame-Step Decision

Frame-step settings `1`, `2`, and `3` were compared on the current review set. Frame-step `1` sometimes detected more repetitions, but it also over-segmented some videos and was much slower. Frame-step `3` had the lowest mean absolute error on the problematic set and remains the current default.

## Remaining Limitations

- Some front-view bicep curl videos are still under- or over-segmented.
- Side-view bicep curls remain sensitive to occlusion and angle-trace quality.
- `vid23` has too little usable movement signal for automatic shallow-squat segmentation.
- `vid15` is much shorter than the expected 10-repetition assumption and remains a review case.
