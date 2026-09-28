# Movement Analysis Rules

Phase 4 fixes MediaPipe Pose / BlazePose as the pose-estimation model and finalises the explainable rule-based movement-analysis layer. The rule layer is not counted as a pretrained AI model. Its job is to convert pose landmarks into joint angles, repetition counts, and a small set of exercise-specific form labels.

The final Phase 4 pipeline is:

```text
video -> MediaPipe Pose / BlazePose -> landmarks -> joint angles/features -> rep counting -> form classification
```

Rep counting and form classification are deliberately separate. A shallow squat is still a squat repetition, and a half-range bicep curl is still a curl repetition. The form label is assigned after the movement cycle has been counted.

## Shared Angle Formula

For all exercises, the angle at joint `B` is calculated from three landmarks `A-B-C`:

```text
angle = absolute angle between vector BA and vector BC
if angle > 180 degrees:
    angle = 360 - angle
```

The result is a 0-180 degree joint angle. Larger values usually mean the limb is more extended; smaller values mean the joint is more flexed.

## Squat Analysis

### Joints Used

The squat analyser uses:

| Feature | Landmarks |
| --- | --- |
| Knee angle | hip, knee, ankle |
| Torso/hip geometry | shoulder, hip, knee |
| Visibility selection | left and right hip/knee/ankle |

Table 1. Squat movement-analysis features.

### Visible-Side Handling

The analyser checks both legs. If both sides are visible, it averages the knee angles to reduce noise. If only one side is reliable, it uses the visible side. The active side for drawing and torso checking is the side with the smaller knee angle.

### Thresholds

| Rule | Threshold |
| --- | ---: |
| Enter down state | knee angle < 165 degrees |
| Finish repetition | after down state, knee angle > 165 degrees |
| Shallow-depth threshold | minimum knee angle > 115 degrees |
| Minimum down frames | 3 frames |
| Minimum landmark visibility | 0.5 |

Table 2. Squat thresholds used in Phase 4.

### Rep State Machine

```text
state = none
for each frame:
    calculate visible knee angle

    if knee angle < 165:
        state = down
        increase down-frame counter
        record lowest knee angle in current rep

    if state is down and down-frame counter >= 3 and knee angle > 165:
        count one repetition
        classify this rep using the lowest knee angle
        reset current-rep tracking
```

### Form Classification

Each counted squat stores the lowest knee angle reached during the repetition.

```text
if lowest knee angle > 115:
    rep issue = not_deep_enough
else:
    rep is depth-correct

if more than half of detected reps are not_deep_enough:
    video label = shallow
else:
    video label = correct
```

This means a shallow squat is counted first, then labelled as shallow. The counting rule does not require the squat to be deep.

## Bicep-Curl Analysis

### Joints Used

The bicep-curl analyser uses:

| Feature | Landmarks |
| --- | --- |
| Elbow angle | shoulder, elbow, wrist |
| Shoulder movement | left/right shoulder y-position |
| Visibility selection | left and right shoulder/elbow/wrist |

Table 3. Bicep-curl movement-analysis features.

### Left/Right Handling

The analyser checks both arms. If both arms are visible, it averages the elbow angles to reduce false counts from one noisy arm. If only one arm is reliable, the visible arm is used. The most flexed visible arm is used for drawing angle feedback.

For uploaded dataset videos, the analyser does not require the clip to start from a ready pose because some source recordings begin mid-movement. For live webcam use, the app can still require an extended-arm ready pose to avoid accidental first reps.

### Thresholds

| Rule | Threshold |
| --- | ---: |
| Extended/down trigger | elbow angle > 125 degrees |
| Curled/up trigger | elbow angle < 115 degrees |
| Half-range threshold | minimum elbow angle > 55 degrees |
| Stable frames required | 4 frames |
| Cooldown after counting | 8 frames |
| Minimum landmark visibility | 0.5 |

Table 4. Bicep-curl thresholds used in Phase 4.

### Rep State Machine

```text
state = none
for each frame:
    calculate visible elbow angle
    record minimum and maximum elbow angle in current movement

    if elbow angle > 125 for stable frames:
        if state was up and cooldown is finished:
            count one repetition
            classify this rep using the minimum elbow angle
            reset current-rep tracking
        else:
            state = down

    if elbow angle < 115 for stable frames and state is down:
        state = up
```

### Form Classification

Each counted curl stores the smallest elbow angle reached during the repetition.

```text
if minimum elbow angle > 55:
    rep issue = half_range_of_motion
else:
    rep is full-range

if more than half of detected reps are half_range_of_motion:
    video label = half_range
else:
    video label = correct
```

This means a half-range curl is counted first, then labelled as incomplete range. The counting rule does not require a full-range curl.

## Known Limitations

- The rules depend on MediaPipe landmark quality, so poor lighting, occlusion, or body parts outside the frame can reduce reliability.
- The current squat depth threshold is explainable but simple; body proportions and camera height can affect measured knee angle.
- The current curl half-range threshold is based on elbow angle only. It does not fully model load, tempo, wrist position, or elbow drift.
- Front and side views can produce different landmark visibility. The rules handle both views, but results may differ by viewpoint.
- The final label is based on the majority of detected repetitions, so videos with severe rep-count errors may also produce unreliable form labels.
