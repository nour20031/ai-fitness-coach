# Phase 8 Orchestration Examples

These examples demonstrate integration of MediaPipe Pose / BlazePose, Whisper small, and gemma2:2b through the orchestration layer. They are not a full system-performance evaluation.

Scenario E2 is the clean three-model success case: the video result is reliable, Whisper transcribes the question correctly, and Gemma answers from the structured facts.

Scenario E is kept as upstream movement-error propagation evidence: Gemma stays grounded to the supplied structured facts, but the structured facts already contain a repetition-count error.

Scenario I is kept as STT error-propagation evidence: Whisper distorts the user question, and Gemma responds to that distorted text.

## A - Correct squat with text coaching request

- Video: `vid16`
- Exercise: `squat`
- Status: `success`
- Confidence: `HIGH`
- Coaching blocked: `False`
- Detected reps: `10`
- Form label: `correct`
- Whisper transcription: ``
- Gemma status: `success`
- Total latency: `28.2255` seconds

Structured result excerpt:

```json
{
  "exercise": "squat",
  "rep_count": 10,
  "form_label": "correct",
  "detected_issues": [],
  "measured_features": {
    "minimum_knee_angle": 56.3,
    "per_rep_min_knee_angles": [
      67.0,
      58.0,
      61.9,
      60.7,
      60.3,
      60.1,
      60.3,
      62.1,
      56.3,
      60.3
    ]
  },
  "confidence_status": "HIGH",
  "confidence_reasons": []
}
```

Final response:

Your squat form looks good overall! You're maintaining a good knee angle throughout the movement.  Focus on keeping your knees tracking in line with your toes.

Trace stages:

```json
[
  "video_pose_movement_confidence",
  "gemma2b"
]
```

## B - Shallow squat with coaching request

- Video: `vid24`
- Exercise: `squat`
- Status: `success`
- Confidence: `HIGH`
- Coaching blocked: `False`
- Detected reps: `10`
- Form label: `shallow`
- Whisper transcription: ``
- Gemma status: `success`
- Total latency: `22.8664` seconds

Structured result excerpt:

```json
{
  "exercise": "squat",
  "rep_count": 10,
  "form_label": "shallow",
  "detected_issues": [
    "not_deep_enough"
  ],
  "measured_features": {
    "minimum_knee_angle": 113.4,
    "per_rep_min_knee_angles": [
      119.1,
      114.7,
      121.9,
      113.4,
      130.6,
      128.2,
      129.0,
      127.8,
      124.5,
      120.6
    ]
  },
  "confidence_status": "HIGH",
  "confidence_reasons": []
}
```

Final response:

The analysis detected that your squat form was shallow.  This means your knees didn't go deep enough during the squat.  The minimum knee angle was 113.4 degrees, which is less than the minimum depth threshold.

Trace stages:

```json
[
  "video_pose_movement_confidence",
  "gemma2b"
]
```

## C - Correct bicep curl

- Video: `vid06`
- Exercise: `bicep_curl`
- Status: `success`
- Confidence: `HIGH`
- Coaching blocked: `False`
- Detected reps: `10`
- Form label: `correct`
- Whisper transcription: ``
- Gemma status: `success`
- Total latency: `22.2716` seconds

Structured result excerpt:

```json
{
  "exercise": "bicep_curl",
  "rep_count": 10,
  "form_label": "correct",
  "detected_issues": [],
  "measured_features": {
    "minimum_elbow_angle": 31.6,
    "maximum_elbow_angle": 174.0,
    "maximum_shoulder_deviation": 45.0,
    "per_rep_min_elbow_angles": [
      31.6,
      37.1,
      35.7,
      36.0,
      36.1,
      33.7,
      37.0,
      37.7,
      39.1,
      37.8
    ],
    "per_rep_max_elbow_angles": [
      168.5,
      174.0,
      169.0,
      169.8,
      165.0,
      164.6,
      166.2,
      163.6,
      160.5,
      163.7
    ]
  },
  "confidence_status": "HIGH",
  "confidence_reasons": []
}
```

Final response:

Based on the analysis, your form appears to be mostly correct.  You're showing good elbow angles and shoulder deviation.  However, there are some areas where you could improve.  

For example, you've shown some variation in elbow angles, and it's important to maintain a consistent range of motion throughout the exercise. 

Remember to focus on maintaining proper form and technique throughout the repetitions.

Trace stages:

```json
[
  "video_pose_movement_confidence",
  "gemma2b"
]
```

## D - Half-range bicep curl

- Video: `vid12`
- Exercise: `bicep_curl`
- Status: `success`
- Confidence: `HIGH`
- Coaching blocked: `False`
- Detected reps: `10`
- Form label: `half_range`
- Whisper transcription: ``
- Gemma status: `success`
- Total latency: `17.1575` seconds

Structured result excerpt:

```json
{
  "exercise": "bicep_curl",
  "rep_count": 10,
  "form_label": "half_range",
  "detected_issues": [
    "half_range_of_motion"
  ],
  "measured_features": {
    "minimum_elbow_angle": 76.4,
    "maximum_elbow_angle": 164.5,
    "maximum_shoulder_deviation": 16.0,
    "per_rep_min_elbow_angles": [
      99.0,
      80.6,
      79.4,
      76.4,
      83.9,
      76.9,
      82.7,
      77.8,
      81.4,
      82.2
    ],
    "per_rep_max_elbow_angles": [
      138.3,
      160.8,
      164.0,
      161.0,
      161.3,
      159.3,
      156.8,
      162.2,
      164.5,
      158.8
    ]
  },
  "confidence_status": "HIGH",
  "confidence_reasons": []
}
```

Final response:

The analysis detected that your elbow movement during the bicep curl was limited to half the typical range of motion. This means your elbow didn't reach the full range of motion during the exercise. 

Remember, proper form is important for maximizing the benefits of your workout and preventing injuries.

Trace stages:

```json
[
  "video_pose_movement_confidence",
  "gemma2b"
]
```

## E - Complete three-model path with real recorded voice question

- Video: `vid09`
- Exercise: `bicep_curl`
- Status: `success`
- Confidence: `HIGH`
- Coaching blocked: `False`
- Detected reps: `1`
- Form label: `half_range`
- Whisper transcription: `How many reps did I complete?`
- Gemma status: `success`
- Total latency: `20.6209` seconds

Structured result excerpt:

```json
{
  "exercise": "bicep_curl",
  "rep_count": 1,
  "form_label": "half_range",
  "detected_issues": [
    "half_range_of_motion"
  ],
  "measured_features": {
    "minimum_elbow_angle": 100.3,
    "maximum_elbow_angle": 178.6,
    "maximum_shoulder_deviation": 17.5,
    "per_rep_min_elbow_angles": [
      100.3
    ],
    "per_rep_max_elbow_angles": [
      178.6
    ]
  },
  "confidence_status": "HIGH",
  "confidence_reasons": []
}
```

Final response:

You completed 1 rep.

Trace stages:

```json
[
  "video_pose_movement_confidence",
  "whisper_small",
  "gemma2b"
]
```

## E2 - Clean three-model path with accurate 10-rep result

- Video: `vid16`
- Exercise: `squat`
- Status: `success`
- Confidence: `HIGH`
- Coaching blocked: `False`
- Detected reps: `10`
- Form label: `correct`
- Whisper transcription: `How many reps did I complete?`
- Gemma status: `success`
- Total latency: `26.4068` seconds

Structured result excerpt:

```json
{
  "exercise": "squat",
  "rep_count": 10,
  "form_label": "correct",
  "detected_issues": [],
  "measured_features": {
    "minimum_knee_angle": 56.3,
    "per_rep_min_knee_angles": [
      67.0,
      58.0,
      61.9,
      60.7,
      60.3,
      60.1,
      60.3,
      62.1,
      56.3,
      60.3
    ]
  },
  "confidence_status": "HIGH",
  "confidence_reasons": []
}
```

Final response:

You completed 10 reps.

Trace stages:

```json
[
  "video_pose_movement_confidence",
  "whisper_small",
  "gemma2b"
]
```

## F - LOW-confidence shallow squat blocked from coaching

- Video: `vid23`
- Exercise: `squat`
- Status: `blocked`
- Confidence: `LOW`
- Coaching blocked: `True`
- Detected reps: `0`
- Form label: `unknown`
- Whisper transcription: ``
- Gemma status: `blocked`
- Total latency: `22.531` seconds

Structured result excerpt:

```json
{
  "exercise": "squat",
  "rep_count": 0,
  "form_label": "unknown",
  "detected_issues": [],
  "measured_features": {
    "minimum_knee_angle": null,
    "per_rep_min_knee_angles": []
  },
  "confidence_status": "LOW",
  "confidence_reasons": [
    "NO_REPS_DETECTED",
    "INSUFFICIENT_MOVEMENT_SIGNAL"
  ]
}
```

Final response:

Analysis confidence is low because no valid repetitions were detected. Please record again with the full movement visible. The system should not give confident form coaching from this attempt.

Trace stages:

```json
[
  "video_pose_movement_confidence",
  "gemma2b"
]
```

## G - Whisper empty-audio failure handling

- Video: `vid05`
- Exercise: `bicep_curl`
- Status: `failed`
- Confidence: `HIGH`
- Coaching blocked: `True`
- Detected reps: `10`
- Form label: `half_range`
- Whisper transcription: ``
- Gemma status: ``
- Total latency: `18.2203` seconds

Structured result excerpt:

```json
{
  "exercise": "bicep_curl",
  "rep_count": 10,
  "form_label": "half_range",
  "detected_issues": [
    "half_range_of_motion"
  ],
  "measured_features": {
    "minimum_elbow_angle": 55.2,
    "maximum_elbow_angle": 163.2,
    "maximum_shoulder_deviation": 13.5,
    "per_rep_min_elbow_angles": [
      56.8,
      58.8,
      55.2,
      64.7,
      67.3,
      60.6,
      66.9,
      71.1,
      74.1,
      73.2
    ],
    "per_rep_max_elbow_angles": [
      163.2,
      149.2,
      155.1,
      149.5,
      153.4,
      150.4,
      140.1,
      147.8,
      151.1,
      152.2
    ]
  },
  "confidence_status": "HIGH",
  "confidence_reasons": []
}
```

Final response:

Audio file is missing or empty: [temporary local path excluded]

Trace stages:

```json
[
  "video_pose_movement_confidence",
  "whisper_small"
]
```

Transcription error:

Audio file is missing or empty: [temporary local path excluded]

## H - Ollama/Gemma unavailable handling

- Video: `vid05`
- Exercise: `bicep_curl`
- Status: `failed`
- Confidence: `HIGH`
- Coaching blocked: `True`
- Detected reps: `10`
- Form label: `half_range`
- Whisper transcription: ``
- Gemma status: `failed`
- Total latency: `18.1957` seconds

Structured result excerpt:

```json
{
  "exercise": "bicep_curl",
  "rep_count": 10,
  "form_label": "half_range",
  "detected_issues": [
    "half_range_of_motion"
  ],
  "measured_features": {
    "minimum_elbow_angle": 55.2,
    "maximum_elbow_angle": 163.2,
    "maximum_shoulder_deviation": 13.5,
    "per_rep_min_elbow_angles": [
      56.8,
      58.8,
      55.2,
      64.7,
      67.3,
      60.6,
      66.9,
      71.1,
      74.1,
      73.2
    ],
    "per_rep_max_elbow_angles": [
      163.2,
      149.2,
      155.1,
      149.5,
      153.4,
      150.4,
      140.1,
      147.8,
      151.1,
      152.2
    ]
  },
  "confidence_status": "HIGH",
  "confidence_reasons": []
}
```

Final response:

Ollama/Gemma request failed: HTTP Error 404: Not Found

Trace stages:

```json
[
  "video_pose_movement_confidence",
  "gemma2b"
]
```

Gemma error:

Ollama/Gemma request failed: HTTP Error 404: Not Found

## I - STT error propagation demonstration

- Video: `vid09`
- Exercise: `bicep_curl`
- Status: `success`
- Confidence: `HIGH`
- Coaching blocked: `False`
- Detected reps: `1`
- Form label: `half_range`
- Whisper transcription: `Why was Michael market as`
- Gemma status: `success`
- Total latency: `17.9296` seconds

Structured result excerpt:

```json
{
  "exercise": "bicep_curl",
  "rep_count": 1,
  "form_label": "half_range",
  "detected_issues": [
    "half_range_of_motion"
  ],
  "measured_features": {
    "minimum_elbow_angle": 100.3,
    "maximum_elbow_angle": 178.6,
    "maximum_shoulder_deviation": 17.5,
    "per_rep_min_elbow_angles": [
      100.3
    ],
    "per_rep_max_elbow_angles": [
      178.6
    ]
  },
  "confidence_status": "HIGH",
  "confidence_reasons": []
}
```

Final response:

The system detected that Michael's bicep curls had a "half_range_of_motion" issue. This means his elbow didn't reach the full range of motion during the exercise. 

To improve, focus on maintaining a full range of motion throughout the exercise.

Trace stages:

```json
[
  "video_pose_movement_confidence",
  "whisper_small",
  "gemma2b"
]
```
