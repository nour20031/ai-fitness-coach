# Phase 11 Technical Test Plan

Phase 11 verifies whether the frozen Phase 10 system behaves correctly for defined technical cases. This is not a new model comparison, tuning pass, robustness experiment, or user evaluation.

## System Under Test

Final frozen pipeline:

```text
Video -> MediaPipe Pose / BlazePose -> movement analysis -> confidence gate
-> structured result -> gemma2:2b -> text coaching -> optional Windows SAPI TTS

Voice question -> Whisper small -> transcribed question -> gemma2:2b
```

The selected models remain fixed:

- Visual model: MediaPipe Pose / BlazePose
- Speech-to-text model: Whisper small
- Language model: `gemma2:2b` through Ollama
- Optional spoken output: Windows SAPI TTS

The rule-based movement analyser and confidence gate are deterministic system components, not pretrained model slots.

## Test Matrix

The Phase 11 matrix contains 64 technical test entries. One entry, `P11-UNIT-001`, runs an automated unit/contract suite containing three individual unit tests; all three passed in the final Phase 11 run. This means the final count remains 64 matrix entries, not 67 separate Phase 11 matrix rows.

| Test ID | Component | Test type | Input | Expected result | Evidence |
| --- | --- | --- | --- | --- | --- |
| P11-UNIT-001 | Automated tests | Unit | `python -m unittest discover -s tests` | Contract/session tests pass | `tests/test_phase11_contracts.py` |
| P11-VID-001 | Video input | Component | Valid MP4 | Readable video and metadata | `technical_test_results.csv` |
| P11-VID-002 | Video input | Negative | Missing path | Graceful failure, no crash | `technical_failure_matrix.csv` |
| P11-VID-003 | Video input | Negative | Corrupted MP4 | Rejected as unreadable | `technical_failure_matrix.csv` |
| P11-VID-004 | Video input | Component | Supported uploaded video | Duration, FPS, width, height returned | `technical_test_results.csv` |
| P11-VID-005 | Video input | Negative | Non-video file | Rejected before analysis | `technical_test_results.csv` |
| P11-POSE-001 | Pose model | Component | `vid16` | Pose detection succeeds | `technical_test_results.csv` |
| P11-POSE-002 | Pose model | Component | `vid16` | Required landmarks available | `technical_test_results.csv` |
| P11-POSE-003 | Pose model | Negative | Blank frame | No crash, no fabricated pose | `technical_failure_matrix.csv` |
| P11-POSE-004 | Pose model | Contract | Usable frame | Angle can be calculated | `technical_test_results.csv` |
| P11-MOVE-001 | Movement analysis | Regression | `vid16` squat | 10 reps, correct, HIGH | `technical_test_results.csv` |
| P11-MOVE-002 | Movement analysis | Regression | `vid24` squat | 10 reps, shallow, HIGH | `technical_test_results.csv` |
| P11-MOVE-003 | Movement analysis | Regression | `vid06` curl | 10 reps, correct, HIGH | `technical_test_results.csv` |
| P11-MOVE-004 | Movement analysis | Regression | `vid12` curl | 10 reps, half_range, HIGH | `technical_test_results.csv` |
| P11-MOVE-005 | Movement analysis | Regression | `vid23` squat | 0 reps, unknown, LOW | `technical_test_results.csv` |
| P11-READY-001 | Bicep ready option | Regression | `vid06`, ready ON | 10 reps, correct, HIGH | `technical_test_results.csv` |
| P11-READY-002 | Bicep ready option | Regression | `vid12`, ready ON | 9 reps, half_range, HIGH | `technical_test_results.csv` |
| P11-FORM-001 | Form classification | Regression | `vid16` | Correct squat | `technical_test_results.csv` |
| P11-FORM-002 | Form classification | Regression | `vid24` | Shallow squat | `technical_test_results.csv` |
| P11-FORM-003 | Form classification | Regression | `vid06` | Correct curl | `technical_test_results.csv` |
| P11-FORM-004 | Form classification | Regression | `vid12` | Half-range curl | `technical_test_results.csv` |
| P11-CONF-001 | Confidence gate | Component | `vid16` | HIGH confidence | `technical_test_results.csv` |
| P11-CONF-002 | Confidence gate | Component | `vid23` | LOW with reason codes | `technical_test_results.csv` |
| P11-CONF-003 | Confidence gate | Integration | `vid23` + question | Normal Gemma coaching blocked | `integration_test_results.csv` |
| P11-CONF-004 | Confidence gate | Integration | `vid16` + question | Gemma coaching permitted | `technical_test_results.csv` |
| P11-CONF-005 | Confidence gate | Negative | Missing evidence | No confident feedback | `technical_test_results.csv` |
| P11-CONTRACT-001 | Structured result | Contract | `StructuredMovementResult` | Required fields exist | `technical_test_results.csv` |
| P11-CONTRACT-002 | Structured result | Contract | `StructuredMovementResult` | Field types are correct | `technical_test_results.csv` |
| P11-CONTRACT-003 | Structured result | Contract | Gemma prompt | No raw video path passed | `technical_test_results.csv` |
| P11-CONTRACT-004 | Structured result | Contract | LOW confidence result | LOW evidence remains traceable | `technical_test_results.csv` |
| P11-CONTRACT-005 | Structured result | Contract | Normal result | Rep history matches rep count | `technical_test_results.csv` |
| P11-STT-001 | Whisper small | Component | `stt_002.wav` | Non-empty expected transcription | `technical_test_results.csv` |
| P11-STT-002 | Whisper small | Negative | Empty audio | Failure, no fabricated text | `technical_failure_matrix.csv` |
| P11-STT-003 | Whisper small | Negative | Missing audio | Failure handled | `technical_failure_matrix.csv` |
| P11-STT-004 | Whisper small | Component | Difficult sample | Actual Whisper output recorded | `technical_test_results.csv` |
| P11-LLM-001 | Gemma | Integration | HIGH facts + rep question | Grounded answer uses rep count | `technical_test_results.csv` |
| P11-LLM-002 | Gemma | Integration | Shallow facts | No unrelated issue invented | `technical_test_results.csv` |
| P11-LLM-003 | Gemma | Integration | Half-range facts | Grounded to half-range | `technical_test_results.csv` |
| P11-LLM-004 | Gemma | Integration | Unmeasured fact question | No fabricated measurement claim | `technical_test_results.csv` |
| P11-LLM-005 | Gemma | Integration | Medical question | No diagnosis/treatment claim | `technical_test_results.csv` |
| P11-LLM-006 | Gemma | Simulated failure | Failing LLM client | Clear failure stage | `technical_failure_matrix.csv` |
| P11-LLM-007 | Gemma | Simulated failure | Empty LLM client | Empty response treated as failure | `technical_failure_matrix.csv` |
| P11-TTS-001 | Windows SAPI TTS | Component | Short text | WAV generated | `technical_test_results.csv` |
| P11-TTS-002 | Windows SAPI TTS | Component | Longer text | WAV generated | `technical_test_results.csv` |
| P11-TTS-003 | Windows SAPI TTS | Negative | Empty text | Failure, no audio | `technical_test_results.csv` |
| P11-TTS-004 | Windows SAPI TTS | Simulated failure | Fake failing TTS | Failed TTS result | `technical_failure_matrix.csv` |
| P11-TTS-005 | Windows SAPI TTS | Component | Generated WAV | Duration greater than zero | `technical_test_results.csv` |
| P11-ORCH-001 | Orchestration | Integration | `vid16` + text question | 10/correct/HIGH + Gemma success | `integration_test_results.csv` |
| P11-ORCH-002 | Orchestration | End-to-end | `vid16` + `stt_002.wav` + TTS | MediaPipe, Whisper, Gemma, TTS success | `integration_test_results.csv` |
| P11-ORCH-003 | Orchestration | Integration | `vid23` | LOW confidence blocks coaching | `integration_test_results.csv` |
| P11-ORCH-004 | Orchestration | Negative | Valid video + empty audio | Failure at Whisper | `integration_test_results.csv` |
| P11-ORCH-005 | Orchestration | Simulated failure | Failing Gemma client | Failure at Gemma | `integration_test_results.csv` |
| P11-ORCH-006 | Orchestration | Simulated failure | Failing TTS | Text preserved, TTS failed | `integration_test_results.csv` |
| P11-UI-001 to P11-UI-011 | Streamlit/session | Component/integration | Existing UI smoke and state checks | Stale-state and visible-output behaviours hold | Streamlit CSV evidence |

## Failure-Handling Matrix

The test suite records expected and observed behaviour for:

- Missing video
- Unreadable video
- No pose / insufficient evidence
- Zero repetitions
- LOW confidence
- Missing audio
- Empty audio
- Whisper failure
- Gemma unavailable
- Empty Gemma response
- TTS failure

The system must not fabricate output when a component fails.

## Output Files

The consolidated runner is:

```powershell
.\venv\Scripts\python.exe scripts\run_phase11_technical_tests.py
```

Expected output files:

- `dataset/processed/evaluation/technical_testing/technical_test_results.csv`
- `dataset/processed/evaluation/technical_testing/technical_test_summary.json`
- `dataset/processed/evaluation/technical_testing/technical_failure_matrix.csv`
- `dataset/processed/evaluation/technical_testing/component_test_summary.csv`
- `dataset/processed/evaluation/technical_testing/integration_test_results.csv`
- `docs/TECHNICAL_TEST_RESULTS.md`

## Boundary

If a test fails, the failure is recorded. Phase 11 does not tune thresholds, change selected models, change dataset labels, or start Phase 12 robustness testing.
