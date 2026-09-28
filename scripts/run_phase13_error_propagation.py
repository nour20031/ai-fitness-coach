from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.orchestration.fitness_coach_orchestrator import FitnessCoachOrchestrator


ROBUSTNESS_RESULTS = ROOT / "dataset" / "processed" / "evaluation" / "robustness" / "robustness_video_results.csv"
STT_RESULTS = ROOT / "dataset" / "processed" / "evaluation" / "stt" / "whisper_small_results.csv"
OUT_DIR = ROOT / "dataset" / "processed" / "evaluation" / "error_propagation"
DOCS_DIR = ROOT / "docs"


VISUAL_CASES = [
    ("VIS-01", "vid16_baseline", "Clean squat baseline control."),
    ("VIS-02", "vid06_baseline", "Clean bicep-curl baseline control."),
    ("VIS-03", "vid16_occlusion", "Strong HIGH-confidence visual propagated error."),
    ("VIS-04", "vid17_occlusion", "Second HIGH-confidence occlusion propagated error."),
    ("VIS-05", "vid13_occlusion", "LOW-confidence occlusion case blocked before Gemma coaching."),
    ("VIS-06", "vid14_occlusion", "Second LOW-confidence occlusion case blocked before Gemma coaching."),
    ("VIS-07", "vid16_partial_body", "LOW-confidence partial-body squat case blocked."),
    ("VIS-08", "vid24_partial_body", "LOW-confidence partial-body shallow-squat case blocked."),
    ("VIS-09", "vid01_crop", "HIGH-confidence crop case with wrong repetition count."),
    ("VIS-10", "vid16_fast_movement", "Fast-movement condition that remained correct enough."),
]


AUDIO_CASES = [
    ("AUD-01", "stt_002", "INTENT_PRESERVED", "Exact repetition-count transcription."),
    ("AUD-02", "stt_007", "INTENT_PRESERVED", "Exact form-question transcription."),
    ("AUD-03", "stt_003", "INTENT_PARTIALLY_CHANGED", "Curl was transcribed as current, but broad improvement intent remains visible."),
    ("AUD-04", "stt_001", "INTENT_PRESERVED", "Marked as shallow became market at shallow, but the question is still understandable."),
    ("AUD-05", "stt_012", "INTENT_PRESERVED", "Auxiliary word was dropped but the range-of-motion intent remains."),
    ("AUD-06", "stt_010", "INTENT_PARTIALLY_CHANGED", "Should I record became Play record, leaving a related but distorted request."),
    ("AUD-07", "stt_020", "INTENT_PARTIALLY_CHANGED", "Low confidence became no confidence, preserving the topic with altered wording."),
    ("AUD-08", "stt_004", "INTENT_CHANGED", "Squats became scraps, changing the fitness intent."),
    ("AUD-09", "stt_005", "INTENT_CHANGED", "Curl half-range question became a Michael phrase."),
    ("AUD-10", "stt_022", "INTENT_CHANGED", "Elbow curl question became mostly unrelated text."),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str | None, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def as_int(value: str | None, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def as_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", value.lower())).strip()


def issue_from_form(form_label: str) -> list[str]:
    if form_label == "shallow":
        return ["not_deep_enough"]
    if form_label == "half_range":
        return ["half_range_of_motion"]
    return []


def confidence_message(status: str, reasons: list[str]) -> str:
    if status == "LOW":
        if "NO_REPS_DETECTED" in reasons:
            return "Analysis confidence is low because no valid repetitions were detected. Please record again with the full movement visible."
        return "Analysis confidence is low because the visual evidence was not reliable enough. Please record again with the full movement visible."
    return "Enough usable visual and movement information was available for analysis. This does not guarantee perfect repetition counting."


def structured_from_robustness(row: dict[str, str]) -> dict[str, Any]:
    reasons = [part for part in (row.get("confidence_reasons") or "").split(";") if part]
    detected_reps = as_int(row.get("detected_reps"))
    form_label = row.get("predicted_form_label") or "unknown"
    return {
        "exercise": row["exercise"],
        "rep_count": detected_reps,
        "form_label": form_label,
        "detected_issues": issue_from_form(form_label),
        "measured_features": {
            "movement_signal_range": as_float(row.get("movement_signal_range")),
            "pose_detection_coverage": as_float(row.get("pose_detection_coverage")),
            "required_joint_availability": as_float(row.get("required_joint_availability")),
            "angle_calculability": as_float(row.get("angle_calculability")),
        },
        "confidence_status": row.get("confidence_status") or "UNKNOWN",
        "confidence_metrics": {
            "usable_frame_rate": as_float(row.get("usable_frame_rate")),
            "required_joint_availability": as_float(row.get("required_joint_availability")),
            "angle_calculability_rate": as_float(row.get("angle_calculability")),
            "dropout_rate": as_float(row.get("tracking_dropout")),
            "movement_signal_range": as_float(row.get("movement_signal_range")),
            "processed_frames": as_int(row.get("processed_frames")),
            "pose_detected_frames": round(as_float(row.get("pose_detection_coverage")) * as_int(row.get("processed_frames"))),
        },
        "confidence_reasons": reasons,
        "confidence_message": confidence_message(row.get("confidence_status") or "", reasons),
        "rep_history": [],
        "thresholds": {},
    }


def classify_visual(row: dict[str, str]) -> tuple[str, str, str]:
    confidence = row.get("confidence_status") or ""
    blocked = as_bool(row.get("coaching_blocked"))
    rep_error = as_int(row.get("absolute_rep_count_error"))
    form_match = as_bool(row.get("form_label_match"))
    condition = row.get("condition") or ""

    if confidence == "LOW" and blocked:
        return (
            "POSE_OR_VISUAL",
            "STOPPED",
            "LOW confidence blocked normal coaching, so the degraded movement result did not reach Gemma.",
        )
    if rep_error <= 1 and form_match:
        return ("NONE", "NO_PROPAGATION", "Movement result remained correct enough for this representative trace.")
    if condition in {"occlusion", "partial_body", "crop", "fast_movement"}:
        return (
            "POSE_OR_VISUAL",
            "PROPAGATED_TO_INCORRECT_OUTPUT",
            "Degraded visual evidence produced HIGH-confidence structured facts with a wrong rep count or form label.",
        )
    return (
        "MOVEMENT_ANALYSIS",
        "PROPAGATED_TO_INCORRECT_OUTPUT",
        "Movement analysis produced incorrect structured facts that were still considered usable.",
    )


def visual_error_propagation(orchestrator: FitnessCoachOrchestrator) -> list[dict[str, Any]]:
    rows = read_csv(ROBUSTNESS_RESULTS)
    by_id = {row["robustness_id"]: row for row in rows}
    by_source_baseline = {
        row["source_video_id"]: row for row in rows if row["condition"] == "baseline"
    }
    output: list[dict[str, Any]] = []

    for case_id, robustness_id, case_note in VISUAL_CASES:
        row = by_id[robustness_id]
        baseline = by_source_baseline[row["source_video_id"]]
        error_origin, propagation_outcome, outcome_note = classify_visual(row)
        structured = structured_from_robustness(row)

        gemma_status = "not_called"
        gemma_response = structured["confidence_message"]
        if row.get("confidence_status") == "HIGH":
            response = orchestrator.ask_follow_up(
                structured,
                user_question="How many reps did I complete and what form did the analysis detect?",
                enable_tts=False,
            )
            gemma_status = (response.get("gemma") or {}).get("status", response.get("status", "unknown"))
            gemma_response = response.get("final_response", "")
        else:
            gemma_status = "blocked"

        output.append(
            {
                "case_id": case_id,
                "source_video_id": row["source_video_id"],
                "condition": row["condition"],
                "exercise": row["exercise"],
                "camera_view": row["camera_view"],
                "expected_reps": row["expected_reps"],
                "baseline_detected_reps": baseline["detected_reps"],
                "condition_detected_reps": row["detected_reps"],
                "rep_error": row["absolute_rep_count_error"],
                "expected_form": row["expected_form_label"],
                "baseline_form": baseline["predicted_form_label"],
                "condition_form": row["predicted_form_label"],
                "form_changed": str(baseline["predicted_form_label"] != row["predicted_form_label"]),
                "pose_detection_coverage": row["pose_detection_coverage"],
                "required_joint_availability": row["required_joint_availability"],
                "angle_calculability": row["angle_calculability"],
                "movement_signal_range": row["movement_signal_range"],
                "confidence_status": row["confidence_status"],
                "confidence_reasons": row["confidence_reasons"],
                "coaching_blocked": row["coaching_blocked"],
                "gemma_status": gemma_status,
                "gemma_input_reps": structured["rep_count"],
                "gemma_input_form": structured["form_label"],
                "gemma_response": gemma_response,
                "error_origin": error_origin,
                "propagation_outcome": propagation_outcome,
                "notes": f"{case_note} {outcome_note}",
            }
        )
    return output


def classify_audio(
    intent: str,
    reference_question: str,
    whisper_question: str,
    reference_response: str,
    transcribed_response: str,
) -> tuple[bool, str, str]:
    answer_changed = normalize_text(reference_response) != normalize_text(transcribed_response)
    if normalize_text(reference_question) == normalize_text(whisper_question):
        return answer_changed, "exact_transcription_no_stt_propagation", "NO_PROPAGATION"
    if intent == "INTENT_PRESERVED" and not answer_changed:
        return answer_changed, "minor_or_no_change", "NO_PROPAGATION"
    if intent == "INTENT_CHANGED":
        return answer_changed, "intent_changed", "PROPAGATED_TO_INCORRECT_OUTPUT"
    if intent == "INTENT_PRESERVED":
        return answer_changed, "transcription_error_visible_but_intent_preserved", "PROPAGATED_TRANSPARENTLY"
    return answer_changed, "intent_partially_changed", "PROPAGATED_TRANSPARENTLY"


def audio_error_propagation(orchestrator: FitnessCoachOrchestrator) -> list[dict[str, Any]]:
    stt_rows = {row["audio_id"]: row for row in read_csv(STT_RESULTS)}
    structured = structured_from_robustness(
        next(row for row in read_csv(ROBUSTNESS_RESULTS) if row["robustness_id"] == "vid16_baseline")
    )
    output: list[dict[str, Any]] = []

    for case_id, audio_id, intent_preserved, note in AUDIO_CASES:
        row = stt_rows[audio_id]
        reference_response = orchestrator.ask_follow_up(
            structured,
            user_question=row["reference_text"],
            enable_tts=False,
        )
        transcribed_response = orchestrator.ask_follow_up(
            structured,
            user_question=row["predicted_text"],
            enable_tts=False,
        )

        reference_text = reference_response.get("final_response", "")
        transcribed_text = transcribed_response.get("final_response", "")
        answer_changed, answer_change_type, propagation_outcome = classify_audio(
            intent_preserved,
            row["reference_text"],
            row["predicted_text"],
            reference_text,
            transcribed_text,
        )

        output.append(
            {
                "case_id": case_id,
                "audio_id": audio_id,
                "reference_text": row["reference_text"],
                "whisper_transcription": row["predicted_text"],
                "wer": row["wer"],
                "intent_preserved": intent_preserved,
                "intent_change_notes": note,
                "structured_result_id": "vid16_baseline",
                "reference_question_gemma_response": reference_text,
                "transcribed_question_gemma_response": transcribed_text,
                "answer_changed": str(answer_changed),
                "answer_change_type": answer_change_type,
                "transcription_visible_in_app": "True",
                "transcription_visible_before_response": "False",
                "error_origin": "NONE" if propagation_outcome == "NO_PROPAGATION" else "STT",
                "propagation_outcome": propagation_outcome,
                "notes": "Streamlit displays the transcription after processing in show_question_result(); the current app does not provide an edit-before-submit step.",
            }
        )
    return output


def build_summary(visual_rows: list[dict[str, Any]], audio_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for pipeline, rows in (("visual", visual_rows), ("audio", audio_rows)):
        counts = Counter(row["propagation_outcome"] for row in rows)
        extra: dict[str, Any] = {}
        if pipeline == "visual":
            extra = {
                "confidence_blocked_error_cases": sum(1 for row in rows if row["propagation_outcome"] == "STOPPED"),
                "high_confidence_allowed_incorrect_facts": sum(
                    1
                    for row in rows
                    if row["confidence_status"] == "HIGH"
                    and row["propagation_outcome"] == "PROPAGATED_TO_INCORRECT_OUTPUT"
                ),
                "intent_preserved_or_changed": "",
                "final_answer_changed": "",
            }
        else:
            extra = {
                "confidence_blocked_error_cases": "",
                "high_confidence_allowed_incorrect_facts": "",
                "intent_preserved_or_changed": "; ".join(
                    f"{key}={value}" for key, value in sorted(Counter(row["intent_preserved"] for row in rows).items())
                ),
                "final_answer_changed": sum(1 for row in rows if str(row["answer_changed"]).lower() == "true"),
            }
        summary.append(
            {
                "pipeline": pipeline,
                "total_cases": len(rows),
                "no_propagation": counts.get("NO_PROPAGATION", 0),
                "stopped": counts.get("STOPPED", 0),
                "propagated_transparently": counts.get("PROPAGATED_TRANSPARENTLY", 0),
                "propagated_to_incorrect_output": counts.get("PROPAGATED_TO_INCORRECT_OUTPUT", 0),
                **extra,
            }
        )
    return summary


def representative_traces(visual_rows: list[dict[str, Any]], audio_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_ids = {"VIS-01", "VIS-03", "VIS-05", "AUD-01", "AUD-03", "AUD-09"}
    rows = []
    for row in visual_rows:
        if row["case_id"] in selected_ids:
            rows.append(
                {
                    "case_id": row["case_id"],
                    "pipeline": "visual",
                    "input_id": row["source_video_id"],
                    "condition_or_audio": row["condition"],
                    "upstream_observation": f"{row['condition_detected_reps']} reps, {row['condition_form']}, confidence {row['confidence_status']}",
                    "downstream_response": row["gemma_response"],
                    "propagation_outcome": row["propagation_outcome"],
                    "interpretation": row["notes"],
                }
            )
    for row in audio_rows:
        if row["case_id"] in selected_ids:
            rows.append(
                {
                    "case_id": row["case_id"],
                    "pipeline": "audio",
                    "input_id": row["audio_id"],
                    "condition_or_audio": row["wer"],
                    "upstream_observation": f"Reference: {row['reference_text']} | Whisper: {row['whisper_transcription']}",
                    "downstream_response": row["transcribed_question_gemma_response"],
                    "propagation_outcome": row["propagation_outcome"],
                    "interpretation": row["intent_change_notes"],
                }
            )
    return rows


def write_protocol() -> None:
    text = """# Phase 13 Error Propagation Protocol

Phase 13 evaluates how upstream errors move through the frozen AI-orchestrated fitness-coaching system. It does not change models, thresholds, verified labels, prompts, confidence rules, or Streamlit behaviour.

## Scope

Two propagation chains are evaluated using representative evidence from existing phases:

1. Visual chain: degraded video evidence -> MediaPipe Pose / BlazePose -> movement analysis -> confidence gate -> structured result -> Gemma coaching.
2. Speech chain: user voice question -> Whisper small transcription -> Gemma answer.

## Evidence Sources

- Phase 6 Whisper small benchmark: `dataset/processed/evaluation/stt/whisper_small_results.csv`.
- Phase 8 orchestration traces and app behaviour.
- Phase 11 technical testing evidence.
- Phase 12 robustness/failure results: `dataset/processed/evaluation/robustness/robustness_video_results.csv`.

## Visual Method

Ten representative visual cases are selected from Phase 12. The set includes two clean baseline controls, two occlusion propagated-error cases, two occlusion blocked cases, two partial-body blocked cases, one crop case, and one fast-movement case.

For HIGH-confidence cases, the frozen Gemma 2 2B client is called using structured facts derived from the frozen Phase 12 CSV row. For LOW-confidence cases, normal coaching is not generated; the existing confidence message is recorded as the final user-facing output.

## Speech Method

Ten existing Whisper small recordings are selected from the Phase 6 benchmark. For each recording, Gemma is called twice using the same structured movement result from the clean `vid16_baseline` squat case:

1. reference/intended question;
2. actual Whisper transcription.

The comparison isolates the downstream effect of transcription error.

## Classifications

- `NO_PROPAGATION`: upstream result remains correct enough and final response remains appropriate.
- `STOPPED`: confidence or failure handling prevents normal downstream coaching.
- `PROPAGATED_TRANSPARENTLY`: an error reaches later stages but remains visible or traceable.
- `PROPAGATED_TO_INCORRECT_OUTPUT`: an upstream error reaches Gemma and produces user-facing feedback based on incorrect or changed facts.

## UI Transcription Visibility

The frozen Streamlit code transcribes a voice question inside `ask_voice_question()` and calls `ask_follow_up()` immediately. The transcription is displayed later by `show_question_result()`. Therefore, the current app makes the transcription visible after processing, but not before Gemma generates the answer, and the user cannot edit it before submission in the same interaction.
"""
    (DOCS_DIR / "ERROR_PROPAGATION_PROTOCOL.md").write_text(text, encoding="utf-8")


def write_results(
    visual_rows: list[dict[str, Any]],
    audio_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    strongest_visual = next(row for row in visual_rows if row["case_id"] == "VIS-03")
    strongest_blocked = next(row for row in visual_rows if row["case_id"] == "VIS-05")
    strongest_audio = next(row for row in audio_rows if row["case_id"] == "AUD-09")
    summary_table = "\n".join(
        "| {pipeline} | {total_cases} | {no_propagation} | {stopped} | {propagated_transparently} | {propagated_to_incorrect_output} |".format(**row)
        for row in summary_rows
    )
    text = f"""# Phase 13 Error Propagation Results

Phase 13 used representative evidence from existing frozen evaluations. It did not modify selected models, movement thresholds, confidence thresholds, verified labels, prompts, or Streamlit behaviour.

## Summary

| Pipeline | Cases | No propagation | Stopped | Propagated transparently | Propagated to incorrect output |
|---|---:|---:|---:|---:|---:|
{summary_table}

Table 1. Representative Phase 13 propagation outcomes.

## Visual Error Propagation

The visual subset included 10 cases. Clean controls `VIS-01` and `VIS-02` remained correct enough. Four LOW-confidence cases were stopped before normal Gemma coaching: two occlusion cases and two partial-body cases.

The strongest propagated visual error was `{strongest_visual['case_id']}` (`{strongest_visual['source_video_id']}`, `{strongest_visual['condition']}`). The baseline result was {strongest_visual['baseline_detected_reps']} reps, while the degraded condition produced {strongest_visual['condition_detected_reps']} reps and form `{strongest_visual['condition_form']}` with HIGH confidence. Gemma received these structured facts and generated a response from them. This is an upstream visual/movement error, not an LLM hallucination.

The strongest blocked visual example was `{strongest_blocked['case_id']}` (`{strongest_blocked['source_video_id']}`, `{strongest_blocked['condition']}`). The degraded condition produced {strongest_blocked['condition_detected_reps']} reps and LOW confidence with reasons `{strongest_blocked['confidence_reasons']}`. Normal coaching was blocked.

## Speech Error Propagation

The speech subset included 10 Whisper small recordings. Exact or intent-preserved transcriptions had limited downstream effect. Larger transcription errors changed the language-model input and could change the final answer.

The strongest STT propagated-error example was `{strongest_audio['case_id']}` (`{strongest_audio['audio_id']}`). The intended question was "{strongest_audio['reference_text']}", but Whisper transcribed "{strongest_audio['whisper_transcription']}". Gemma answered the transcription it received, so the final response could become inappropriate for the original spoken question.

## Confidence Gate Finding

The confidence gate successfully stopped some low-quality visual inputs, especially cases with no detected repetitions or poor required-joint/angle availability. However, HIGH confidence did not guarantee semantic correctness. Some degraded videos still produced usable-looking evidence and HIGH confidence while the movement result was wrong.

## Gemma Grounding Finding

In visual propagated-error cases, Gemma generally followed the structured movement facts it received. When those facts were wrong, the final answer could still be wrong overall even though Gemma was grounded. This distinguishes upstream-origin error from LLM-origin hallucination.

## Transcription Visibility Finding

The current Streamlit app displays successful Whisper transcriptions after the voice interaction has been processed. This improves transparency because the user can see what text was used, but it does not prevent propagation during that interaction because the transcription is not shown for correction before Gemma is called.

## Limitations

This is a representative error-propagation study, not a new full benchmark. It uses 10 visual cases and 10 speech cases selected from frozen evidence. Intent and answer-change classifications include researcher judgement and should be reported as qualitative analysis rather than population-level statistics.
"""
    (DOCS_DIR / "ERROR_PROPAGATION_RESULTS.md").write_text(text, encoding="utf-8")


def write_manual_checks() -> None:
    text = """# Phase 13 Manual Checks

No blocking manual checks are required to complete Phase 13.

Optional evidence to capture for the final report:

- A screenshot of the Streamlit voice-question result showing the transcription displayed above the AI Coach response.
- A short note from the researcher confirming that ambiguous STT intent labels were reviewed using the reference text and Whisper output in `audio_error_propagation.csv`.
"""
    (DOCS_DIR / "PHASE13_MANUAL_CHECKS.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    orchestrator = FitnessCoachOrchestrator()

    visual_rows = visual_error_propagation(orchestrator)
    audio_rows = audio_error_propagation(orchestrator)
    summary_rows = build_summary(visual_rows, audio_rows)
    trace_rows = representative_traces(visual_rows, audio_rows)

    write_csv(
        OUT_DIR / "visual_error_propagation.csv",
        visual_rows,
        [
            "case_id",
            "source_video_id",
            "condition",
            "exercise",
            "camera_view",
            "expected_reps",
            "baseline_detected_reps",
            "condition_detected_reps",
            "rep_error",
            "expected_form",
            "baseline_form",
            "condition_form",
            "form_changed",
            "pose_detection_coverage",
            "required_joint_availability",
            "angle_calculability",
            "movement_signal_range",
            "confidence_status",
            "confidence_reasons",
            "coaching_blocked",
            "gemma_status",
            "gemma_input_reps",
            "gemma_input_form",
            "gemma_response",
            "error_origin",
            "propagation_outcome",
            "notes",
        ],
    )
    write_csv(
        OUT_DIR / "audio_error_propagation.csv",
        audio_rows,
        [
            "case_id",
            "audio_id",
            "reference_text",
            "whisper_transcription",
            "wer",
            "intent_preserved",
            "intent_change_notes",
            "structured_result_id",
            "reference_question_gemma_response",
            "transcribed_question_gemma_response",
            "answer_changed",
            "answer_change_type",
            "transcription_visible_in_app",
            "transcription_visible_before_response",
            "error_origin",
            "propagation_outcome",
            "notes",
        ],
    )
    write_csv(
        OUT_DIR / "propagation_summary.csv",
        summary_rows,
        [
            "pipeline",
            "total_cases",
            "no_propagation",
            "stopped",
            "propagated_transparently",
            "propagated_to_incorrect_output",
            "confidence_blocked_error_cases",
            "high_confidence_allowed_incorrect_facts",
            "intent_preserved_or_changed",
            "final_answer_changed",
        ],
    )
    write_csv(
        OUT_DIR / "representative_traces.csv",
        trace_rows,
        [
            "case_id",
            "pipeline",
            "input_id",
            "condition_or_audio",
            "upstream_observation",
            "downstream_response",
            "propagation_outcome",
            "interpretation",
        ],
    )
    write_protocol()
    write_results(visual_rows, audio_rows, summary_rows)
    write_manual_checks()

    print(f"Wrote {OUT_DIR / 'visual_error_propagation.csv'}")
    print(f"Wrote {OUT_DIR / 'audio_error_propagation.csv'}")
    print(f"Wrote {OUT_DIR / 'propagation_summary.csv'}")
    print(f"Wrote {OUT_DIR / 'representative_traces.csv'}")
    print(f"Wrote {DOCS_DIR / 'ERROR_PROPAGATION_PROTOCOL.md'}")
    print(f"Wrote {DOCS_DIR / 'ERROR_PROPAGATION_RESULTS.md'}")
    print(f"Wrote {DOCS_DIR / 'PHASE13_MANUAL_CHECKS.md'}")


if __name__ == "__main__":
    main()
