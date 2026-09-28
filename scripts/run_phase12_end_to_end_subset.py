import csv
import sys
from pathlib import Path

from dataset_common import ROOT_DIR, read_csv, write_csv

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.orchestration import FitnessCoachOrchestrator


ROBUSTNESS_DIR = ROOT_DIR / "dataset" / "processed" / "evaluation" / "robustness"
VIDEO_RESULTS = ROBUSTNESS_DIR / "robustness_video_results.csv"
OUTPUT_CSV = ROBUSTNESS_DIR / "robustness_end_to_end_subset.csv"
OUTPUT_MD = ROBUSTNESS_DIR / "robustness_end_to_end_subset.md"

SCENARIOS = [
    ("P12-E2E-001", "baseline_success", "vid16_baseline"),
    ("P12-E2E-002", "darker_success", "vid16_darker"),
    ("P12-E2E-003", "crop_high_confidence_rep_error", "vid01_crop"),
    ("P12-E2E-004", "occlusion_low_confidence_block", "vid13_occlusion"),
    ("P12-E2E-005", "fast_movement_timing_stress", "vid16_fast_movement"),
]

FIELDS = [
    "scenario_id",
    "scenario",
    "robustness_id",
    "source_video_id",
    "condition",
    "exercise",
    "verified_label",
    "status",
    "coaching_blocked",
    "detected_reps",
    "predicted_form_label",
    "confidence_status",
    "confidence_reasons",
    "gemma_status",
    "final_response",
    "failure_stage",
    "expected_behaviour",
    "passed",
]


def row_by_id(rows, robustness_id):
    for row in rows:
        if row["robustness_id"] == robustness_id:
            return row
    raise KeyError(f"Missing robustness row: {robustness_id}")


def expected_behaviour(row):
    if row["confidence_status"] == "LOW":
        return "LOW confidence should block normal Gemma coaching and return re-record guidance."
    return "HIGH confidence should allow Gemma to produce grounded feedback from the transformed analysis facts."


def main():
    rows = read_csv(VIDEO_RESULTS)
    lookup = {row["robustness_id"]: row for row in rows}
    orchestrator = FitnessCoachOrchestrator()
    output_rows = []

    for scenario_id, scenario, robustness_id in SCENARIOS:
        row = lookup[robustness_id]
        result = orchestrator.run(
            row["analysis_path"],
            row["exercise"],
            user_question="Give concise coaching feedback based only on this transformed robustness analysis.",
            enable_tts=False,
            preview_interval=0,
            require_ready=False,
        )
        structured = result.get("structured_result") or {}
        gemma = result.get("gemma") or {}
        confidence = structured.get("confidence_status", "")
        should_block = row["confidence_status"] == "LOW"
        passed = (
            result.get("status") == "blocked" and result.get("coaching_blocked") is True
            if should_block
            else result.get("status") == "success" and gemma.get("status") == "success"
        )
        output_rows.append({
            "scenario_id": scenario_id,
            "scenario": scenario,
            "robustness_id": robustness_id,
            "source_video_id": row["source_video_id"],
            "condition": row["condition"],
            "exercise": row["exercise"],
            "verified_label": row["verified_label"],
            "status": result.get("status", ""),
            "coaching_blocked": result.get("coaching_blocked", ""),
            "detected_reps": structured.get("rep_count", ""),
            "predicted_form_label": structured.get("form_label", ""),
            "confidence_status": confidence,
            "confidence_reasons": ";".join(structured.get("confidence_reasons", [])),
            "gemma_status": gemma.get("status", ""),
            "final_response": result.get("final_response", ""),
            "failure_stage": result.get("failure_stage", ""),
            "expected_behaviour": expected_behaviour(row),
            "passed": passed,
        })

    write_csv(OUTPUT_CSV, output_rows, FIELDS)
    lines = [
        "# Phase 12 Representative End-to-End Robustness Checks",
        "",
        "This small subset checks that HIGH-confidence transformed inputs receive grounded Gemma feedback and LOW-confidence transformed inputs block normal coaching. It is not the full Phase 13 error-propagation study.",
        "",
        "| Scenario | Robustness ID | Condition | Status | Confidence | Gemma | Passed |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in output_rows:
        lines.append(
            f"| {row['scenario']} | {row['robustness_id']} | {row['condition']} | "
            f"{row['status']} | {row['confidence_status']} | {row['gemma_status']} | {row['passed']} |"
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_MD}")
    if not all(str(row["passed"]).lower() == "true" for row in output_rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
