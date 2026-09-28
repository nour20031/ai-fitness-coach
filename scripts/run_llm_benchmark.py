import csv
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib import request, error


ROOT = Path(__file__).resolve().parents[1]
TEST_CASES_PATH = ROOT / "dataset" / "llm" / "llm_test_cases.json"
SYSTEM_PROMPT_PATH = ROOT / "dataset" / "llm" / "system_prompt.txt"
OUTPUT_DIR = ROOT / "dataset" / "processed" / "evaluation" / "llm"

CANDIDATES = [
    {
        "family": "llama",
        "model": "llama3.2:3b",
        "estimated_size": "about 2.0 GB",
        "pull_command": "ollama pull llama3.2:3b",
    },
    {
        "family": "gemma",
        "model": "gemma2:2b",
        "estimated_size": "about 1.6 GB",
        "pull_command": "ollama pull gemma2:2b",
    },
    {
        "family": "mistral",
        "model": "mistral:7b-instruct",
        "estimated_size": "about 4.1 GB",
        "pull_command": "ollama pull mistral:7b-instruct",
    },
]

GENERATION_OPTIONS = {
    "temperature": 0,
    "num_predict": 220,
}

KNOWN_ISSUES = {
    "shallow_depth",
    "half_range_motion",
    "body_swinging",
    "NO_REPS_DETECTED",
    "INSUFFICIENT_MOVEMENT_SIGNAL",
}

ISSUE_TERMS = {
    "knees cave": "knee_valgus",
    "knee cave": "knee_valgus",
    "knees collapsed": "knee_valgus",
    "back rounded": "rounded_back",
    "rounded back": "rounded_back",
    "torso leaned": "torso_lean",
    "leaned too far": "torso_lean",
    "heel": "heel_lift",
    "elbow flared": "elbow_flare",
    "wrist bent": "wrist_bend",
}

MEDICAL_TERMS = [
    "diagnosis",
    "diagnose",
    "injury",
    "torn",
    "tear",
    "sprain",
    "strain",
    "arthritis",
    "tendonitis",
    "meniscus",
    "ligament",
    "treatment",
    "rehab",
]

VIDEO_CLAIMS = [
    "i can see",
    "i saw",
    "from your video",
    "watching your video",
    "in the video",
    "your video shows",
]


def run_ollama_list() -> list[str]:
    try:
        with request.urlopen("http://127.0.0.1:11434/api/tags", timeout=10) as response:
            result = json.loads(response.read().decode("utf-8"))
        return [item.get("model") or item.get("name") for item in result.get("models", [])]
    except Exception:
        pass

    if not shutil.which("ollama"):
        return []

    completed = subprocess.run(
        ["ollama", "list"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        return []

    models = []
    for line in completed.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models


def stop_for_missing_models(installed: list[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [item for item in CANDIDATES if item["model"] not in installed]
    lines = [
        "# LLM Benchmark Not Run",
        "",
        "Ollama or the required candidate models are not available locally.",
        "",
        "Recommended candidate models:",
        "",
    ]
    for item in CANDIDATES:
        status = "installed" if item["model"] in installed else "missing"
        lines.append(
            f"- `{item['model']}` ({item['family']}, {item['estimated_size']}) - {status}"
        )
    lines.extend(["", "Pull commands:", ""])
    for item in CANDIDATES:
        lines.append(f"```powershell\n{item['pull_command']}\n```")
    lines.extend(
        [
            "",
            "After installing Ollama and pulling the models, rerun:",
            "",
            "```powershell",
            r".\venv\Scripts\python.exe scripts\run_llm_benchmark.py",
            "```",
        ]
    )
    (OUTPUT_DIR / "llm_benchmark_not_run.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print("\n".join(lines))
    raise SystemExit(2)


def build_prompt(system_prompt: str, test_case: dict) -> str:
    facts = json.dumps(test_case["structured_facts"], indent=2, sort_keys=True)
    return (
        f"{system_prompt}\n\n"
        "Structured movement-analysis facts:\n"
        f"{facts}\n\n"
        f"User question: {test_case['user_question']}\n\n"
        "Answer:"
    )


def call_ollama(model: str, prompt: str) -> tuple[str, float, str]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": GENERATION_OPTIONS,
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with request.urlopen(req, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
        latency = time.perf_counter() - start
        return result.get("response", "").strip(), latency, ""
    except error.URLError as exc:
        latency = time.perf_counter() - start
        return "", latency, f"OLLAMA_REQUEST_ERROR: {exc}"


def automatic_checks(response: str, test_case: dict) -> dict:
    text = response.lower()
    supplied = set(test_case["structured_facts"].get("detected_issues", []))
    supplied.update(test_case["structured_facts"].get("confidence_reasons", []))
    invented_terms = []
    for phrase, label in ISSUE_TERMS.items():
        if phrase in text and label not in supplied and label not in KNOWN_ISSUES:
            invented_terms.append(label)

    supplied_issue_words = " ".join(supplied).lower()
    if "shallow" in text and "shallow" not in supplied_issue_words and test_case["structured_facts"].get("form_label") != "shallow":
        invented_terms.append("shallow_depth")
    if "half range" in text and "half_range_motion" not in supplied and test_case["structured_facts"].get("form_label") != "half_range":
        invented_terms.append("half_range_motion")
    if "swing" in text and "body_swinging" not in supplied:
        invented_terms.append("body_swinging")

    medical_claim = any(term in text for term in MEDICAL_TERMS)
    video_claim = any(phrase in text for phrase in VIDEO_CLAIMS)
    failed = not response.strip()

    return {
        "auto_invented_issue": "yes" if invented_terms else "no",
        "auto_invented_issue_terms": ";".join(sorted(set(invented_terms))),
        "auto_medical_language": "yes" if medical_claim else "no",
        "auto_video_claim": "yes" if video_claim else "no",
        "auto_empty_or_failed": "yes" if failed else "no",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        latencies = [float(row["latency_seconds"]) for row in model_rows]
        summary.append(
            {
                "model": model,
                "cases": len(model_rows),
                "mean_latency_seconds": round(sum(latencies) / len(latencies), 4),
                "failed_responses": sum(row["auto_empty_or_failed"] == "yes" for row in model_rows),
                "auto_invented_issue_count": sum(row["auto_invented_issue"] == "yes" for row in model_rows),
                "auto_medical_language_count": sum(row["auto_medical_language"] == "yes" for row in model_rows),
                "auto_video_claim_count": sum(row["auto_video_claim"] == "yes" for row in model_rows),
            }
        )
    return summary


def main() -> None:
    installed = run_ollama_list()
    required_models = [item["model"] for item in CANDIDATES]
    if len(installed) == 0 or any(model not in installed for model in required_models):
        stop_for_missing_models(installed)

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    test_cases = json.loads(TEST_CASES_PATH.read_text(encoding="utf-8"))
    all_rows = []

    for candidate in CANDIDATES:
        model_rows = []
        for case in test_cases:
            prompt = build_prompt(system_prompt, case)
            response, latency, error_text = call_ollama(candidate["model"], prompt)
            checks = automatic_checks(response, case)
            row = {
                "model_family": candidate["family"],
                "model": candidate["model"],
                "case_id": case["case_id"],
                "category": case["category"],
                "structured_facts": json.dumps(case["structured_facts"], sort_keys=True),
                "user_question": case["user_question"],
                "response": response,
                "latency_seconds": round(latency, 4),
                "error": error_text,
                **checks,
                "manual_factual_grounding_0_2": "",
                "manual_clarity_0_2": "",
                "manual_conciseness_0_2": "",
                "manual_safety_0_2": "",
                "manual_invented_issue_yes_no": "",
                "manual_medical_claim_yes_no": "",
                "manual_answered_question_yes_no": "",
                "manual_notes": "",
            }
            model_rows.append(row)
            all_rows.append(row)
        write_csv(OUTPUT_DIR / f"{candidate['family']}_results.csv", model_rows)

    write_csv(OUTPUT_DIR / "llm_manual_review.csv", all_rows)
    write_csv(OUTPUT_DIR / "llm_model_comparison.csv", summarize(all_rows))
    (OUTPUT_DIR / "llm_model_selection.md").write_text(
        "# LLM Model Selection\n\n"
        "Benchmark responses and automatic checks were generated. Complete the manual scoring fields in "
        "`llm_manual_review.csv` before selecting the final language model.\n",
        encoding="utf-8",
    )
    print(f"Wrote LLM benchmark outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
