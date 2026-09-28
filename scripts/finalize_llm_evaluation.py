import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LLM_DIR = ROOT / "dataset" / "processed" / "evaluation" / "llm"
MANUAL_REVIEW = LLM_DIR / "llm_manual_review.csv"
COMPARISON = LLM_DIR / "llm_model_comparison.csv"
SELECTION = LLM_DIR / "llm_model_selection.md"

SCORE_COLS = [
    "manual_factual_grounding_0_2",
    "manual_clarity_0_2",
    "manual_conciseness_0_2",
    "manual_safety_0_2",
]

BINARY_COLS = [
    "manual_invented_issue_yes_no",
    "manual_medical_claim_yes_no",
    "manual_answered_question_yes_no",
]


def load_scored_rows() -> pd.DataFrame:
    df = pd.read_csv(MANUAL_REVIEW)
    missing = {}
    for col in SCORE_COLS + BINARY_COLS:
        empty = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if empty.any():
            missing[col] = int(empty.sum())
    if missing:
        raise SystemExit(f"Manual review is incomplete: {missing}")

    for col in SCORE_COLS:
        df[col] = pd.to_numeric(df[col], errors="raise")
    for col in BINARY_COLS:
        df[col] = df[col].astype(str).str.strip().str.lower()
    df["manual_total_score"] = df[SCORE_COLS].sum(axis=1)
    return df


def yes_count(series: pd.Series) -> int:
    return int((series == "yes").sum())


def no_count(series: pd.Series) -> int:
    return int((series == "no").sum())


def build_comparison(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["model_family", "model"], as_index=False).agg(
        cases=("case_id", "count"),
        mean_latency_seconds=("latency_seconds", "mean"),
        manual_factual_grounding_avg=("manual_factual_grounding_0_2", "mean"),
        manual_clarity_avg=("manual_clarity_0_2", "mean"),
        manual_conciseness_avg=("manual_conciseness_0_2", "mean"),
        manual_safety_avg=("manual_safety_0_2", "mean"),
        manual_total_avg=("manual_total_score", "mean"),
        manual_invented_issue_count=("manual_invented_issue_yes_no", yes_count),
        manual_medical_claim_count=("manual_medical_claim_yes_no", yes_count),
        manual_answered_question_count=("manual_answered_question_yes_no", yes_count),
        failed_responses=("auto_empty_or_failed", yes_count),
        auto_invented_issue_count=("auto_invented_issue", yes_count),
        auto_medical_language_count=("auto_medical_language", yes_count),
        auto_video_claim_count=("auto_video_claim", yes_count),
    )
    return grouped.round(4)


def response_for(df: pd.DataFrame, model: str, case_id: str) -> str:
    rows = df[(df["model"] == model) & (df["case_id"] == case_id)]
    if rows.empty:
        return ""
    return str(rows.iloc[0]["response"]).replace("\r\n", "\n").strip()


def write_selection(df: pd.DataFrame, comparison: pd.DataFrame) -> None:
    selected = "gemma2:2b"
    llama = comparison[comparison["model"] == "llama3.2:3b"].iloc[0]
    gemma = comparison[comparison["model"] == selected].iloc[0]
    mistral = comparison[comparison["model"] == "mistral:7b-instruct"].iloc[0]

    table_header = (
        "| Model | Total avg /8 | Grounding | Clarity | Conciseness | Safety | "
        "Invented issues | Medical claims | Answered | Latency |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    table_rows = []
    for _, row in comparison.sort_values("manual_total_avg", ascending=False).iterrows():
        table_rows.append(
            f"| `{row['model']}` | {row['manual_total_avg']:.4f} | "
            f"{row['manual_factual_grounding_avg']:.4f} | {row['manual_clarity_avg']:.4f} | "
            f"{row['manual_conciseness_avg']:.4f} | {row['manual_safety_avg']:.4f} | "
            f"{int(row['manual_invented_issue_count'])} | {int(row['manual_medical_claim_count'])} | "
            f"{int(row['manual_answered_question_count'])}/20 | {row['mean_latency_seconds']:.4f}s |"
        )

    lines = [
        "# LLM Model Selection",
        "",
        "Selected model: `gemma2:2b`",
        "",
        "Ollama is the local runtime. The selected pretrained language model is `gemma2:2b`.",
        "",
        "## Selection Basis",
        "",
        "The final LLM was selected using the Phase 7 priority order: factual grounding and no invented issues first, safety second, ability to answer the question third, clarity/conciseness fourth, and latency/resource use last.",
        "",
        "Gemma was selected because it achieved the highest manual total score, the highest factual-grounding average, no manually identified invented issues, no manually identified medical claims, and the lowest mean latency. The selection is task-specific: Gemma is not claimed to be universally superior, only the best fit for grounded coaching explanations in this prototype.",
        "",
        "## Manual Comparison",
        "",
        table_header + "\n".join(table_rows),
        "",
        "## Why The Other Models Were Rejected",
        "",
        f"- `llama3.2:3b` answered all 20 questions and had safe responses overall, but it had {int(llama['manual_invented_issue_count'])} manually identified invented issues and a lower factual-grounding average ({llama['manual_factual_grounding_avg']:.4f}) than Gemma.",
        f"- `mistral:7b-instruct` answered all 20 questions but had the lowest manual total score ({mistral['manual_total_avg']:.4f}), one manually identified medical-claim violation, and much higher latency ({mistral['mean_latency_seconds']:.4f}s).",
        "",
        "## Representative Failure Examples",
        "",
        "### Llama invented issue",
        "",
        "Case `llm_018` asked what to focus on when bicep-curl form was correct. Llama introduced unsupported elbow-angle advice and suggested the elbow angle was outside a recommended range, even though that issue was not supplied by the movement-analysis facts.",
        "",
        "> " + response_for(df, "llama3.2:3b", "llm_018").replace("\n", "\n> "),
        "",
        "### Mistral safety/medical overreach",
        "",
        "Case `llm_013` asked about knee pain. Mistral correctly refused diagnosis, but it also suggested the detected shallow depth might contribute to stress on the joints. This was treated as an overreach because the supplied movement facts did not establish a pain cause.",
        "",
        "> " + response_for(df, "mistral:7b-instruct", "llm_013").replace("\n", "\n> "),
        "",
        "### Gemma limitation",
        "",
        "Case `llm_015` asked what to do when the system could not analyse a video. Gemma explained the failure reason but did not give enough practical re-recording guidance, so it was marked as not fully answering the question.",
        "",
        "> " + response_for(df, "gemma2:2b", "llm_015").replace("\n", "\n> "),
        "",
        "## Limitations",
        "",
        "- The test set contains 20 fixed structured cases, not live end-to-end user conversations.",
        "- Manual scoring was completed by the project researcher, so some judgement is subjective.",
        "- The automatic checks only identify obvious violations and do not replace manual review.",
        "- Results may change with longer questions, different prompts, other Ollama model tags, or different hardware.",
        "",
        "## Next Step",
        "",
        "Phase 8 - Final Orchestration: connect the selected pose model, movement-analysis layer, confidence gate, Whisper small, Gemma, and text-to-speech into the final user flow.",
    ]

    SELECTION.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = load_scored_rows()
    comparison = build_comparison(df)
    comparison.to_csv(COMPARISON, index=False, quoting=csv.QUOTE_MINIMAL)
    write_selection(df, comparison)
    print(f"Wrote {COMPARISON}")
    print(f"Wrote {SELECTION}")


if __name__ == "__main__":
    main()
