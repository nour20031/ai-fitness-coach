# LLM Evaluation Results

Phase 7 compared three local pretrained language models for grounded coaching feedback. Ollama was used only as the runtime; the selected pretrained language model is the third AI model in the final orchestration.

## Environment Check

| Item | Result |
| --- | --- |
| Ollama version | 0.34.4 |
| CPU | Detected as `AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD` |
| Logical processors | 16 |
| Available memory during check | about 3167 MB |
| GPU/VRAM | Could not be read because the Windows hardware query was blocked |
| Ollama server | Available at `http://127.0.0.1:11434` |

Table 1. Local LLM benchmark environment.

## Candidate Models

| Family | Ollama model tag | Approximate local size | Parameter size / quantization |
| --- | --- | ---: | --- |
| Llama | `llama3.2:3b` | about 2.0 GB | 3.2B / Q4_K_M |
| Gemma | `gemma2:2b` | about 1.6 GB | 2.6B / Q4_0 |
| Mistral | `mistral:7b-instruct` | about 4.1 GB | 7.2B / Q4_K_M |

Table 2. Candidate language models.

## Benchmark Setup

All models received the same:

- 20 structured movement-analysis test cases from `dataset/llm/llm_test_cases.json`
- fixed system prompt from `dataset/llm/system_prompt.txt`
- structured movement facts and user question format
- generation settings: temperature `0`, `num_predict=220`
- local Ollama runtime and CPU environment

The LLM was not allowed to analyse images or video directly. It only explained structured facts already produced by the pose-estimation, movement-analysis, and confidence-gate stages.

## Manual Scoring Rubric

Each response was manually scored by the project researcher.

| Metric | Scale |
| --- | --- |
| Factual grounding | 0-2 |
| Clarity | 0-2 |
| Conciseness | 0-2 |
| Safety | 0-2 |
| Invented issue | yes/no |
| Medical claim | yes/no |
| Answered question | yes/no |

Table 3. Manual LLM evaluation rubric.

Automatic checks were also used to flag obvious invented issue terms, direct video-inspection claims, medical language, and failed/empty responses. These checks supported the manual review but did not replace it.

## Main Results

| Model | Total avg /8 | Grounding | Clarity | Conciseness | Safety | Invented issues | Medical claims | Answered | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gemma2:2b` | 7.4000 | 1.5000 | 2.0000 | 2.0000 | 1.9000 | 0 | 0 | 19/20 | 0.7288s |
| `llama3.2:3b` | 6.7500 | 1.2500 | 1.9500 | 1.6500 | 1.9000 | 3 | 0 | 20/20 | 1.3632s |
| `mistral:7b-instruct` | 5.9000 | 1.0000 | 2.0000 | 1.1500 | 1.7500 | 0 | 1 | 20/20 | 5.8849s |

Table 4. LLM comparison using manual scores and latency.

## Selected Model

Selected model: `gemma2:2b`.

Gemma was selected using the Phase 7 priority order: factual grounding and no invented issues first, safety second, ability to answer the question third, clarity and conciseness fourth, and latency/resource use last.

Gemma achieved the highest overall manual score, the highest factual-grounding average, no manually identified invented issues, no manually identified medical claims, and the lowest mean latency. This is a task-specific selection for grounded coaching explanations, not evidence that Gemma is universally superior.

## Failure Analysis

`llama3.2:3b` produced three manually identified invented issues. For example, in case `llm_018`, it introduced unsupported elbow-angle advice and suggested the elbow angle was outside a recommended range even though that issue was not supplied by the movement-analysis facts.

`mistral:7b-instruct` produced one manually identified medical-claim violation. In case `llm_013`, it correctly refused to diagnose knee pain but also suggested the detected shallow squat depth might contribute to stress on the joints. This was treated as overreach because the supplied facts did not establish a pain cause.

`gemma2:2b` had one answer that did not fully answer the question. In case `llm_015`, it explained why the video could not be analysed but did not provide enough practical re-recording guidance.

## Limitations

- The test set contains 20 fixed structured cases, not live end-to-end user conversations.
- Manual scoring was completed by the project researcher, so some judgement is subjective.
- The automatic checks detect only obvious violations and do not replace manual review.
- Results may change with longer user questions, different prompts, other Ollama model tags, or different hardware.
- The selected model still depends on upstream movement-analysis facts; it does not correct pose or rep-counting errors.

## Evidence Files

- `dataset/processed/evaluation/llm/llm_manual_review.csv`
- `dataset/processed/evaluation/llm/llm_model_comparison.csv`
- `dataset/processed/evaluation/llm/llm_model_selection.md`
- `dataset/processed/evaluation/llm/llama_results.csv`
- `dataset/processed/evaluation/llm/gemma_results.csv`
- `dataset/processed/evaluation/llm/mistral_results.csv`

## Next Step

Phase 8 has now connected MediaPipe Pose / BlazePose, the movement-analysis layer, the confidence gate, Whisper small, and Gemma into the text coaching flow. The next phase is Phase 9 - Add Text-to-Speech.
