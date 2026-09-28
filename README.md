# AI-Orchestrated Fitness Coach

Final-year Artificial Intelligence project for the University of London template:

**CM3020 Artificial Intelligence - Project Idea 1: Orchestrating AI models to achieve a goal**

This project is a Streamlit prototype that analyses short exercise videos and produces grounded coaching feedback. It combines visual pose estimation, speech-to-text, language-model feedback, and optional text-to-speech output.

The public repository contains source code, documentation, anonymised evaluation evidence, and reproducible test scripts. It intentionally does **not** include private volunteer videos, consent forms, raw audio recordings, or downloaded model binaries.

## Final System

The final prototype supports two exercise families:

- Squat: correct/deep form vs shallow/incomplete depth.
- Bicep curl: correct/full range vs half/incomplete range of motion.

Core pipeline:

```text
Exercise video
-> MediaPipe Pose / BlazePose
-> pose landmarks and visibility
-> rule-based movement analysis
-> repetitions, form label, measured features
-> confidence validation gate
-> structured movement result
-> gemma2:2b through Ollama
-> grounded coaching feedback

Voice question
-> Whisper small
-> transcribed question
-> gemma2:2b with the structured movement result
-> grounded answer

Optional output
-> Windows SAPI text-to-speech
-> spoken feedback
```

The deterministic movement-analysis rules and confidence gate are part of the system logic. They are not counted as pretrained models.

## Selected Pretrained Models

| Role | Selected model | Purpose |
| --- | --- | --- |
| Visual pose estimation | MediaPipe Pose / BlazePose | Extract body landmarks from video frames. |
| Speech-to-text | Whisper small | Transcribe spoken user questions. |
| Language feedback | gemma2:2b via Ollama | Generate grounded coaching text from structured facts. |
| Optional presentation | Windows SAPI | Speak the final text response locally on Windows. |

## Repository Structure

```text
app/                         Streamlit application
src/pose/                    MediaPipe pose detection, squat/curl analyzers, confidence validator
src/orchestration/           Three-model orchestration layer
src/audio/                   Windows SAPI text-to-speech wrapper
scripts/                     Dataset, evaluation, benchmarking, and testing scripts
tests/                       Contract tests for final app/session behavior
dataset/llm/                 LLM prompt and safe benchmark cases
dataset/stt/                 STT manifest only; raw audio is excluded
dataset/tts/                 TTS test cases
dataset/processed/evaluation Safe anonymised metrics and report evidence
docs/                        Design, implementation, evaluation, and decision documentation
models/                      Download instructions only; model binaries are excluded
```

## Privacy Notice

The original dataset was recorded with consenting adult volunteers. Raw videos, derived clips, robustness videos, audio recordings, and consent materials are excluded from this public repository because they contain personal data or identifiable media.

The included dataset files are intended to be non-identifying evidence such as aggregate metrics, anonymised participant IDs, benchmark summaries, model-comparison results, and report figures. If you collect your own dataset, store private media locally and do not commit it.

## Setup

This project was developed on Windows with Python and Streamlit.

1. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install Python dependencies:

```powershell
pip install -r requirements.txt
```

3. Install and start Ollama, then download the selected language model:

```powershell
ollama pull gemma2:2b
```

Ollama must be running when using LLM coaching. If `ollama serve` reports that port `11434` is already in use, Ollama is already running.

4. Whisper small is downloaded automatically by the `openai-whisper` package on first use into `models/whisper/`. The downloaded `.pt` file is ignored by Git.

5. Whisper requires FFmpeg to read many audio formats. Install FFmpeg separately if audio transcription fails.

## Run the App

From the repository root:

```powershell
streamlit run app\app.py
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

Use the upload path for the most stable demonstration. Upload an exercise video, choose the exercise, run analysis, and then ask a text or voice question after analysis is complete.

## Run Tests

Run the final contract tests:

```powershell
python -m unittest tests.test_phase11_contracts
```

Compile-check the public source files:

```powershell
python -m compileall app src scripts tests
```

## Evaluation Scripts

The repository keeps the scripts used during the final evaluation, including pose-model comparison, movement analysis, confidence validation, STT, LLM, orchestration, TTS, Streamlit smoke/session checks, technical testing, robustness testing, and error propagation.

Some scripts require the private local dataset and will not fully rerun from this public repository alone. The public repository keeps the anonymised output evidence under `dataset/processed/evaluation/`.

Examples:

```powershell
python scripts\run_pose_model_benchmark.py
python scripts\run_movement_analysis_evaluation.py
python scripts\run_confidence_gate_evaluation.py
python scripts\run_whisper_benchmark.py
python scripts\run_llm_benchmark.py
python scripts\run_orchestration_test.py
python scripts\run_tts_evaluation.py
python scripts\run_phase11_technical_tests.py
python scripts\run_phase12_robustness_tests.py
python scripts\run_phase13_error_propagation.py
```

## Main Evaluation Evidence

The final evaluation evidence is documented in `docs/` and supported by CSV/JSON files in `dataset/processed/evaluation/`.

Important results include:

- Pose-model comparison between MediaPipe Pose / BlazePose and MoveNet.
- Final movement-analysis results for squat and bicep curl.
- Confidence-gate behavior.
- Whisper tiny/base/small comparison, with Whisper small selected using an accuracy-prioritised strategy.
- LLM comparison, with gemma2:2b selected for grounded coaching feedback.
- Three-model orchestration evidence.
- TTS integration evidence.
- Phase 11 technical testing: 64/64 technical test entries passed.
- Robustness and failure testing.
- Combined error-propagation analysis.

## Limitations

The prototype is not a medical diagnostic tool and does not replace a personal trainer, physiotherapist, or clinician. It is designed for healthy adult users and gives basic form feedback from visible movement evidence.

Known limitations include limited participant diversity, sensitivity to camera angle and joint visibility, imperfect repetition counting under occlusion or partial-body framing, and possible propagation of upstream pose or transcription errors into the final language response.
