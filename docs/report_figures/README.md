# Chapter 4 Report Figures

This folder contains report figures prepared for Chapter 4 (Implementation).

## Created diagrams

| Figure | File | Source | Caption |
| --- | --- | --- | --- |
| 4.1 | `Figure_4_1_Final_System_Architecture.png` | Created from `app/app.py`, `src/orchestration/fitness_coach_orchestrator.py`, `docs/ORCHESTRATION_ARCHITECTURE.md`, and project structure | Figure 4.1. Final software architecture of the AI-Orchestrated Fitness Coach. |
| 4.2 | `Figure_4_2_Dataset_Preparation_Pipeline.png` | Created from dataset folders, manifest files, `docs/DATASET_CARD.md`, and `docs/EXPERIMENT_LOG.md` | Figure 4.2. Dataset preparation pipeline from original volunteer recordings to evaluation data. |
| 4.4 | `Figure_4_4_Movement_State_Machines.png` | Created from `src/pose/squat_analyzer.py`, `src/pose/bicep_curl_analyzer.py`, and `docs/MOVEMENT_ANALYSIS_RULES.md` | Figure 4.4. State-machine logic used to count squat and bicep-curl repetitions before form classification. |
| 4.5 | `Figure_4_5_Confidence_Validation_Gate.png` | Created from `src/pose/confidence_validator.py` and `docs/CONFIDENCE_GATE.md` | Figure 4.5. Confidence-validation gate used before user-facing coaching feedback. |
| 4.6 | `Figure_4_6_Three_Model_Orchestration.png` | Created from `src/orchestration/fitness_coach_orchestrator.py` and `docs/ORCHESTRATION_ARCHITECTURE.md` | Figure 4.6. Data flow between the three selected pretrained models and supporting processing layers. |

Each created diagram also has an editable `.svg` and a Mermaid `.mmd` source file.

## Screenshot placeholders

No genuine app screenshots suitable for Figures 4.3, 4.7, or 4.8 were found in the project image files. To avoid fake screenshots, capture instructions were created instead:

- `Figure_4_3_CAPTURE_INSTRUCTIONS.txt`
- `Figure_4_7_CAPTURE_INSTRUCTIONS.txt`
- `Figure_4_8_CAPTURE_INSTRUCTIONS.txt`

## Numeric consistency

Thresholds shown in these figures were checked against the final code:

- Squat down/up trigger: 165 degrees.
- Squat shallow-depth threshold: 115 degrees.
- Squat minimum down frames: 3.
- Bicep-curl extended/down trigger: 125 degrees.
- Bicep-curl curled/up trigger: 115 degrees.
- Bicep-curl half-range threshold: 55 degrees.
- Bicep-curl stable frames: 4.
- Bicep-curl cooldown: 8 frames.
- Landmark visibility threshold: 0.5.
- Confidence minimum processed frames: 15.
- Confidence minimum pose detection rate: 0.70.
- Confidence minimum required-joint availability: 0.65.
- Confidence minimum angle calculability: 0.65.
- Confidence maximum dropout: 0.25.
- Confidence squat movement signal: 10 degrees.
- Confidence bicep-curl movement signal: 20 degrees.

No application implementation files were modified to create these figures.
