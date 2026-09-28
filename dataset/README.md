# Dataset Directory

This public repository does not include raw volunteer videos, derived repetition clips, private audio recordings, consent forms, or any other personally identifying media.

## What is included

- `llm/`: the system prompt and safe LLM benchmark cases.
- `stt/`: the speech-to-text manifest only. The original audio files are excluded.
- `tts/`: representative text-to-speech test cases.
- `processed/evaluation/`: anonymised evaluation outputs used as evidence in the project report.
- `processed/splits/`: safe split notes and leave-one-participant-out planning evidence where no private source filenames are exposed.

## What is excluded

- `dataset/raw/`: original volunteer videos.
- `dataset/clips/`: derived single-repetition clips.
- `dataset/augmented/`: augmented training media.
- `dataset/external/`: local source dataset staging area.
- `dataset/processed/robustness/videos/`: generated robustness videos.
- `dataset/stt/audio/`: recorded speech questions.
- generated TTS/audio outputs.

## Label meaning

The final evaluated dataset used two exercises:

- `squat/correct`: full-depth squat.
- `squat/shallow`: incomplete-depth squat.
- `bicep_curl/correct`: full-range bicep curl.
- `bicep_curl/half_range`: incomplete-range bicep curl.

The labels intentionally avoid unsafe exercise demonstrations. Incorrect examples represent incomplete range of motion, not harmful movement.

## Privacy

Participants were anonymised using participant IDs. Public files should not contain volunteer names, original phone-export filenames, local machine paths, or private media.
