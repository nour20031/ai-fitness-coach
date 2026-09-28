# Model Files

Large model binaries are intentionally excluded from the public repository.

## Language model

The selected language model is `gemma2:2b` through Ollama.

Install Ollama, then run:

```powershell
ollama pull gemma2:2b
```

Ollama is the runtime. The third mandatory pretrained model is `gemma2:2b`.

## Speech-to-text

The selected STT model is Whisper small. The `openai-whisper` package downloads it automatically on first use into:

```text
models/whisper/
```

That directory is ignored by Git because it contains large `.pt` model files.

## Pose models

The final application uses MediaPipe Pose / BlazePose through the `mediapipe` Python package.

MoveNet was used only during the pose-model comparison phase and is not required to run the final Streamlit app. Any `.tflite` files are excluded from the public repository.
