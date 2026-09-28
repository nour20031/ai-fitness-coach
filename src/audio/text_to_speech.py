import base64
import subprocess
import time
import uuid
import wave
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TTS_OUTPUT_DIR = ROOT_DIR / "dataset" / "processed" / "evaluation" / "tts" / "audio"


@dataclass
class TextToSpeechResult:
    status: str
    audio_path: str = ""
    latency_seconds: float = 0.0
    engine: str = "Windows SAPI"
    voice: str = ""
    rate: int = 0
    audio_format: str = "wav"
    audio_duration_seconds: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class WindowsSapiTextToSpeech:
    """Local/offline Windows SAPI text-to-speech wrapper."""

    def __init__(
        self,
        output_dir: str | Path = DEFAULT_TTS_OUTPUT_DIR,
        voice_name: str | None = None,
        rate: int = 0,
    ):
        self.output_dir = Path(output_dir)
        self.voice_name = voice_name or ""
        self.rate = rate

    def synthesize(
        self,
        text: str,
        output_name: str | None = None,
        output_dir: str | Path | None = None,
    ) -> TextToSpeechResult:
        text = (text or "").strip()
        if not text:
            return TextToSpeechResult(
                status="failed",
                voice=self.voice_name,
                rate=self.rate,
                error="TTS input text is empty.",
            )

        target_dir = Path(output_dir) if output_dir else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_name = output_name or f"tts_{uuid.uuid4().hex}.wav"
        if not safe_name.lower().endswith(".wav"):
            safe_name = f"{safe_name}.wav"
        output_path = target_dir / safe_name

        text_path = target_dir / f"tts_input_{uuid.uuid4().hex}.txt"
        text_path.write_text(text, encoding="utf-8")

        start = time.perf_counter()
        encoded_command = base64.b64encode(
            self._powershell_command(
                text_path=text_path,
                output_path=output_path,
            ).encode("utf-16le")
        ).decode("ascii")
        cmd = [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-EncodedCommand",
            encoded_command,
        ]

        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
        except Exception as exc:
            return TextToSpeechResult(
                status="failed",
                latency_seconds=round(time.perf_counter() - start, 4),
                voice=self.voice_name,
                rate=self.rate,
                error=f"Windows SAPI TTS failed to run: {exc}",
            )
        finally:
            try:
                text_path.unlink(missing_ok=True)
            except OSError:
                pass

        latency = round(time.perf_counter() - start, 4)
        if output_path.exists() and output_path.stat().st_size > 0:
            return TextToSpeechResult(
                status="success",
                audio_path=str(output_path),
                latency_seconds=latency,
                voice=self.voice_name or self._default_voice_from_stdout(completed.stdout),
                rate=self.rate,
                audio_duration_seconds=self._wav_duration(output_path),
            )

        if completed.returncode != 0:
            return TextToSpeechResult(
                status="failed",
                latency_seconds=latency,
                voice=self.voice_name,
                rate=self.rate,
                error=(completed.stderr or completed.stdout or "Windows SAPI returned a non-zero exit code.").strip(),
            )

        if not output_path.exists() or output_path.stat().st_size == 0:
            return TextToSpeechResult(
                status="failed",
                latency_seconds=latency,
                voice=self.voice_name,
                rate=self.rate,
                error="Windows SAPI did not create a readable audio file.",
            )

        return TextToSpeechResult(
            status="success",
            audio_path=str(output_path),
            latency_seconds=latency,
            voice=self.voice_name or self._default_voice_from_stdout(completed.stdout),
            rate=self.rate,
            audio_duration_seconds=self._wav_duration(output_path),
        )

    @staticmethod
    def _ps_quote(value: str | Path) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    def _powershell_command(self, text_path: Path, output_path: Path) -> str:
        text_literal = self._ps_quote(text_path)
        output_literal = self._ps_quote(output_path)
        voice_literal = self._ps_quote(self.voice_name)
        return r'''
$TextPath = __TEXT_PATH__
$OutputPath = __OUTPUT_PATH__
$Rate = __RATE__
$VoiceName = __VOICE_NAME__
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$speaker = New-Object -ComObject SAPI.SpVoice
$stream = New-Object -ComObject SAPI.SpFileStream
try {
    if ($VoiceName -ne "") {
        foreach ($voice in $speaker.GetVoices()) {
            if ($voice.GetDescription() -eq $VoiceName) {
                $speaker.Voice = $voice
                break
            }
        }
    }
    $speaker.Rate = $Rate
    $text = Get-Content -LiteralPath $TextPath -Raw
    $stream.Open($OutputPath, 3, $false)
    $speaker.AudioOutputStream = $stream
    [void]$speaker.Speak($text)
    $stream.Close()
    Write-Output $speaker.Voice.GetDescription()
}
finally {
    if ($stream) {
        try { $stream.Close() } catch {}
    }
}
'''.replace("__TEXT_PATH__", text_literal).replace(
            "__OUTPUT_PATH__",
            output_literal,
        ).replace(
            "__RATE__",
            str(int(self.rate)),
        ).replace(
            "__VOICE_NAME__",
            voice_literal,
        )

    @staticmethod
    def _default_voice_from_stdout(stdout: str) -> str:
        lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
        return lines[-1] if lines else ""

    @staticmethod
    def _wav_duration(path: Path) -> float:
        try:
            with wave.open(str(path), "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                return round(frames / rate, 4) if rate else 0.0
        except wave.Error:
            return 0.0
