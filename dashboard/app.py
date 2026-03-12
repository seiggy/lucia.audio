"""FastAPI dashboard for managing voice profiles and testing TTS."""

import io
import logging
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice

from wyoming_chatterbox import __version__
from wyoming_chatterbox.engine_manager import EngineManager
from wyoming_chatterbox.voice_manager import VoiceManager

_LOGGER = logging.getLogger(__name__)

DASHBOARD_DIR = Path(__file__).parent


def create_app(
    engine_mgr: EngineManager,
    voice_manager: VoiceManager,
    wyoming_info: Info,
) -> FastAPI:
    app = FastAPI(title="Lucia.Audio — Chatterbox TTS Dashboard", version=__version__)

    templates = Jinja2Templates(directory=str(DASHBOARD_DIR / "templates"))
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        profiles = voice_manager.list_profiles()
        engines = engine_mgr.list_engines()
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "profiles": profiles,
                "engines": engines,
                "version": __version__,
            },
        )

    @app.get("/api/voices")
    async def list_voices():
        profiles = voice_manager.list_profiles()
        return [p.to_dict() for p in profiles]

    @app.get("/api/engines")
    async def list_engines():
        return engine_mgr.list_engines()

    @app.get("/api/engine/status")
    async def engine_status():
        return engine_mgr.get_status()

    @app.post("/api/engine/activate")
    async def activate_engine(engine_id: str = Form(...)):
        if engine_id not in engine_mgr.engines:
            raise HTTPException(404, f"Unknown engine: {engine_id}")
        try:
            await engine_mgr.activate_engine(engine_id)
            _update_wyoming_voices(wyoming_info, voice_manager)
            return engine_mgr.get_status()
        except Exception as e:
            _LOGGER.error("Engine activation failed: %s", e, exc_info=True)
            raise HTTPException(500, f"Failed to activate engine: {str(e)}")

    @app.post("/api/voices")
    async def create_voice(
        name: str = Form(...),
        description: str = Form(""),
        exaggeration: float = Form(0.5),
        audio: list[UploadFile] = File(...),
    ):
        # Validate at least one file
        if not audio or not audio[0].filename:
            raise HTTPException(400, "No file uploaded")

        # Read and convert each file to WAV
        wav_chunks = []
        for f in audio:
            data = await f.read()
            if len(data) < 500:
                continue
            filename = (f.filename or "").lower()
            if not filename.endswith(".wav"):
                try:
                    data = _convert_to_wav(data)
                except Exception as e:
                    _LOGGER.error("Audio conversion failed for %s: %s", f.filename, e)
                    raise HTTPException(400, f"Failed to convert {f.filename}: {e}")
            wav_chunks.append(data)

        if not wav_chunks:
            raise HTTPException(400, "No valid audio files uploaded")

        # Concatenate multiple clips into one WAV
        if len(wav_chunks) > 1:
            try:
                audio_data = _concatenate_wavs(wav_chunks)
            except Exception as e:
                _LOGGER.error("Audio concatenation failed: %s", e)
                raise HTTPException(400, f"Failed to combine audio clips: {e}")
        else:
            audio_data = wav_chunks[0]

        # Validate combined duration
        if not _is_valid_audio(audio_data):
            raise HTTPException(400, "Combined audio too short. Please upload at least 5 seconds total.")

        exaggeration = max(0.0, min(1.0, exaggeration))

        # Create profile
        profile = voice_manager.create_profile(name, audio_data, description, exaggeration)

        # Compute Chatterbox conditionals
        try:
            audio_path = voice_manager.get_audio_path(profile.id)
            conds_path = voice_manager.get_profile_dir(profile.id) / "conds.pt"
            chatterbox = engine_mgr.get_chatterbox()
            if chatterbox and chatterbox._model is not None:
                await chatterbox.compute_conditionals_async(
                    str(audio_path), str(conds_path), exaggeration=exaggeration
                )
                voice_manager.mark_ready(profile.id)
            else:
                # No Chatterbox — mark ready anyway (Qwen3 uses raw audio)
                voice_manager.mark_ready(profile.id)
            voice_manager.mark_ready(profile.id)
            profile = voice_manager.get_profile(profile.id)
        except Exception as e:
            _LOGGER.error("Failed to compute conditionals: %s", e)
            voice_manager.delete_profile(profile.id)
            raise HTTPException(500, f"Failed to process voice: {str(e)}")

        # Update Wyoming info with new voice
        _update_wyoming_voices(wyoming_info, voice_manager)

        return profile.to_dict()

    @app.delete("/api/voices/{profile_id}")
    async def delete_voice(profile_id: str):
        if not voice_manager.delete_profile(profile_id):
            raise HTTPException(404, "Voice profile not found")
        _update_wyoming_voices(wyoming_info, voice_manager)
        return {"status": "deleted"}

    @app.put("/api/voices/{profile_id}")
    async def update_voice(
        profile_id: str,
        name: Optional[str] = Form(None),
        description: Optional[str] = Form(None),
        exaggeration: Optional[float] = Form(None),
    ):
        profile = voice_manager.get_profile(profile_id)
        if profile is None:
            raise HTTPException(404, "Voice profile not found")

        # If exaggeration changed, recompute conditionals
        if exaggeration is not None and abs(exaggeration - profile.exaggeration) > 0.01:
            exaggeration = max(0.0, min(1.0, exaggeration))
            audio_path = voice_manager.get_audio_path(profile_id)
            conds_path = voice_manager.get_profile_dir(profile_id) / "conds.pt"
            if audio_path and audio_path.exists():
                try:
                    chatterbox = engine_mgr.get_chatterbox()
                    if chatterbox and chatterbox._model is not None:
                        await chatterbox.compute_conditionals_async(
                            str(audio_path), str(conds_path), exaggeration=exaggeration
                        )
                        voice_manager.mark_ready(profile_id)
                except Exception as e:
                    _LOGGER.error("Failed to recompute conditionals: %s", e)
                    raise HTTPException(500, f"Failed to reprocess voice: {str(e)}")

        profile = voice_manager.update_profile(profile_id, name, description, exaggeration)
        if profile is None:
            raise HTTPException(404, "Voice profile not found")
        _update_wyoming_voices(wyoming_info, voice_manager)
        return profile.to_dict()

    @app.post("/api/tts")
    async def test_tts(
        text: str = Form(...),
        voice_id: str = Form(""),
        engine_id: str = Form(""),
        temperature: float = Form(0.8),
        top_p: float = Form(0.95),
        top_k: int = Form(1000),
        repetition_penalty: float = Form(1.2),
    ):
        if not text.strip():
            raise HTTPException(400, "Text is required")

        temperature = max(0.1, min(2.0, temperature))
        top_p = max(0.0, min(1.0, top_p))
        top_k = max(1, min(5000, top_k))
        repetition_penalty = max(1.0, min(3.0, repetition_penalty))

        conds_path = None
        audio_prompt_path = None
        if voice_id:
            conds_path = voice_manager.get_conds_path(voice_id)
            audio_prompt_path = voice_manager.get_audio_path(voice_id)
        if conds_path is None:
            default = voice_manager.get_default_voice()
            if default:
                conds_path = voice_manager.get_conds_path(default.id)
                audio_prompt_path = voice_manager.get_audio_path(default.id)

        try:
            wav_bytes = await engine_mgr.synthesize(
                text=text.strip(),
                engine_id=engine_id or None,
                voice_conds_path=conds_path,
                audio_prompt_path=str(audio_prompt_path) if audio_prompt_path else None,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            return Response(content=wav_bytes, media_type="audio/wav")
        except Exception as e:
            _LOGGER.error("TTS error: %s", e, exc_info=True)
            raise HTTPException(500, f"Synthesis failed: {str(e)}")

    @app.get("/api/voices/{profile_id}/audio")
    async def get_voice_audio(profile_id: str):
        audio_path = voice_manager.get_audio_path(profile_id)
        if audio_path is None or not audio_path.exists():
            raise HTTPException(404, "Audio not found")
        return Response(
            content=audio_path.read_bytes(),
            media_type="audio/wav",
        )

    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "version": __version__,
            "engines": engine_mgr.list_engines(),
            "voices": len(voice_manager.list_profiles()),
        }

    @app.post("/api/voices/audition")
    async def audition_exaggeration(
        text: str = Form("Hello! This is a test of how my voice sounds with different expression levels."),
        audio: list[UploadFile] = File(...),
        values: str = Form(""),  # comma-separated exaggeration values, or empty for defaults
    ):
        """Generate TTS samples at multiple exaggeration levels for A/B comparison.

        Phase 1 (coarse): values="" → uses 0.0, 0.25, 0.5, 0.75, 1.0
        Phase 2 (fine): values="0.2,0.25,0.3,0.35,0.4" → uses specified values
        """
        chatterbox = engine_mgr.get_chatterbox()
        if not chatterbox or chatterbox._model is None:
            raise HTTPException(400, "Chatterbox engine not loaded (required for exaggeration tuning)")

        # Parse exaggeration values
        if values.strip():
            exag_values = [max(0.0, min(1.0, float(v.strip()))) for v in values.split(",")]
        else:
            exag_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        # Read and prepare audio
        wav_chunks = []
        for f in audio:
            data = await f.read()
            if len(data) < 500:
                continue
            filename = (f.filename or "").lower()
            if not filename.endswith(".wav"):
                try:
                    data = _convert_to_wav(data)
                except Exception as e:
                    raise HTTPException(400, f"Failed to convert {f.filename}: {e}")
            wav_chunks.append(data)

        if not wav_chunks:
            raise HTTPException(400, "No valid audio files")

        if len(wav_chunks) > 1:
            audio_data = _concatenate_wavs(wav_chunks)
        else:
            audio_data = wav_chunks[0]

        if not _is_valid_audio(audio_data):
            raise HTTPException(400, "Audio too short (need 5+ seconds)")

        # Save temp audio file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_audio_path = tmp.name

        results = []
        try:
            for exag in exag_values:
                _LOGGER.info("Audition: generating sample at exaggeration=%.2f", exag)
                try:
                    # Compute conditionals at this exaggeration
                    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_conds:
                        tmp_conds_path = tmp_conds.name

                    await chatterbox.compute_conditionals_async(
                        tmp_audio_path, tmp_conds_path, exaggeration=exag,
                    )

                    # Generate sample audio
                    wav_bytes = await chatterbox.synthesize(
                        text=text.strip(),
                        voice_conds_path=Path(tmp_conds_path),
                    )

                    import base64
                    results.append({
                        "exaggeration": round(exag, 2),
                        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
                    })

                    Path(tmp_conds_path).unlink(missing_ok=True)

                except Exception as e:
                    _LOGGER.error("Audition failed at exag=%.2f: %s", exag, e)
                    results.append({
                        "exaggeration": round(exag, 2),
                        "error": str(e),
                    })
        finally:
            Path(tmp_audio_path).unlink(missing_ok=True)

        return results

    @app.post("/api/benchmark")
    async def run_benchmark(
        text: str = Form(...),
        voice_id: str = Form(""),
    ):
        if not text.strip():
            raise HTTPException(400, "Text is required")

        from wyoming_chatterbox.benchmark import run_comparative_benchmark

        try:
            results = await run_comparative_benchmark(
                engine_mgr, text.strip(), voice_id or None, voice_manager,
            )
            return [r.to_dict() for r in results]
        except Exception as e:
            _LOGGER.error("Benchmark error: %s", e, exc_info=True)
            raise HTTPException(500, f"Benchmark failed: {str(e)}")

    return app


def _update_wyoming_voices(wyoming_info: Info, voice_manager: VoiceManager) -> None:
    """Rebuild the Wyoming TTS voice list from current profiles."""
    voices = []
    for profile in voice_manager.list_profiles():
        if profile.is_ready:
            voices.append(
                TtsVoice(
                    name=profile.name,
                    description=profile.description or f"Chatterbox voice: {profile.name}",
                    attribution=Attribution(
                        name="Resemble AI",
                        url="https://github.com/resemble-ai/chatterbox",
                    ),
                    installed=True,
                    version=__version__,
                    languages=["en"],
                )
            )
    if not voices:
        voices.append(
            TtsVoice(
                name="default",
                description="Chatterbox Turbo default voice",
                attribution=Attribution(
                    name="Resemble AI",
                    url="https://github.com/resemble-ai/chatterbox",
                ),
                installed=True,
                version=__version__,
                languages=["en"],
            )
        )
    if wyoming_info.tts:
        wyoming_info.tts[0].voices = sorted(voices, key=lambda v: v.name)


def _convert_to_wav(audio_data: bytes) -> bytes:
    """Convert any audio format to 16-bit mono WAV using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".input", delete=True) as infile, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as outfile:
        infile.write(audio_data)
        infile.flush()

        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", infile.name,
                "-ac", "1",           # mono
                "-ar", "44100",       # 44.1kHz (will be resampled by engines as needed)
                "-sample_fmt", "s16", # 16-bit
                "-f", "wav",
                outfile.name,
            ],
            capture_output=True, timeout=30,
        )

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")[:500]
            raise RuntimeError(f"ffmpeg failed: {stderr}")

        return Path(outfile.name).read_bytes()


def _concatenate_wavs(wav_chunks: list[bytes]) -> bytes:
    """Concatenate multiple WAV files into one using ffmpeg, with 0.3s silence gaps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        # Write each chunk and build ffmpeg concat list
        list_path = tmpdir / "list.txt"
        silence_path = tmpdir / "silence.wav"

        # Generate a short silence WAV
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
             "-t", "0.3", "-sample_fmt", "s16", silence_path],
            capture_output=True, timeout=10,
        )

        entries = []
        for i, chunk in enumerate(wav_chunks):
            chunk_path = tmpdir / f"chunk_{i}.wav"
            # Normalize to same format
            raw_path = tmpdir / f"raw_{i}.wav"
            raw_path.write_bytes(chunk)
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(raw_path),
                 "-ac", "1", "-ar", "44100", "-sample_fmt", "s16", str(chunk_path)],
                capture_output=True, timeout=30,
            )
            entries.append(f"file '{chunk_path}'")
            if i < len(wav_chunks) - 1:
                entries.append(f"file '{silence_path}'")

        list_path.write_text("\n".join(entries))
        out_path = tmpdir / "combined.wav"

        result = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(list_path), "-c", "copy", str(out_path)],
            capture_output=True, timeout=60,
        )

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")[:500]
            raise RuntimeError(f"ffmpeg concat failed: {stderr}")

        return out_path.read_bytes()


def _is_valid_audio(data: bytes) -> bool:
    """Check if audio data is a valid WAV file."""
    try:
        with io.BytesIO(data) as buf:
            with wave.open(buf, "rb") as wf:
                duration = wf.getnframes() / wf.getframerate()
                return duration >= 5.0
    except Exception:
        return False
