"""FastAPI dashboard for managing voice profiles and testing TTS."""

import io
import logging
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

    @app.post("/api/voices")
    async def create_voice(
        name: str = Form(...),
        description: str = Form(""),
        exaggeration: float = Form(0.5),
        audio: UploadFile = File(...),
    ):
        # Validate audio file
        if not audio.filename:
            raise HTTPException(400, "No file uploaded")

        audio_data = await audio.read()
        if len(audio_data) < 1000:
            raise HTTPException(400, "Audio file too small")

        # Validate it's a valid WAV or convert
        if not _is_valid_audio(audio_data):
            raise HTTPException(400, "Invalid audio file. Please upload a WAV file (at least 5 seconds).")

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


def _is_valid_audio(data: bytes) -> bool:
    """Check if audio data is a valid WAV file."""
    try:
        with io.BytesIO(data) as buf:
            with wave.open(buf, "rb") as wf:
                duration = wf.getnframes() / wf.getframerate()
                return duration >= 5.0
    except Exception:
        return False
