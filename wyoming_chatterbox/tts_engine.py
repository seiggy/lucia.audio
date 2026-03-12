"""Chatterbox Turbo TTS engine.

Manages model lifecycle, voice conditional caching, and thread-safe inference.
"""

import asyncio
import io
import logging
import os
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_LOGGER = logging.getLogger(__name__)


def audio_to_wav_bytes(audio_np: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy audio to 16-bit PCM WAV bytes."""
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


class ChatterboxEngine:
    """Chatterbox Turbo TTS engine with pre-computed conditional caching."""

    ENGINE_ID = "chatterbox"
    ENGINE_NAME = "Chatterbox Turbo"

    def __init__(self, device: str = "cuda", half_precision: bool = True):
        self.device = device
        self.half_precision = half_precision and device == "cuda"
        self._model = None
        self._lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self.sample_rate = 24000

    async def load_model(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)

    async def unload_model(self) -> None:
        """Unload model and free GPU memory."""
        import torch
        self._model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        _LOGGER.info("Chatterbox Turbo unloaded")

    def _load_model_sync(self) -> None:
        from huggingface_hub import snapshot_download
        from chatterbox.tts_turbo import ChatterboxTurboTTS, REPO_ID

        _LOGGER.info("Downloading Chatterbox Turbo model weights...")
        start = time.monotonic()
        local_path = snapshot_download(
            repo_id=REPO_ID,
            token=os.getenv("HF_TOKEN") or True,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
        )
        _LOGGER.info("Download complete in %.1fs", time.monotonic() - start)

        _LOGGER.info("Loading Chatterbox Turbo onto %s...", self.device)
        start = time.monotonic()
        self._model = ChatterboxTurboTTS.from_local(local_path, device=self.device)
        _LOGGER.info("Model loaded in %.1fs", time.monotonic() - start)

        if self._model.conds is not None:
            _LOGGER.info("Warming up Chatterbox...")
            try:
                self._generate_with_autocast("Hello.")
                _LOGGER.info("Warm-up complete")
            except Exception as e:
                _LOGGER.warning("Warm-up failed (non-fatal): %s", e)

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Chatterbox model not loaded")
        return self._model

    def _generate_with_autocast(self, text: str, **kwargs):
        if self.half_precision:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                return self.model.generate(text, **kwargs)
        return self.model.generate(text, **kwargs)

    async def synthesize(
        self,
        text: str,
        voice_conds_path: Optional[Path] = None,
        audio_prompt_path: Optional[str] = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 1000,
        repetition_penalty: float = 1.2,
        **kwargs,
    ) -> bytes:
        loop = asyncio.get_event_loop()
        async with self._inference_lock:
            return await loop.run_in_executor(
                None, self._synthesize_sync,
                text, voice_conds_path, audio_prompt_path,
                temperature, top_p, top_k, repetition_penalty,
            )

    def _synthesize_sync(
        self, text, voice_conds_path, audio_prompt_path,
        temperature, top_p, top_k, repetition_penalty,
    ) -> bytes:
        from chatterbox.tts_turbo import Conditionals

        start = time.monotonic()
        gen_kwargs = dict(
            temperature=temperature, top_p=top_p,
            top_k=top_k, repetition_penalty=repetition_penalty,
        )

        if voice_conds_path and voice_conds_path.exists():
            conds = Conditionals.load(voice_conds_path, map_location=self.device)
            self.model.conds = conds
            wav_tensor = self._generate_with_autocast(text, **gen_kwargs)
        elif audio_prompt_path:
            wav_tensor = self._generate_with_autocast(
                text, audio_prompt_path=audio_prompt_path, **gen_kwargs
            )
        else:
            wav_tensor = self._generate_with_autocast(text, **gen_kwargs)

        _LOGGER.info("Chatterbox synthesized %d chars in %.2fs", len(text), time.monotonic() - start)
        return audio_to_wav_bytes(wav_tensor.squeeze(0).numpy(), self.sample_rate)

    def compute_conditionals(self, audio_path: str, output_path: str, exaggeration: float = 0.5) -> None:
        _LOGGER.info("Computing Chatterbox conditionals (exaggeration=%.2f)", exaggeration)
        start = time.monotonic()
        self.model.prepare_conditionals(audio_path, exaggeration=exaggeration)
        self.model.conds.save(Path(output_path))
        _LOGGER.info("Conditionals computed in %.1fs", time.monotonic() - start)

    async def compute_conditionals_async(self, audio_path: str, output_path: str, exaggeration: float = 0.5) -> None:
        loop = asyncio.get_event_loop()
        async with self._lock:
            await loop.run_in_executor(
                None, self.compute_conditionals, audio_path, output_path, exaggeration
            )
