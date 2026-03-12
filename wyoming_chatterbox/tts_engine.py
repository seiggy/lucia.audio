"""Chatterbox Turbo TTS engine wrapper.

Manages model lifecycle, voice conditional caching, and thread-safe inference.
"""

import asyncio
import io
import logging
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_LOGGER = logging.getLogger(__name__)

# Chatterbox Turbo outputs at this sample rate
OUTPUT_SAMPLE_RATE = 24000


class TTSEngine:
    """Singleton wrapper around ChatterboxTurboTTS with performance optimizations."""

    def __init__(self, device: str = "cuda", half_precision: bool = True):
        self.device = device
        self.half_precision = half_precision and device == "cuda"
        self._model = None
        self._lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self.sample_rate = OUTPUT_SAMPLE_RATE

    async def load_model(self) -> None:
        """Load the Chatterbox Turbo model onto the GPU."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)

    def _load_model_sync(self) -> None:
        import os
        from huggingface_hub import snapshot_download
        from chatterbox.tts_turbo import ChatterboxTurboTTS, REPO_ID

        # Step 1: Download model weights with progress
        _LOGGER.info("Downloading Chatterbox Turbo model weights (this may take a while on first run)...")
        start = time.monotonic()

        local_path = snapshot_download(
            repo_id=REPO_ID,
            token=os.getenv("HF_TOKEN") or True,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
        )

        elapsed = time.monotonic() - start
        _LOGGER.info("Model download complete in %.1fs -> %s", elapsed, local_path)

        # Step 2: Load model from local cache (keep fp32 — autocast handles mixed precision)
        _LOGGER.info("Loading model onto %s...", self.device)
        start = time.monotonic()

        self._model = ChatterboxTurboTTS.from_local(local_path, device=self.device)

        elapsed = time.monotonic() - start
        _LOGGER.info("Model loaded in %.1fs", elapsed)

        # Warm up: only possible if model has built-in default conditionals
        if self._model.conds is not None:
            _LOGGER.info("Warming up model...")
            start = time.monotonic()
            try:
                _ = self._generate_with_autocast("Hello.")
                elapsed = time.monotonic() - start
                _LOGGER.info("Warm-up complete in %.1fs", elapsed)
            except Exception as e:
                _LOGGER.warning("Warm-up failed (non-fatal): %s", e)
        else:
            _LOGGER.info("No default conditionals — skipping warm-up (upload a voice profile to enable TTS)")

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model

    async def synthesize(
        self,
        text: str,
        voice_conds_path: Optional[Path] = None,
        audio_prompt_path: Optional[str] = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 1000,
        repetition_penalty: float = 1.2,
    ) -> bytes:
        """Synthesize text to WAV bytes (16-bit PCM, mono).

        Uses pre-computed conditionals for minimum latency when available.
        Returns raw WAV file bytes.
        """
        loop = asyncio.get_event_loop()

        async with self._inference_lock:
            wav_bytes = await loop.run_in_executor(
                None,
                self._synthesize_sync,
                text,
                voice_conds_path,
                audio_prompt_path,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
            )

        return wav_bytes

    def _generate_with_autocast(self, text: str, **kwargs):
        """Run model.generate with CUDA autocast for automatic mixed precision."""
        if self.half_precision:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                return self.model.generate(text, **kwargs)
        return self.model.generate(text, **kwargs)

    def _synthesize_sync(
        self,
        text: str,
        voice_conds_path: Optional[Path],
        audio_prompt_path: Optional[str],
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> bytes:
        from chatterbox.tts_turbo import Conditionals

        start = time.monotonic()
        gen_kwargs = dict(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        # Load pre-computed conditionals if available
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

        elapsed = time.monotonic() - start
        _LOGGER.info("Synthesized %d chars in %.2fs", len(text), elapsed)

        # Convert to 16-bit PCM WAV bytes
        audio_np = wav_tensor.squeeze(0).numpy()
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_np * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())

        return buf.getvalue()

    def compute_conditionals(self, audio_path: str, output_path: str, exaggeration: float = 0.5) -> None:
        """Pre-compute voice conditionals from a reference audio clip and save to disk."""
        _LOGGER.info("Computing conditionals from %s (exaggeration=%.2f)", audio_path, exaggeration)
        start = time.monotonic()

        self.model.prepare_conditionals(audio_path, exaggeration=exaggeration)

        # Save the computed conditionals
        self.model.conds.save(Path(output_path))

        elapsed = time.monotonic() - start
        _LOGGER.info("Conditionals computed in %.1fs -> %s", elapsed, output_path)

    async def compute_conditionals_async(self, audio_path: str, output_path: str, exaggeration: float = 0.5) -> None:
        """Async wrapper for conditional computation."""
        loop = asyncio.get_event_loop()
        async with self._lock:
            await loop.run_in_executor(
                None, self.compute_conditionals, audio_path, output_path, exaggeration
            )
