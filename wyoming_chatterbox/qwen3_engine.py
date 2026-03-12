"""Faster Qwen3-TTS engine.

CUDA-graph accelerated Qwen3-TTS with voice cloning support.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .tts_engine import audio_to_wav_bytes

_LOGGER = logging.getLogger(__name__)

QWEN3_MODEL_0_6B = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
QWEN3_MODEL_1_7B = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


class Qwen3Engine:
    """Faster Qwen3-TTS engine with CUDA graph acceleration and voice cloning."""

    ENGINE_ID = "qwen3"
    ENGINE_NAME = "Qwen3-TTS"

    def __init__(self, device: str = "cuda", model_name: str = QWEN3_MODEL_0_6B):
        self.device = device
        self.model_name = model_name
        self._model = None
        self._inference_lock = asyncio.Lock()
        self.sample_rate = 24000  # will be updated from model

    async def load_model(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)

    async def unload_model(self) -> None:
        """Unload model and free GPU memory."""
        import gc
        import torch
        self._model = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        _LOGGER.info("Qwen3-TTS unloaded")

    def _load_model_sync(self) -> None:
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        _LOGGER.info("Loading Qwen3-TTS model: %s", self.model_name)
        start = time.monotonic()

        self._model = FasterQwen3TTS.from_pretrained(
            self.model_name,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.sample_rate = self._model.sample_rate

        _LOGGER.info(
            "Qwen3-TTS loaded in %.1fs (sample_rate=%d)",
            time.monotonic() - start, self.sample_rate,
        )

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Qwen3-TTS model not loaded")
        return self._model

    async def synthesize(
        self,
        text: str,
        voice_conds_path: Optional[Path] = None,
        audio_prompt_path: Optional[str] = None,
        ref_text: str = "",
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        **kwargs,
    ) -> bytes:
        loop = asyncio.get_event_loop()
        async with self._inference_lock:
            return await loop.run_in_executor(
                None, self._synthesize_sync,
                text, audio_prompt_path, ref_text,
                temperature, top_p, top_k, repetition_penalty,
            )

    async def synthesize_streaming(
        self,
        text: str,
        voice_conds_path: Optional[Path] = None,
        audio_prompt_path: Optional[str] = None,
        ref_text: str = "",
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        chunk_tokens: int = 8,
        **kwargs,
    ):
        """Async generator yielding (audio_np, sample_rate) chunks via Qwen3 streaming."""
        import queue
        import threading

        async with self._inference_lock:
            if not audio_prompt_path or not Path(audio_prompt_path).exists():
                raise RuntimeError("Qwen3 streaming requires reference audio")

            loop = asyncio.get_event_loop()
            q = queue.Queue()
            error_holder = [None]

            def _run():
                try:
                    for audio_chunk, sr, timing in self.model.generate_voice_clone_streaming(
                        text=text,
                        language="English",
                        ref_audio=str(audio_prompt_path),
                        ref_text=ref_text or "",
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        chunk_size=chunk_tokens,
                    ):
                        q.put((audio_chunk.flatten().astype(np.float32), sr))
                    q.put(None)
                except Exception as e:
                    error_holder[0] = e
                    q.put(None)

            thread = threading.Thread(target=_run, daemon=True)
            thread.start()

            while True:
                item = await loop.run_in_executor(None, q.get)
                if item is None:
                    if error_holder[0]:
                        raise error_holder[0]
                    break
                yield item

    def _synthesize_sync(
        self, text, audio_prompt_path, ref_text,
        temperature, top_p, top_k, repetition_penalty,
    ) -> bytes:
        start = time.monotonic()

        if audio_prompt_path and Path(audio_prompt_path).exists():
            # Voice cloning mode
            audio_list, sr = self.model.generate_voice_clone(
                text=text,
                language="English",
                ref_audio=str(audio_prompt_path),
                ref_text=ref_text or "",
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        else:
            # No reference audio — use base generation
            audio_list, sr = self.model.generate_voice_clone(
                text=text,
                language="English",
                ref_audio=str(audio_prompt_path) if audio_prompt_path else "",
                ref_text=ref_text or "",
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

        elapsed = time.monotonic() - start
        _LOGGER.info("Qwen3 synthesized %d chars in %.2fs", len(text), elapsed)

        # Concatenate audio chunks
        if audio_list:
            audio_np = np.concatenate([a.flatten() for a in audio_list])
        else:
            audio_np = np.zeros(1, dtype=np.float32)

        return audio_to_wav_bytes(audio_np, sr)
