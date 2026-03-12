"""Benchmark runner — measures TTS engine performance across multiple samples."""

import base64
import io
import logging
import time
import wave
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

_LOGGER = logging.getLogger(__name__)

BENCHMARK_SENTENCES = [
    "Hello, how are you today?",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "I wanted to let you know that your appointment has been rescheduled to next Thursday at three thirty in the afternoon.",
    "In a world where technology continues to evolve at an unprecedented pace, it becomes increasingly important to consider the ethical implications of artificial intelligence and its impact on society as a whole.",
    "Ladies and gentlemen, welcome aboard flight seven twenty three with non-stop service to San Francisco. Our estimated flight time today is five hours and forty two minutes. We'll be cruising at an altitude of thirty six thousand feet. Please make sure your seatbelts are fastened, your tray tables are stowed, and your seat backs are in the upright position for takeoff.",
]


@dataclass
class SampleResult:
    text: str
    char_count: int
    total_time_ms: float
    ttfa_ms: float  # time to first audio chunk (0 if non-streaming)
    audio_duration_s: float
    rtf: float
    audio_b64: str  # base64 WAV
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EngineBenchmarkResult:
    engine_id: str
    engine_name: str
    samples: List[SampleResult]
    avg_rtf: float
    avg_time_ms: float
    avg_ttfa_ms: float
    total_audio_s: float
    total_time_ms: float
    vram_peak_mb: float
    vram_allocated_mb: float
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _wav_duration(wav_bytes: bytes) -> tuple[float, int]:
    """Return (duration_seconds, sample_rate) from WAV bytes."""
    with io.BytesIO(wav_bytes) as buf:
        with wave.open(buf, "rb") as wf:
            return wf.getnframes() / wf.getframerate(), wf.getframerate()


async def _run_sample(engine, text, voice_conds_path, audio_prompt_path) -> SampleResult:
    """Run a single synthesis sample, using streaming to measure TTFA when available."""
    from .tts_engine import audio_to_wav_bytes

    start = time.monotonic()
    ttfa = 0.0
    audio_chunks = []
    sample_rate = 24000
    used_streaming = False

    # Try streaming to get real TTFA
    if hasattr(engine, 'synthesize_streaming'):
        try:
            first_chunk = True
            async for audio_np, sr in engine.synthesize_streaming(
                text=text,
                voice_conds_path=voice_conds_path,
                audio_prompt_path=audio_prompt_path,
            ):
                if first_chunk:
                    ttfa = (time.monotonic() - start) * 1000
                    first_chunk = False
                    sample_rate = sr
                audio_chunks.append(audio_np)
            used_streaming = True
        except Exception as e:
            _LOGGER.debug("Streaming failed for sample, falling back: %s", e)
            audio_chunks = []
            used_streaming = False

    # Fall back to non-streaming
    if not used_streaming:
        start = time.monotonic()  # reset timer for fair measurement
        try:
            wav_bytes = await engine.synthesize(
                text=text,
                voice_conds_path=voice_conds_path,
                audio_prompt_path=audio_prompt_path,
            )
        except Exception as e:
            return SampleResult(
                text=text, char_count=len(text),
                total_time_ms=0, ttfa_ms=0, audio_duration_s=0, rtf=0,
                audio_b64="", error=str(e),
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.monotonic() - start
        audio_s, _ = _wav_duration(wav_bytes)
        return SampleResult(
            text=text, char_count=len(text),
            total_time_ms=round(elapsed * 1000, 1),
            ttfa_ms=round(elapsed * 1000, 1),  # non-streaming: TTFA = total
            audio_duration_s=round(audio_s, 2),
            rtf=round(audio_s / elapsed if elapsed > 0 else 0, 2),
            audio_b64=base64.b64encode(wav_bytes).decode("ascii"),
        )

    # Process streaming results
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.monotonic() - start

    if audio_chunks:
        combined = np.concatenate([c.flatten() for c in audio_chunks])
        wav_bytes = audio_to_wav_bytes(combined, sample_rate)
        audio_s = len(combined) / sample_rate
    else:
        wav_bytes = audio_to_wav_bytes(np.zeros(1, dtype=np.float32), sample_rate)
        audio_s = 0

    return SampleResult(
        text=text, char_count=len(text),
        total_time_ms=round(elapsed * 1000, 1),
        ttfa_ms=round(ttfa, 1),
        audio_duration_s=round(audio_s, 2),
        rtf=round(audio_s / elapsed if elapsed > 0 else 0, 2),
        audio_b64=base64.b64encode(wav_bytes).decode("ascii"),
    )


async def run_comparative_benchmark(
    engine_mgr,
    text: str,
    voice_id: Optional[str],
    voice_manager,
) -> List[EngineBenchmarkResult]:
    """Run multi-sample benchmarks across all engines sequentially.

    Uses BENCHMARK_SENTENCES for varied-length samples.
    The user's custom text is prepended as the first sample.
    """
    results = []

    # Build sample list: user text + built-in sentences
    sentences = [text] + BENCHMARK_SENTENCES

    # Resolve voice paths
    conds_path = None
    audio_path = None
    if voice_id:
        conds_path = voice_manager.get_conds_path(voice_id)
        audio_path = voice_manager.get_audio_path(voice_id)
    if conds_path is None:
        default = voice_manager.get_default_voice()
        if default:
            conds_path = voice_manager.get_conds_path(default.id)
            audio_path = voice_manager.get_audio_path(default.id)

    audio_prompt = str(audio_path) if audio_path else None

    for engine_id, engine in engine_mgr.engines.items():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load engine if not loaded
        was_loaded = engine._model is not None
        if not was_loaded:
            _LOGGER.info("Loading %s for benchmark...", engine.ENGINE_NAME)
            try:
                await engine.load_model()
            except Exception as e:
                results.append(EngineBenchmarkResult(
                    engine_id=engine_id, engine_name=engine.ENGINE_NAME,
                    samples=[], avg_rtf=0, avg_time_ms=0, avg_ttfa_ms=0,
                    total_audio_s=0, total_time_ms=0,
                    vram_peak_mb=0, vram_allocated_mb=0,
                    error=f"Failed to load: {e}",
                ))
                continue

        # Warm up (throwaway)
        _LOGGER.info("Warm-up run for %s...", engine.ENGINE_NAME)
        try:
            await engine.synthesize(
                text="Warming up audio engine, please wait.",
                voice_conds_path=conds_path,
                audio_prompt_path=audio_prompt,
            )
        except Exception as e:
            _LOGGER.warning("Warm-up failed for %s: %s", engine.ENGINE_NAME, e)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Run all samples
        _LOGGER.info("Running %d samples on %s...", len(sentences), engine.ENGINE_NAME)
        samples = []
        for i, sentence in enumerate(sentences):
            _LOGGER.info("  Sample %d/%d (%d chars)", i + 1, len(sentences), len(sentence))
            result = await _run_sample(engine, sentence, conds_path, audio_prompt)
            samples.append(result)

        # Aggregate
        valid = [s for s in samples if not s.error]
        avg_rtf = sum(s.rtf for s in valid) / len(valid) if valid else 0
        avg_time = sum(s.total_time_ms for s in valid) / len(valid) if valid else 0
        avg_ttfa = sum(s.ttfa_ms for s in valid) / len(valid) if valid else 0
        total_audio = sum(s.audio_duration_s for s in valid)
        total_time = sum(s.total_time_ms for s in valid)

        vram_peak = 0.0
        vram_alloc = 0.0
        if torch.cuda.is_available():
            vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            vram_alloc = torch.cuda.memory_allocated() / (1024 * 1024)

        _LOGGER.info(
            "Benchmark %s: avg RTF=%.2f, avg TTFA=%.0fms, avg time=%.0fms, VRAM=%.0fMB",
            engine.ENGINE_NAME, avg_rtf, avg_ttfa, avg_time, vram_peak,
        )

        results.append(EngineBenchmarkResult(
            engine_id=engine_id,
            engine_name=engine.ENGINE_NAME,
            samples=samples,
            avg_rtf=round(avg_rtf, 2),
            avg_time_ms=round(avg_time, 1),
            avg_ttfa_ms=round(avg_ttfa, 1),
            total_audio_s=round(total_audio, 2),
            total_time_ms=round(total_time, 1),
            vram_peak_mb=round(vram_peak, 1),
            vram_allocated_mb=round(vram_alloc, 1),
        ))

        # Always unload after benchmarking to free VRAM for next engine
        _LOGGER.info("Unloading %s after benchmark", engine.ENGINE_NAME)
        await engine.unload_model()

    # Reload the active engine after benchmark completes
    active_engine = engine_mgr.get_engine(engine_mgr.active_engine_id)
    if active_engine and active_engine._model is None:
        _LOGGER.info("Reloading active engine %s after benchmark", engine_mgr.active_engine_id)
        await active_engine.load_model()

    return results
