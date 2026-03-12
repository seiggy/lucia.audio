"""Benchmark runner — measures TTS engine performance metrics."""

import asyncio
import base64
import io
import logging
import time
import wave
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import torch

_LOGGER = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    engine_id: str
    engine_name: str
    total_time_ms: float
    audio_duration_s: float
    rtf: float  # audio_duration / generation_time — >1.0 = faster than realtime
    vram_peak_mb: float
    vram_allocated_mb: float
    sample_rate: int
    audio_b64: str  # base64-encoded WAV for playback
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _wav_duration(wav_bytes: bytes) -> tuple[float, int]:
    """Return (duration_seconds, sample_rate) from WAV bytes."""
    with io.BytesIO(wav_bytes) as buf:
        with wave.open(buf, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / rate, rate


async def run_single_benchmark(
    engine,
    text: str,
    voice_conds_path: Optional[Path] = None,
    audio_prompt_path: Optional[str] = None,
) -> BenchmarkResult:
    """Benchmark a single engine: synthesize text and capture metrics."""
    engine_id = engine.ENGINE_ID
    engine_name = engine.ENGINE_NAME

    if engine._model is None:
        return BenchmarkResult(
            engine_id=engine_id, engine_name=engine_name,
            total_time_ms=0, audio_duration_s=0, rtf=0,
            vram_peak_mb=0, vram_allocated_mb=0, sample_rate=0,
            audio_b64="", error="Model not loaded",
        )

    # Reset CUDA memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    _LOGGER.info("Benchmarking %s...", engine_name)
    start = time.monotonic()

    try:
        wav_bytes = await engine.synthesize(
            text=text,
            voice_conds_path=voice_conds_path,
            audio_prompt_path=audio_prompt_path,
        )
    except Exception as e:
        _LOGGER.error("Benchmark failed for %s: %s", engine_name, e)
        return BenchmarkResult(
            engine_id=engine_id, engine_name=engine_name,
            total_time_ms=0, audio_duration_s=0, rtf=0,
            vram_peak_mb=0, vram_allocated_mb=0, sample_rate=0,
            audio_b64="", error=str(e),
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_time = time.monotonic() - start
    total_time_ms = total_time * 1000

    # VRAM stats
    vram_peak_mb = 0.0
    vram_allocated_mb = 0.0
    if torch.cuda.is_available():
        vram_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        vram_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)

    # Audio duration
    audio_duration_s, sample_rate = _wav_duration(wav_bytes)
    rtf = audio_duration_s / total_time if total_time > 0 else 0

    # Base64 audio for playback
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

    _LOGGER.info(
        "Benchmark %s: %.0fms total, %.2fs audio, RTF=%.2f, VRAM=%.0fMB peak",
        engine_name, total_time_ms, audio_duration_s, rtf, vram_peak_mb,
    )

    return BenchmarkResult(
        engine_id=engine_id,
        engine_name=engine_name,
        total_time_ms=round(total_time_ms, 1),
        audio_duration_s=round(audio_duration_s, 2),
        rtf=round(rtf, 2),
        vram_peak_mb=round(vram_peak_mb, 1),
        vram_allocated_mb=round(vram_allocated_mb, 1),
        sample_rate=sample_rate,
        audio_b64=audio_b64,
    )


async def run_comparative_benchmark(
    engine_mgr,
    text: str,
    voice_id: Optional[str],
    voice_manager,
) -> List[BenchmarkResult]:
    """Run benchmarks across all loaded engines sequentially."""
    results = []

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
        # Clear cache between runs for fair measurement
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load engine if not loaded
        was_loaded = engine._model is not None
        if not was_loaded:
            _LOGGER.info("Loading %s for benchmark...", engine.ENGINE_NAME)
            try:
                await engine.load_model()
            except Exception as e:
                results.append(BenchmarkResult(
                    engine_id=engine_id, engine_name=engine.ENGINE_NAME,
                    total_time_ms=0, audio_duration_s=0, rtf=0,
                    vram_peak_mb=0, vram_allocated_mb=0, sample_rate=0,
                    audio_b64="", error=f"Failed to load: {e}",
                ))
                continue

        result = await run_single_benchmark(
            engine, text,
            voice_conds_path=conds_path,
            audio_prompt_path=audio_prompt,
        )
        results.append(result)

        # Unload if we loaded it just for the benchmark
        if not was_loaded:
            _LOGGER.info("Unloading %s after benchmark", engine.ENGINE_NAME)
            await engine.unload_model()

    return results
