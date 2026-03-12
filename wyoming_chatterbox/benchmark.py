"""Benchmark runner — measures TTS engine performance across multiple samples.

Benchmarks the active Wyoming server by sending Synthesize events over TCP
and measuring time to first audio chunk (TTFA) and total response time.
This captures the real end-to-end latency including protocol overhead.
"""

import asyncio
import base64
import io
import logging
import time
import wave
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import async_read_event, async_write_event
from wyoming.tts import Synthesize as WyomingSynthesize, SynthesizeVoice

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


async def _run_sample_via_wyoming(
    host: str, port: int, text: str, voice_name: Optional[str] = None,
) -> SampleResult:
    """Benchmark a single synthesis by sending a Wyoming Synthesize event over TCP.

    Measures real TTFA (time to first AudioChunk) and total time.
    """
    from .tts_engine import audio_to_wav_bytes

    start = time.monotonic()
    ttfa = 0.0
    audio_chunks = []
    sample_rate = 24000
    width = 2
    channels = 1

    try:
        reader, writer = await asyncio.open_connection(host, port)

        # Send Synthesize event
        voice = SynthesizeVoice(name=voice_name) if voice_name else None
        synth = WyomingSynthesize(text=text, voice=voice)
        await async_write_event(synth.event(), writer)

        # Read response events
        first_audio = True
        while True:
            event = await asyncio.wait_for(async_read_event(reader), timeout=120)
            if event is None:
                break

            if AudioStart.is_type(event.type):
                audio_start = AudioStart.from_event(event)
                sample_rate = audio_start.rate
                width = audio_start.width
                channels = audio_start.channels

            elif AudioChunk.is_type(event.type):
                if first_audio:
                    ttfa = (time.monotonic() - start) * 1000
                    first_audio = False
                chunk = AudioChunk.from_event(event)
                audio_chunks.append(chunk.audio)

            elif AudioStop.is_type(event.type):
                break

        writer.close()

    except Exception as e:
        _LOGGER.error("Wyoming benchmark failed: %s", e)
        return SampleResult(
            text=text, char_count=len(text),
            total_time_ms=0, ttfa_ms=0, audio_duration_s=0, rtf=0,
            audio_b64="", error=str(e),
        )

    elapsed = time.monotonic() - start
    elapsed_ms = elapsed * 1000

    if not audio_chunks:
        return SampleResult(
            text=text, char_count=len(text),
            total_time_ms=round(elapsed_ms, 1), ttfa_ms=0,
            audio_duration_s=0, rtf=0, audio_b64="",
            error="No audio received",
        )

    # Combine raw PCM chunks into WAV
    all_audio = b"".join(audio_chunks)
    audio_samples = len(all_audio) // (width * channels)
    audio_s = audio_samples / sample_rate
    rtf = audio_s / elapsed if elapsed > 0 else 0

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(sample_rate)
        wf.writeframes(all_audio)
    wav_bytes = buf.getvalue()

    return SampleResult(
        text=text, char_count=len(text),
        total_time_ms=round(elapsed_ms, 1),
        ttfa_ms=round(ttfa, 1),
        audio_duration_s=round(audio_s, 2),
        rtf=round(rtf, 2),
        audio_b64=base64.b64encode(wav_bytes).decode("ascii"),
    )


async def run_comparative_benchmark(
    engine_mgr,
    text: str,
    voice_id: Optional[str],
    voice_manager,
    wyoming_host: str = "127.0.0.1",
    wyoming_port: int = 10200,
) -> List[EngineBenchmarkResult]:
    """Run multi-sample benchmarks across all engines via the Wyoming protocol.

    For each engine: activates it on the server, runs warm-up + samples
    through the Wyoming TCP connection, then moves to the next engine.
    This measures real end-to-end latency including streaming and protocol overhead.
    """
    results = []
    sentences = [text] + BENCHMARK_SENTENCES

    # Resolve voice name for Wyoming Synthesize events
    voice_name = None
    if voice_id:
        profile = voice_manager.get_profile(voice_id)
        if profile:
            voice_name = profile.name
    if voice_name is None:
        default = voice_manager.get_default_voice()
        if default:
            voice_name = default.name

    for engine_id, engine in engine_mgr.engines.items():
        _LOGGER.info("Benchmarking %s via Wyoming protocol...", engine.ENGINE_NAME)

        # Activate this engine on the server
        try:
            await engine_mgr.activate_engine(engine_id)
        except Exception as e:
            _LOGGER.error("Failed to activate %s: %s", engine.ENGINE_NAME, e)
            results.append(EngineBenchmarkResult(
                engine_id=engine_id, engine_name=engine.ENGINE_NAME,
                samples=[], avg_rtf=0, avg_time_ms=0, avg_ttfa_ms=0,
                total_audio_s=0, total_time_ms=0,
                vram_peak_mb=0, vram_allocated_mb=0,
                error=f"Failed to activate: {e}",
            ))
            continue

        # Wait a moment for the server to be ready
        await asyncio.sleep(0.5)

        # Warm up via Wyoming
        _LOGGER.info("Warm-up for %s via Wyoming...", engine.ENGINE_NAME)
        await _run_sample_via_wyoming(
            wyoming_host, wyoming_port,
            "Warming up audio engine, please wait.",
            voice_name=voice_name,
        )

        # Reset VRAM stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Run samples
        _LOGGER.info("Running %d samples on %s...", len(sentences), engine.ENGINE_NAME)
        samples = []
        for i, sentence in enumerate(sentences):
            _LOGGER.info("  Sample %d/%d (%d chars)", i + 1, len(sentences), len(sentence))
            result = await _run_sample_via_wyoming(
                wyoming_host, wyoming_port, sentence, voice_name=voice_name,
            )
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

    return results
