"""Wyoming protocol event handler for TTS with streaming support."""

import io
import logging
import math
import wave
from typing import Optional

import numpy as np

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import Synthesize

from .engine_manager import EngineManager
from .voice_manager import VoiceManager

_LOGGER = logging.getLogger(__name__)

SAMPLES_PER_CHUNK = 1024
# Output format for Wyoming: 16-bit mono PCM
WYOMING_RATE = 24000
WYOMING_WIDTH = 2
WYOMING_CHANNELS = 1


class ChatterboxEventHandler(AsyncEventHandler):
    """Handles Wyoming TTS protocol events with streaming audio output."""

    def __init__(
        self,
        wyoming_info: Info,
        engine_mgr: EngineManager,
        voice_manager: VoiceManager,
        default_voice: Optional[str],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info
        self.wyoming_info_event = wyoming_info.event()
        self.engine_mgr = engine_mgr
        self.voice_manager = voice_manager
        self.default_voice = default_voice

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        if not Synthesize.is_type(event.type):
            _LOGGER.debug("Ignoring event type: %s", event.type)
            return True

        try:
            synthesize = Synthesize.from_event(event)
            await self._handle_synthesize(synthesize)
        except Exception as err:
            _LOGGER.error("Synthesis error: %s", err, exc_info=True)
            await self.write_event(
                Error(text=str(err), code=err.__class__.__name__).event()
            )

        return True

    async def _handle_synthesize(self, synthesize: Synthesize) -> None:
        text = synthesize.text.strip()
        if not text:
            _LOGGER.warning("Empty text received")
            return

        _LOGGER.info("Synthesize request: '%s'", text[:100])

        # Resolve voice profile
        voice_name = None
        if synthesize.voice and synthesize.voice.name:
            voice_name = synthesize.voice.name

        if voice_name is None:
            voice_name = self.default_voice

        conds_path = None
        audio_path = None
        if voice_name:
            profile = self.voice_manager.get_profile_by_name(voice_name)
            if profile is None:
                profile = self.voice_manager.get_profile(voice_name)
            if profile and profile.is_ready:
                conds_path = self.voice_manager.get_conds_path(profile.id)
                audio_path = self.voice_manager.get_audio_path(profile.id)
                _LOGGER.debug("Using voice profile: %s", profile.name)
            else:
                _LOGGER.warning("Voice '%s' not found or not ready, using default", voice_name)

        if conds_path is None:
            default = self.voice_manager.get_default_voice()
            if default:
                conds_path = self.voice_manager.get_conds_path(default.id)
                audio_path = self.voice_manager.get_audio_path(default.id)
                _LOGGER.debug("Using default voice: %s", default.name)

        # Try streaming synthesis first, fall back to non-streaming
        try:
            await self._synthesize_streaming(text, conds_path, audio_path)
        except Exception as stream_err:
            _LOGGER.debug("Streaming not available, falling back to non-streaming: %s", stream_err)
            await self._synthesize_batch(text, conds_path, audio_path)

        _LOGGER.info("Synthesis complete for: '%s'", text[:50])

    async def _synthesize_streaming(self, text, conds_path, audio_path):
        """Stream audio chunks as they're generated."""
        start_sent = False

        async for audio_np, sr in self.engine_mgr.synthesize_streaming(
            text=text,
            voice_conds_path=conds_path,
            audio_prompt_path=str(audio_path) if audio_path else None,
        ):
            # Convert float32 numpy to 16-bit PCM bytes
            audio_np = np.clip(audio_np.flatten(), -1.0, 1.0)
            audio_int16 = (audio_np * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            if not start_sent:
                await self.write_event(
                    AudioStart(rate=sr, width=WYOMING_WIDTH, channels=WYOMING_CHANNELS).event()
                )
                start_sent = True

            # Send in sub-chunks for Wyoming
            bytes_per_chunk = WYOMING_WIDTH * WYOMING_CHANNELS * SAMPLES_PER_CHUNK
            for offset in range(0, len(audio_bytes), bytes_per_chunk):
                chunk = audio_bytes[offset : offset + bytes_per_chunk]
                await self.write_event(
                    AudioChunk(
                        audio=chunk, rate=sr,
                        width=WYOMING_WIDTH, channels=WYOMING_CHANNELS,
                    ).event()
                )

        if start_sent:
            await self.write_event(AudioStop().event())
        else:
            raise RuntimeError("No audio chunks produced")

    async def _synthesize_batch(self, text, conds_path, audio_path):
        """Non-streaming fallback: synthesize all at once, then send."""
        wav_bytes = await self.engine_mgr.synthesize(
            text=text,
            voice_conds_path=conds_path,
            audio_prompt_path=str(audio_path) if audio_path else None,
        )

        with io.BytesIO(wav_bytes) as wav_io:
            wav_file: wave.Wave_read = wave.open(wav_io, "rb")
            with wav_file:
                rate = wav_file.getframerate()
                width = wav_file.getsampwidth()
                channels = wav_file.getnchannels()

                await self.write_event(
                    AudioStart(rate=rate, width=width, channels=channels).event()
                )

                audio_bytes = wav_file.readframes(wav_file.getnframes())
                bytes_per_chunk = width * channels * SAMPLES_PER_CHUNK

                for offset in range(0, len(audio_bytes), bytes_per_chunk):
                    chunk = audio_bytes[offset : offset + bytes_per_chunk]
                    await self.write_event(
                        AudioChunk(
                            audio=chunk, rate=rate,
                            width=width, channels=channels,
                        ).event()
                    )

                await self.write_event(AudioStop().event())
