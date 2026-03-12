"""Wyoming protocol event handler for Chatterbox TTS."""

import io
import logging
import math
import wave
from typing import Optional

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


class ChatterboxEventHandler(AsyncEventHandler):
    """Handles Wyoming TTS protocol events using Chatterbox Turbo."""

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

        # Look up voice profile
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

        # If no profile found, try default
        if conds_path is None:
            default = self.voice_manager.get_default_voice()
            if default:
                conds_path = self.voice_manager.get_conds_path(default.id)
                audio_path = self.voice_manager.get_audio_path(default.id)
                _LOGGER.debug("Using default voice: %s", default.name)

        # Synthesize
        wav_bytes = await self.engine_mgr.synthesize(
            text=text,
            voice_conds_path=conds_path,
            audio_prompt_path=str(audio_path) if audio_path else None,
        )

        # Stream WAV audio back as Wyoming events
        with io.BytesIO(wav_bytes) as wav_io:
            wav_file: wave.Wave_read = wave.open(wav_io, "rb")
            with wav_file:
                rate = wav_file.getframerate()
                width = wav_file.getsampwidth()
                channels = wav_file.getnchannels()

                await self.write_event(
                    AudioStart(
                        rate=rate,
                        width=width,
                        channels=channels,
                    ).event()
                )

                audio_bytes = wav_file.readframes(wav_file.getnframes())
                bytes_per_sample = width * channels
                bytes_per_chunk = bytes_per_sample * SAMPLES_PER_CHUNK
                num_chunks = int(math.ceil(len(audio_bytes) / bytes_per_chunk))

                for i in range(num_chunks):
                    offset = i * bytes_per_chunk
                    chunk = audio_bytes[offset : offset + bytes_per_chunk]
                    await self.write_event(
                        AudioChunk(
                            audio=chunk,
                            rate=rate,
                            width=width,
                            channels=channels,
                        ).event()
                    )

                await self.write_event(AudioStop().event())

        _LOGGER.info("Synthesis complete for: '%s'", text[:50])
