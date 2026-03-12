"""Engine manager — routes TTS requests to the active engine."""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional

from .tts_engine import ChatterboxEngine
from .qwen3_engine import Qwen3Engine

_LOGGER = logging.getLogger(__name__)

ENGINES = {
    ChatterboxEngine.ENGINE_ID: ChatterboxEngine,
    Qwen3Engine.ENGINE_ID: Qwen3Engine,
}

# Engine states
STATE_UNLOADED = "unloaded"
STATE_LOADING = "loading"
STATE_READY = "ready"
STATE_UNLOADING = "unloading"
STATE_ERROR = "error"


class EngineManager:
    """Manages multiple TTS engines and routes synthesis requests."""

    def __init__(self, device: str = "cuda", half_precision: bool = True, qwen3_model: str = ""):
        self.device = device
        self.engines: Dict[str, object] = {}
        self._active_engine_id = ChatterboxEngine.ENGINE_ID
        self._engine_states: Dict[str, str] = {}
        self._swap_lock = asyncio.Lock()

        # Always create Chatterbox
        self.engines[ChatterboxEngine.ENGINE_ID] = ChatterboxEngine(
            device=device, half_precision=half_precision,
        )
        self._engine_states[ChatterboxEngine.ENGINE_ID] = STATE_UNLOADED

        # Optionally create Qwen3
        if qwen3_model:
            self.engines[Qwen3Engine.ENGINE_ID] = Qwen3Engine(
                device=device, model_name=qwen3_model,
            )
            self._engine_states[Qwen3Engine.ENGINE_ID] = STATE_UNLOADED

    async def load_all(self) -> None:
        """Load all configured engines."""
        for engine_id, engine in self.engines.items():
            try:
                self._engine_states[engine_id] = STATE_LOADING
                _LOGGER.info("Loading engine: %s", engine_id)
                await engine.load_model()
                self._engine_states[engine_id] = STATE_READY
                _LOGGER.info("Engine loaded: %s", engine_id)
            except Exception as e:
                self._engine_states[engine_id] = STATE_ERROR
                _LOGGER.error("Failed to load engine %s: %s", engine_id, e)

    async def activate_engine(self, engine_id: str) -> None:
        """Switch the active server engine. Unloads current, loads target."""
        if engine_id not in self.engines:
            raise ValueError(f"Unknown engine: {engine_id}")
        if engine_id == self._active_engine_id and self._engine_states.get(engine_id) == STATE_READY:
            return  # Already active

        async with self._swap_lock:
            # Unload current active engine
            old_id = self._active_engine_id
            old_engine = self.engines.get(old_id)
            if old_engine and self._engine_states.get(old_id) == STATE_READY:
                _LOGGER.info("Unloading engine: %s", old_id)
                self._engine_states[old_id] = STATE_UNLOADING
                try:
                    await old_engine.unload_model()
                    self._engine_states[old_id] = STATE_UNLOADED
                except Exception as e:
                    _LOGGER.error("Error unloading %s: %s", old_id, e)
                    self._engine_states[old_id] = STATE_ERROR

            # Load new engine
            new_engine = self.engines[engine_id]
            _LOGGER.info("Loading engine: %s", engine_id)
            self._engine_states[engine_id] = STATE_LOADING
            try:
                await new_engine.load_model()
                self._engine_states[engine_id] = STATE_READY
                self._active_engine_id = engine_id
                _LOGGER.info("Engine activated: %s", engine_id)
            except Exception as e:
                self._engine_states[engine_id] = STATE_ERROR
                _LOGGER.error("Failed to load %s: %s", engine_id, e)
                raise

    def get_engine(self, engine_id: Optional[str] = None):
        """Get an engine by ID, falling back to active."""
        eid = engine_id if engine_id and engine_id in self.engines else self._active_engine_id
        engine = self.engines.get(eid)
        if engine and engine._model is not None:
            return engine
        return None

    def get_chatterbox(self) -> Optional[ChatterboxEngine]:
        return self.engines.get(ChatterboxEngine.ENGINE_ID)

    def get_qwen3(self) -> Optional[Qwen3Engine]:
        return self.engines.get(Qwen3Engine.ENGINE_ID)

    @property
    def active_engine_id(self) -> str:
        return self._active_engine_id

    def get_status(self) -> dict:
        """Full server status for the dashboard."""
        engines_status = []
        for engine_id, engine in self.engines.items():
            engines_status.append({
                "id": engine_id,
                "name": engine.ENGINE_NAME,
                "state": self._engine_states.get(engine_id, STATE_UNLOADED),
                "active": engine_id == self._active_engine_id,
            })
        return {
            "active_engine": self._active_engine_id,
            "engines": engines_status,
            "swapping": self._swap_lock.locked(),
        }

    def list_engines(self) -> list:
        """Return info about available engines."""
        result = []
        for engine_id, engine in self.engines.items():
            result.append({
                "id": engine_id,
                "name": engine.ENGINE_NAME,
                "state": self._engine_states.get(engine_id, STATE_UNLOADED),
                "active": engine_id == self._active_engine_id,
                "loaded": engine._model is not None,
            })
        return result

    async def synthesize(
        self,
        text: str,
        engine_id: Optional[str] = None,
        voice_conds_path: Optional[Path] = None,
        audio_prompt_path: Optional[str] = None,
        ref_text: str = "",
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 1000,
        repetition_penalty: float = 1.2,
    ) -> bytes:
        """Synthesize using the active engine (or specified engine if loaded)."""
        engine = self.get_engine(engine_id)
        if engine is None:
            raise RuntimeError(f"No engine available (active: {self._active_engine_id})")

        return await engine.synthesize(
            text=text,
            voice_conds_path=voice_conds_path,
            audio_prompt_path=audio_prompt_path,
            ref_text=ref_text,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
