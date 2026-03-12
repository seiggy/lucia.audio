"""Engine manager — routes TTS requests to the active engine."""

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


class EngineManager:
    """Manages multiple TTS engines and routes synthesis requests."""

    def __init__(self, device: str = "cuda", half_precision: bool = True, qwen3_model: str = ""):
        self.device = device
        self.engines: Dict[str, object] = {}
        self._default_engine_id = ChatterboxEngine.ENGINE_ID

        # Always create Chatterbox
        self.engines[ChatterboxEngine.ENGINE_ID] = ChatterboxEngine(
            device=device, half_precision=half_precision,
        )

        # Optionally create Qwen3
        if qwen3_model:
            from .qwen3_engine import QWEN3_MODEL_0_6B
            self.engines[Qwen3Engine.ENGINE_ID] = Qwen3Engine(
                device=device, model_name=qwen3_model,
            )

    async def load_all(self) -> None:
        """Load all configured engines."""
        for engine_id, engine in self.engines.items():
            try:
                _LOGGER.info("Loading engine: %s", engine_id)
                await engine.load_model()
                _LOGGER.info("Engine loaded: %s", engine_id)
            except Exception as e:
                _LOGGER.error("Failed to load engine %s: %s", engine_id, e)

    def get_engine(self, engine_id: Optional[str] = None):
        """Get an engine by ID, falling back to default."""
        if engine_id and engine_id in self.engines:
            return self.engines[engine_id]
        return self.engines.get(self._default_engine_id)

    def get_chatterbox(self) -> Optional[ChatterboxEngine]:
        return self.engines.get(ChatterboxEngine.ENGINE_ID)

    def get_qwen3(self) -> Optional[Qwen3Engine]:
        return self.engines.get(Qwen3Engine.ENGINE_ID)

    def list_engines(self) -> list:
        """Return info about available engines."""
        result = []
        for engine_id, engine in self.engines.items():
            result.append({
                "id": engine_id,
                "name": engine.ENGINE_NAME,
                "loaded": engine._model is not None,
            })
        return result

    @property
    def default_engine_id(self) -> str:
        return self._default_engine_id

    @default_engine_id.setter
    def default_engine_id(self, engine_id: str):
        if engine_id in self.engines:
            self._default_engine_id = engine_id

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
        """Synthesize using the specified or default engine."""
        engine = self.get_engine(engine_id)
        if engine is None:
            raise RuntimeError(f"No engine available (requested: {engine_id})")

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
