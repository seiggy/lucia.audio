"""Voice profile manager.

Handles CRUD operations for voice profiles: upload audio, compute+cache conditionals,
list, delete. Each profile is stored as a directory under the voices root.
"""

import json
import logging
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_LOGGER = logging.getLogger(__name__)

PROFILE_META = "profile.json"
AUDIO_FILE = "reference.wav"
CONDS_FILE = "conds.pt"


@dataclass
class VoiceProfile:
    id: str
    name: str
    created_at: float
    audio_file: str  # path relative to profile dir
    conds_file: Optional[str] = None  # path relative to profile dir
    description: str = ""
    is_ready: bool = False  # True once conditionals are computed

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "VoiceProfile":
        return VoiceProfile(**{k: v for k, v in data.items() if k in VoiceProfile.__dataclass_fields__})


class VoiceManager:
    """Manages voice profiles on disk."""

    def __init__(self, voices_dir: str | Path):
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self._profiles: Dict[str, VoiceProfile] = {}
        self._load_profiles()

    def _load_profiles(self) -> None:
        """Load all profiles from disk."""
        for profile_dir in self.voices_dir.iterdir():
            if not profile_dir.is_dir():
                continue
            meta_path = profile_dir / PROFILE_META
            if not meta_path.exists():
                continue
            try:
                with open(meta_path, "r") as f:
                    data = json.load(f)
                profile = VoiceProfile.from_dict(data)
                # Verify conditionals exist
                conds_path = profile_dir / CONDS_FILE
                profile.is_ready = conds_path.exists()
                if profile.is_ready:
                    profile.conds_file = CONDS_FILE
                self._profiles[profile.id] = profile
                _LOGGER.info("Loaded voice profile: %s (%s)", profile.name, profile.id)
            except Exception as e:
                _LOGGER.warning("Failed to load profile from %s: %s", profile_dir, e)

    def list_profiles(self) -> List[VoiceProfile]:
        """Return all voice profiles sorted by name."""
        return sorted(self._profiles.values(), key=lambda p: p.name)

    def get_profile(self, profile_id: str) -> Optional[VoiceProfile]:
        """Get a profile by ID."""
        return self._profiles.get(profile_id)

    def get_profile_by_name(self, name: str) -> Optional[VoiceProfile]:
        """Get a profile by name (case-insensitive)."""
        name_lower = name.lower()
        for profile in self._profiles.values():
            if profile.name.lower() == name_lower:
                return profile
        return None

    def get_profile_dir(self, profile_id: str) -> Path:
        return self.voices_dir / profile_id

    def get_conds_path(self, profile_id: str) -> Optional[Path]:
        """Get the path to pre-computed conditionals for a profile."""
        profile = self._profiles.get(profile_id)
        if profile and profile.is_ready:
            return self.get_profile_dir(profile_id) / CONDS_FILE
        return None

    def get_audio_path(self, profile_id: str) -> Optional[Path]:
        """Get the path to the reference audio for a profile."""
        profile = self._profiles.get(profile_id)
        if profile:
            return self.get_profile_dir(profile_id) / profile.audio_file
        return None

    def create_profile(self, name: str, audio_data: bytes, description: str = "") -> VoiceProfile:
        """Create a new voice profile from uploaded audio data.

        The audio clip is saved to disk. Conditionals must be computed separately.
        """
        profile_id = str(uuid.uuid4())[:8]
        profile_dir = self.get_profile_dir(profile_id)
        profile_dir.mkdir(parents=True, exist_ok=True)

        # Save audio file
        audio_path = profile_dir / AUDIO_FILE
        with open(audio_path, "wb") as f:
            f.write(audio_data)

        profile = VoiceProfile(
            id=profile_id,
            name=name,
            created_at=time.time(),
            audio_file=AUDIO_FILE,
            description=description,
            is_ready=False,
        )

        # Save metadata
        self._save_profile_meta(profile)
        self._profiles[profile.id] = profile

        _LOGGER.info("Created voice profile: %s (%s)", name, profile_id)
        return profile

    def mark_ready(self, profile_id: str) -> None:
        """Mark a profile as ready (conditionals computed)."""
        profile = self._profiles.get(profile_id)
        if profile:
            profile.is_ready = True
            profile.conds_file = CONDS_FILE
            self._save_profile_meta(profile)

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a voice profile."""
        profile = self._profiles.pop(profile_id, None)
        if profile is None:
            return False

        profile_dir = self.get_profile_dir(profile_id)
        if profile_dir.exists():
            shutil.rmtree(profile_dir)

        _LOGGER.info("Deleted voice profile: %s (%s)", profile.name, profile_id)
        return True

    def update_profile(self, profile_id: str, name: Optional[str] = None, description: Optional[str] = None) -> Optional[VoiceProfile]:
        """Update profile metadata."""
        profile = self._profiles.get(profile_id)
        if profile is None:
            return None

        if name is not None:
            profile.name = name
        if description is not None:
            profile.description = description

        self._save_profile_meta(profile)
        return profile

    def _save_profile_meta(self, profile: VoiceProfile) -> None:
        profile_dir = self.get_profile_dir(profile.id)
        meta_path = profile_dir / PROFILE_META
        with open(meta_path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)

    def get_default_voice(self) -> Optional[VoiceProfile]:
        """Get the first ready profile, or None."""
        for profile in self._profiles.values():
            if profile.is_ready:
                return profile
        return None
