"""
Target Profile System - Genre-specific mastering targets

Defines the spectral and dynamic characteristics for different musical genres.
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field
import json
from pathlib import Path


class StemTarget(BaseModel):
    """Target characteristics for a single stem (Drums, Bass, Vocals, Other)"""

    lufs_integrated: float = Field(
        description="Target LUFS-I (Integrated Loudness)"
    )
    dynamic_range: float = Field(
        description="Target dynamic range in dB (PLR - Program Loudness Range)"
    )
    spectral_centroid: float = Field(
        description="Target spectral centroid in Hz (brightness)"
    )
    eq_curve: Dict[str, float] = Field(
        default_factory=dict,
        description="Target EQ curve {freq_hz: gain_db}"
    )
    compression_ratio: Optional[float] = Field(
        default=None,
        description="Target compression ratio (e.g., 3.0 for 3:1)"
    )
    compression_threshold: Optional[float] = Field(
        default=None,
        description="Compression threshold in dBFS"
    )


class TargetProfile(BaseModel):
    """Complete mastering profile for a genre"""

    name: str = Field(description="Profile name (e.g., 'Modern_Metal')")
    description: str = Field(description="Profile description")

    # Final master targets
    final_lufs_integrated: float = Field(
        default=-14.0,
        description="Final master LUFS-I target"
    )
    final_lufs_short_term: float = Field(
        default=-11.0,
        description="Final master LUFS-S target"
    )
    final_true_peak: float = Field(
        default=-1.0,
        description="Final true peak ceiling in dBTP"
    )

    # Per-stem targets
    drums: StemTarget
    bass: StemTarget
    vocals: StemTarget
    other: StemTarget

    # Stem balance (relative levels)
    stem_balance: Dict[str, float] = Field(
        description="Relative stem levels in dB (0 = reference)"
    )

    @classmethod
    def from_json(cls, path: Path) -> "TargetProfile":
        """Load profile from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: Path) -> None:
        """Save profile to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)

    def get_stem_target(self, stem_name: str) -> StemTarget:
        """Get target for a specific stem"""
        return getattr(self, stem_name.lower())


# Predefined profiles
MODERN_METAL_PROFILE = TargetProfile(
    name="Modern_Metal",
    description="Aggressive, punchy metal mastering with tight low end",
    final_lufs_integrated=-8.0,  # Very loud
    final_lufs_short_term=-6.0,
    final_true_peak=-0.5,
    drums=StemTarget(
        lufs_integrated=-12.0,
        dynamic_range=6.0,  # Compressed
        spectral_centroid=3500.0,  # Bright, snappy
        eq_curve={
            "60": 2.0,    # Kick thump
            "200": -1.5,  # Reduce mud
            "3000": 3.0,  # Snare crack
            "8000": 2.0,  # Cymbal air
        },
        compression_ratio=6.0,
        compression_threshold=-18.0,
    ),
    bass=StemTarget(
        lufs_integrated=-14.0,
        dynamic_range=5.0,
        spectral_centroid=200.0,  # Deep and tight
        eq_curve={
            "50": 1.0,
            "100": 2.0,   # Bass fundamental
            "500": -2.0,  # Cut boxiness
        },
        compression_ratio=4.0,
        compression_threshold=-20.0,
    ),
    vocals=StemTarget(
        lufs_integrated=-10.0,  # Loud and upfront
        dynamic_range=8.0,
        spectral_centroid=2000.0,
        eq_curve={
            "200": -2.0,  # Reduce proximity
            "1000": 1.0,  # Presence
            "4000": 2.5,  # Clarity
            "10000": 1.5, # Air
        },
        compression_ratio=5.0,
        compression_threshold=-15.0,
    ),
    other=StemTarget(
        lufs_integrated=-16.0,  # Guitars, synths
        dynamic_range=7.0,
        spectral_centroid=1500.0,
        eq_curve={
            "250": -1.0,
            "2000": 1.0,
            "5000": 0.5,
        },
        compression_ratio=3.0,
        compression_threshold=-18.0,
    ),
    stem_balance={
        "drums": 0.0,    # Reference
        "bass": -2.0,    # Slightly quieter
        "vocals": 2.0,   # Louder
        "other": -1.0,   # Background
    }
)


CINEMATIC_POP_PROFILE = TargetProfile(
    name="Cinematic_Pop",
    description="Wide, dynamic pop mastering with lush vocals",
    final_lufs_integrated=-11.0,  # Moderate loudness
    final_lufs_short_term=-9.0,
    final_true_peak=-1.0,
    drums=StemTarget(
        lufs_integrated=-14.0,
        dynamic_range=10.0,  # More dynamic
        spectral_centroid=2500.0,
        eq_curve={
            "80": 1.0,
            "200": -0.5,
            "3000": 1.5,
            "10000": 2.0,  # Airy
        },
        compression_ratio=3.0,
        compression_threshold=-20.0,
    ),
    bass=StemTarget(
        lufs_integrated=-16.0,
        dynamic_range=8.0,
        spectral_centroid=150.0,  # Deep
        eq_curve={
            "40": 0.5,
            "80": 1.5,
            "200": -1.0,
        },
        compression_ratio=3.0,
        compression_threshold=-22.0,
    ),
    vocals=StemTarget(
        lufs_integrated=-9.0,   # Very upfront
        dynamic_range=12.0,     # Natural dynamics
        spectral_centroid=2500.0,
        eq_curve={
            "150": -1.5,
            "1000": 0.5,
            "3000": 2.0,  # Presence
            "8000": 1.5,  # Silk
            "12000": 1.0, # Air
        },
        compression_ratio=2.5,
        compression_threshold=-18.0,
    ),
    other=StemTarget(
        lufs_integrated=-18.0,
        dynamic_range=10.0,
        spectral_centroid=1000.0,
        eq_curve={
            "500": 0.5,
            "2000": 0.0,
            "8000": 1.0,
        },
        compression_ratio=2.0,
        compression_threshold=-20.0,
    ),
    stem_balance={
        "drums": -1.0,
        "bass": -3.0,
        "vocals": 3.0,   # Lead vocal focus
        "other": -2.0,
    }
)


PROFILE_REGISTRY = {
    "Modern_Metal": MODERN_METAL_PROFILE,
    "Cinematic_Pop": CINEMATIC_POP_PROFILE,
}


def get_profile(name: str) -> TargetProfile:
    """Get a predefined profile by name"""
    if name not in PROFILE_REGISTRY:
        raise ValueError(
            f"Profile '{name}' not found. "
            f"Available profiles: {list(PROFILE_REGISTRY.keys())}"
        )
    return PROFILE_REGISTRY[name]


def list_profiles() -> list[str]:
    """List all available profile names"""
    return list(PROFILE_REGISTRY.keys())
