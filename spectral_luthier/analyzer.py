"""
Stem Analyzer - Spectral and dynamic analysis of separated stems

Analyzes LUFS, spectral centroid, dynamic range, and other metrics
for comparison against target profiles.
"""

import numpy as np
import librosa
import pyloudnorm as pyln
from scipy import signal
from typing import Dict, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class StemAnalysis:
    """Analysis results for a single stem"""

    name: str
    lufs_integrated: float
    lufs_short_term: float
    lufs_momentary: float
    dynamic_range: float  # PLR (Program Loudness Range)
    spectral_centroid: float
    spectral_rolloff: float
    spectral_bandwidth: float
    rms_energy: float
    peak_level: float
    crest_factor: float  # Peak-to-RMS ratio
    zero_crossing_rate: float

    def __repr__(self) -> str:
        return (
            f"StemAnalysis(name={self.name}, "
            f"LUFS-I={self.lufs_integrated:.1f}, "
            f"DR={self.dynamic_range:.1f}, "
            f"SpectralCentroid={self.spectral_centroid:.0f}Hz)"
        )


class StemAnalyzer:
    """Analyzes audio stems for spectral and dynamic characteristics"""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.meter = pyln.Meter(sample_rate)  # LUFS meter

    def analyze_stem(
        self, audio: np.ndarray, stem_name: str
    ) -> StemAnalysis:
        """
        Analyze a single stem

        Args:
            audio: Audio data (samples, channels) or (samples,) for mono
            stem_name: Name of the stem (e.g., 'drums', 'bass')

        Returns:
            StemAnalysis object with all metrics
        """
        logger.info(f"Analyzing stem: {stem_name}")

        # Ensure stereo format for LUFS measurement
        if audio.ndim == 1:
            audio_stereo = np.stack([audio, audio], axis=1)
        else:
            audio_stereo = audio

        # Convert to mono for spectral analysis
        if audio.ndim == 2:
            audio_mono = np.mean(audio, axis=1)
        else:
            audio_mono = audio

        # === LOUDNESS ANALYSIS ===
        lufs_integrated = self.meter.integrated_loudness(audio_stereo)

        # Short-term loudness (3 second window)
        lufs_short_term = self._calculate_short_term_lufs(audio_stereo)

        # Momentary loudness (400ms window)
        lufs_momentary = self._calculate_momentary_lufs(audio_stereo)

        # Dynamic range (PLR)
        dynamic_range = self._calculate_dynamic_range(audio_stereo)

        # === SPECTRAL ANALYSIS ===
        spectral_centroid = self._calculate_spectral_centroid(audio_mono)
        spectral_rolloff = self._calculate_spectral_rolloff(audio_mono)
        spectral_bandwidth = self._calculate_spectral_bandwidth(audio_mono)

        # === TIME-DOMAIN ANALYSIS ===
        rms_energy = np.sqrt(np.mean(audio_mono ** 2))
        peak_level = np.max(np.abs(audio_mono))
        crest_factor = peak_level / (rms_energy + 1e-10)

        # Zero crossing rate (indicates noisiness/brightness)
        zero_crossing_rate = np.mean(
            librosa.feature.zero_crossing_rate(audio_mono)[0]
        )

        analysis = StemAnalysis(
            name=stem_name,
            lufs_integrated=lufs_integrated,
            lufs_short_term=lufs_short_term,
            lufs_momentary=lufs_momentary,
            dynamic_range=dynamic_range,
            spectral_centroid=spectral_centroid,
            spectral_rolloff=spectral_rolloff,
            spectral_bandwidth=spectral_bandwidth,
            rms_energy=float(rms_energy),
            peak_level=float(peak_level),
            crest_factor=float(crest_factor),
            zero_crossing_rate=float(zero_crossing_rate),
        )

        logger.debug(f"Analysis complete: {analysis}")
        return analysis

    def analyze_all_stems(
        self, stems: Dict[str, np.ndarray]
    ) -> Dict[str, StemAnalysis]:
        """
        Analyze all stems in a dictionary

        Args:
            stems: Dictionary mapping stem names to audio data

        Returns:
            Dictionary mapping stem names to StemAnalysis objects
        """
        return {
            name: self.analyze_stem(audio, name)
            for name, audio in stems.items()
        }

    def _calculate_short_term_lufs(self, audio: np.ndarray) -> float:
        """Calculate short-term LUFS (3 second window, max)"""
        window_size = int(3.0 * self.sample_rate)  # 3 seconds
        hop_size = int(0.1 * self.sample_rate)  # 100ms hop

        max_lufs = -70.0
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i : i + window_size]
            try:
                lufs = self.meter.integrated_loudness(window)
                if lufs > max_lufs:
                    max_lufs = lufs
            except ValueError:
                # Too quiet or too short
                continue

        return max_lufs

    def _calculate_momentary_lufs(self, audio: np.ndarray) -> float:
        """Calculate momentary LUFS (400ms window, max)"""
        window_size = int(0.4 * self.sample_rate)  # 400ms
        hop_size = int(0.1 * self.sample_rate)  # 100ms hop

        max_lufs = -70.0
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i : i + window_size]
            try:
                lufs = self.meter.integrated_loudness(window)
                if lufs > max_lufs:
                    max_lufs = lufs
            except ValueError:
                continue

        return max_lufs

    def _calculate_dynamic_range(self, audio: np.ndarray) -> float:
        """
        Calculate dynamic range (PLR - Program Loudness Range)

        This is similar to EBU R128 PLR measurement
        """
        # Calculate short-term loudness across entire track
        window_size = int(3.0 * self.sample_rate)
        hop_size = int(0.1 * self.sample_rate)

        loudness_values = []
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i : i + window_size]
            try:
                lufs = self.meter.integrated_loudness(window)
                loudness_values.append(lufs)
            except ValueError:
                continue

        if not loudness_values:
            return 0.0

        # Sort and remove top/bottom 10%
        loudness_values = sorted(loudness_values)
        trim_count = int(len(loudness_values) * 0.1)
        if trim_count > 0:
            loudness_values = loudness_values[trim_count:-trim_count]

        # Dynamic range = difference between max and min
        return float(max(loudness_values) - min(loudness_values))

    def _calculate_spectral_centroid(self, audio: np.ndarray) -> float:
        """Calculate spectral centroid (brightness) in Hz"""
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate
        )[0]
        return float(np.mean(centroid))

    def _calculate_spectral_rolloff(self, audio: np.ndarray) -> float:
        """Calculate spectral rolloff frequency (85% of energy)"""
        rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, roll_percent=0.85
        )[0]
        return float(np.mean(rolloff))

    def _calculate_spectral_bandwidth(self, audio: np.ndarray) -> float:
        """Calculate spectral bandwidth"""
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate
        )[0]
        return float(np.mean(bandwidth))

    def compare_to_target(
        self, analysis: StemAnalysis, target_lufs: float, target_centroid: float
    ) -> Dict[str, float]:
        """
        Compare analysis to target values

        Returns:
            Dictionary of deltas (current - target)
        """
        return {
            "lufs_delta": analysis.lufs_integrated - target_lufs,
            "centroid_delta": analysis.spectral_centroid - target_centroid,
        }
