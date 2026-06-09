"""
Stem Rebalancer - Corrective DSP chain for stem rebalancing

Applies EQ, compression, and gain adjustments to stems based on
analysis deltas from target profiles.
"""

import numpy as np
from pedalboard import (
    Pedalboard,
    Compressor,
    Gain,
    HighShelfFilter,
    LowShelfFilter,
    PeakFilter,
)
from typing import Dict, Optional
from dataclasses import dataclass
from loguru import logger

from .analyzer import StemAnalysis
from .profiles import StemTarget


@dataclass
class RebalanceConfig:
    """Configuration for stem rebalancing"""

    # EQ
    apply_eq: bool = True
    eq_q_factor: float = 0.707  # Butterworth default

    # Compression
    apply_compression: bool = True
    compressor_attack_ms: float = 5.0
    compressor_release_ms: float = 50.0
    compressor_knee_db: float = 6.0

    # Gain
    apply_gain_adjustment: bool = True

    # Safety
    max_eq_gain: float = 12.0  # dB
    max_gain_adjustment: float = 12.0  # dB
    max_compression_ratio: float = 10.0


class StemRebalancer:
    """Applies corrective DSP to rebalance stems"""

    def __init__(
        self,
        sample_rate: int = 44100,
        config: Optional[RebalanceConfig] = None,
    ):
        self.sample_rate = sample_rate
        self.config = config or RebalanceConfig()

    def rebalance_stem(
        self,
        audio: np.ndarray,
        analysis: StemAnalysis,
        target: StemTarget,
        balance_gain_db: float = 0.0,
    ) -> np.ndarray:
        """
        Apply corrective DSP to a stem

        Args:
            audio: Input audio (samples, channels) or (samples,)
            analysis: Current analysis of the stem
            target: Target characteristics
            balance_gain_db: Additional gain for stem balance

        Returns:
            Processed audio
        """
        logger.info(f"Rebalancing stem: {analysis.name}")

        # Build DSP chain
        board = Pedalboard()

        # === 1. GAIN ADJUSTMENT (LUFS matching) ===
        if self.config.apply_gain_adjustment:
            lufs_delta = analysis.lufs_integrated - target.lufs_integrated
            gain_adjustment = -lufs_delta + balance_gain_db

            # Clamp to safety limits
            gain_adjustment = np.clip(
                gain_adjustment,
                -self.config.max_gain_adjustment,
                self.config.max_gain_adjustment,
            )

            if abs(gain_adjustment) > 0.1:
                logger.debug(
                    f"Applying gain adjustment: {gain_adjustment:.2f} dB"
                )
                board.append(Gain(gain_db=float(gain_adjustment)))

        # === 2. COMPRESSION (Dynamic range control) ===
        if self.config.apply_compression and target.compression_ratio:
            ratio = min(target.compression_ratio, self.config.max_compression_ratio)
            threshold_db = target.compression_threshold or -20.0

            logger.debug(
                f"Applying compression: ratio={ratio:.1f}:1, "
                f"threshold={threshold_db:.1f} dBFS"
            )

            board.append(
                Compressor(
                    threshold_db=float(threshold_db),
                    ratio=float(ratio),
                    attack_ms=self.config.compressor_attack_ms,
                    release_ms=self.config.compressor_release_ms,
                )
            )

        # === 3. EQ (Spectral shaping) ===
        if self.config.apply_eq and target.eq_curve:
            eq_filters = self._build_eq_chain(target.eq_curve)
            for eq_filter in eq_filters:
                board.append(eq_filter)

        # Process audio
        if len(board) > 0:
            # Ensure correct shape for pedalboard
            if audio.ndim == 1:
                # Mono -> Stereo
                audio_stereo = np.stack([audio, audio], axis=0)
            else:
                # (samples, channels) -> (channels, samples)
                audio_stereo = audio.T

            # Process
            processed = board(audio_stereo, sample_rate=self.sample_rate)

            # Convert back to original shape
            if audio.ndim == 1:
                processed = processed[0]  # Take first channel
            else:
                processed = processed.T

            logger.info(f"Rebalancing complete: {analysis.name}")
            return processed
        else:
            logger.info(f"No processing needed: {analysis.name}")
            return audio

    def rebalance_all_stems(
        self,
        stems: Dict[str, np.ndarray],
        analyses: Dict[str, StemAnalysis],
        targets: Dict[str, StemTarget],
        balance_gains: Dict[str, float],
    ) -> Dict[str, np.ndarray]:
        """
        Rebalance all stems

        Args:
            stems: Dictionary of stem audio
            analyses: Dictionary of stem analyses
            targets: Dictionary of stem targets
            balance_gains: Dictionary of balance gain adjustments

        Returns:
            Dictionary of processed stems
        """
        processed_stems = {}

        for stem_name, audio in stems.items():
            analysis = analyses[stem_name]
            target = targets[stem_name]
            balance_gain = balance_gains.get(stem_name, 0.0)

            processed = self.rebalance_stem(
                audio, analysis, target, balance_gain
            )
            processed_stems[stem_name] = processed

        return processed_stems

    def _build_eq_chain(
        self, eq_curve: Dict[str, float]
    ) -> list:
        """
        Build EQ filter chain from frequency-gain pairs

        Args:
            eq_curve: Dictionary mapping frequency (Hz as string) to gain (dB)

        Returns:
            List of pedalboard filters
        """
        filters = []

        # Sort by frequency
        sorted_curve = sorted(
            [(float(freq), gain) for freq, gain in eq_curve.items()]
        )

        for freq_hz, gain_db in sorted_curve:
            # Clamp gain to safety limits
            gain_db = np.clip(
                gain_db, -self.config.max_eq_gain, self.config.max_eq_gain
            )

            if abs(gain_db) < 0.1:
                continue  # Skip tiny adjustments

            logger.debug(f"EQ: {freq_hz:.0f} Hz -> {gain_db:+.1f} dB")

            # Choose filter type based on frequency
            if freq_hz < 100:
                # Low shelf for bass
                filters.append(
                    LowShelfFilter(
                        cutoff_frequency_hz=float(freq_hz),
                        gain_db=float(gain_db),
                        q=self.config.eq_q_factor,
                    )
                )
            elif freq_hz > 8000:
                # High shelf for air
                filters.append(
                    HighShelfFilter(
                        cutoff_frequency_hz=float(freq_hz),
                        gain_db=float(gain_db),
                        q=self.config.eq_q_factor,
                    )
                )
            else:
                # Parametric peak for mids
                filters.append(
                    PeakFilter(
                        cutoff_frequency_hz=float(freq_hz),
                        gain_db=float(gain_db),
                        q=self.config.eq_q_factor,
                    )
                )

        return filters

    def calculate_makeup_gain(
        self, analysis: StemAnalysis, target: StemTarget
    ) -> float:
        """
        Calculate makeup gain needed after compression

        Returns:
            Makeup gain in dB
        """
        # Simple heuristic: harder compression = more makeup gain
        if target.compression_ratio and target.compression_ratio > 2.0:
            # Approximate gain reduction
            dynamic_diff = analysis.dynamic_range - target.dynamic_range
            if dynamic_diff > 0:
                # We're reducing dynamic range
                makeup = dynamic_diff * 0.5  # Conservative makeup
                return min(makeup, 6.0)  # Cap at +6dB

        return 0.0
