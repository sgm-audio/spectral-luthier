"""
Transparent Limiter - Final stage mastering limiter

True-peak limiting with minimal phase artifacts and transparent loudness maximization.
"""

import numpy as np
from pedalboard import Pedalboard, Limiter, Gain
from typing import Optional
from loguru import logger
import pyloudnorm as pyln


class TransparentLimiter:
    """
    Transparent true-peak limiter for final mastering stage

    Uses lookahead limiting with soft-knee to maximize loudness
    while avoiding clipping and minimizing artifacts.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        target_lufs: float = -14.0,
        true_peak_ceiling: float = -1.0,
        release_ms: float = 100.0,
    ):
        """
        Args:
            sample_rate: Audio sample rate
            target_lufs: Target integrated LUFS
            true_peak_ceiling: True peak ceiling in dBTP
            release_ms: Limiter release time in milliseconds
        """
        self.sample_rate = sample_rate
        self.target_lufs = target_lufs
        self.true_peak_ceiling = true_peak_ceiling
        self.release_ms = release_ms

        self.meter = pyln.Meter(sample_rate)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply transparent limiting to reach target loudness

        Args:
            audio: Stereo audio (samples, channels) or (samples,)

        Returns:
            Limited audio at target loudness
        """
        logger.info(
            f"Applying transparent limiter: "
            f"target={self.target_lufs:.1f} LUFS, "
            f"ceiling={self.true_peak_ceiling:.1f} dBTP"
        )

        # Ensure stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=1)

        # Measure current loudness
        current_lufs = self.meter.integrated_loudness(audio)
        logger.debug(f"Current LUFS: {current_lufs:.2f}")

        # Calculate required gain
        gain_db = self.target_lufs - current_lufs
        logger.debug(f"Required gain: {gain_db:+.2f} dB")

        # Build processing chain
        board = Pedalboard()

        # 1. Apply gain staging to approach target
        if abs(gain_db) > 0.1:
            board.append(Gain(gain_db=float(gain_db)))

        # 2. Apply true-peak limiter
        # Convert dBTP to linear threshold
        threshold_linear = 10 ** (self.true_peak_ceiling / 20.0)

        board.append(
            Limiter(
                threshold_db=float(self.true_peak_ceiling),
                release_ms=self.release_ms,
            )
        )

        # Process (pedalboard expects (channels, samples))
        audio_T = audio.T
        processed = board(audio_T, sample_rate=self.sample_rate)
        processed = processed.T

        # Verify true peak
        true_peak_db = self._measure_true_peak(processed)
        logger.debug(f"True peak after limiting: {true_peak_db:.2f} dBTP")

        # Verify final loudness
        final_lufs = self.meter.integrated_loudness(processed)
        logger.info(f"Final LUFS: {final_lufs:.2f} (target: {self.target_lufs:.1f})")

        # Safety check
        if true_peak_db > 0.0:
            logger.warning(
                f"True peak exceeded 0 dBFS: {true_peak_db:.2f} dBTP. "
                f"Applying safety gain reduction."
            )
            safety_gain = -true_peak_db - 0.1  # Extra 0.1dB headroom
            processed = processed * (10 ** (safety_gain / 20.0))

        return processed

    def _measure_true_peak(self, audio: np.ndarray) -> float:
        """
        Measure true peak level in dBTP

        True peak uses 4x oversampling to detect inter-sample peaks
        """
        from scipy import signal as sp_signal

        # Upsample by 4x
        upsampled = sp_signal.resample(audio, len(audio) * 4, axis=0)

        # Find peak
        peak = np.max(np.abs(upsampled))

        # Convert to dBTP
        if peak > 0:
            return float(20 * np.log10(peak))
        else:
            return -np.inf

    def process_with_multiband(
        self, audio: np.ndarray, bands: int = 3
    ) -> np.ndarray:
        """
        Advanced: Apply multiband limiting for even more transparency

        Args:
            audio: Input audio
            bands: Number of frequency bands (2-5)

        Returns:
            Processed audio
        """
        logger.info(f"Applying {bands}-band multiband limiter")

        # Split into frequency bands
        band_audios = self._split_bands(audio, bands)

        # Process each band
        processed_bands = []
        for i, band_audio in enumerate(band_audios):
            logger.debug(f"Processing band {i+1}/{bands}")

            # Create limiter for this band
            band_limiter = TransparentLimiter(
                sample_rate=self.sample_rate,
                target_lufs=self.target_lufs,
                true_peak_ceiling=self.true_peak_ceiling,
                release_ms=self.release_ms * (1 + i * 0.2),  # Vary release
            )

            processed = band_limiter.process(band_audio)
            processed_bands.append(processed)

        # Recombine bands
        result = self._combine_bands(processed_bands)
        return result

    def _split_bands(
        self, audio: np.ndarray, num_bands: int
    ) -> list[np.ndarray]:
        """Split audio into frequency bands using Linkwitz-Riley crossovers"""
        from scipy import signal as sp_signal

        if num_bands == 2:
            crossover_freqs = [500]
        elif num_bands == 3:
            crossover_freqs = [250, 2500]
        elif num_bands == 4:
            crossover_freqs = [150, 800, 4000]
        else:  # 5 bands
            crossover_freqs = [100, 500, 2000, 8000]

        bands = []

        # Low band
        b, a = sp_signal.butter(
            4, crossover_freqs[0], btype='low', fs=self.sample_rate
        )
        low = sp_signal.filtfilt(b, a, audio, axis=0)
        bands.append(low)

        # Mid bands
        for i in range(len(crossover_freqs) - 1):
            b, a = sp_signal.butter(
                4,
                [crossover_freqs[i], crossover_freqs[i + 1]],
                btype='band',
                fs=self.sample_rate,
            )
            mid = sp_signal.filtfilt(b, a, audio, axis=0)
            bands.append(mid)

        # High band
        b, a = sp_signal.butter(
            4, crossover_freqs[-1], btype='high', fs=self.sample_rate
        )
        high = sp_signal.filtfilt(b, a, audio, axis=0)
        bands.append(high)

        return bands

    def _combine_bands(self, bands: list[np.ndarray]) -> np.ndarray:
        """Recombine frequency bands"""
        return sum(bands)
