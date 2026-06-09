"""
Stem Master - Main orchestrator for stem-aware spectral mastering

Coordinates source separation, analysis, rebalancing, and final limiting.
"""

import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from typing import Optional, Dict
from loguru import logger

from .analyzer import StemAnalyzer
from .rebalancer import StemRebalancer, RebalanceConfig
from .limiter import TransparentLimiter
from .profiles import TargetProfile, get_profile


class StemMaster:
    """
    Main class for stem-aware spectral mastering

    Pipeline:
    1. Load stereo WAV file
    2. Source separation (Demucs) -> 4 stems (drums, bass, vocals, other)
    3. Analyze each stem (LUFS, spectral centroid, dynamic range)
    4. Compare to target profile
    5. Apply corrective DSP (EQ, compression, gain)
    6. Remix stems with balanced levels
    7. Apply final transparent limiter
    8. Export master
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        device: str = "auto",
        demucs_model: str = "htdemucs",
    ):
        """
        Args:
            sample_rate: Target sample rate (resampling if needed)
            device: 'cuda', 'cpu', or 'auto' for automatic selection
            demucs_model: Demucs model name ('htdemucs', 'htdemucs_ft', 'mdx_extra')
        """
        self.sample_rate = sample_rate
        self.demucs_model_name = demucs_model

        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"StemMaster initialized on device: {self.device}")

        # Initialize components
        self.analyzer = StemAnalyzer(sample_rate=sample_rate)
        self.rebalancer = StemRebalancer(sample_rate=sample_rate)

        # Lazy-load Demucs (heavy model)
        self._separator = None

    def process(
        self,
        input_path: str,
        profile: str = "Modern_Metal",
        output_path: str = "master.wav",
        multiband_limiting: bool = False,
    ) -> Dict[str, any]:
        """
        Complete stem-aware mastering pipeline

        Args:
            input_path: Path to input stereo WAV file
            profile: Profile name from profiles.py (e.g., 'Modern_Metal')
            output_path: Path for output master WAV
            multiband_limiting: Use multiband limiter (slower, more transparent)

        Returns:
            Dictionary with processing metadata
        """
        logger.info("=" * 60)
        logger.info("SPECTRAL NEURAL LUTHIER - Stem-Aware Mastering")
        logger.info("=" * 60)

        # Load profile
        target_profile = get_profile(profile)
        logger.info(f"Profile: {target_profile.name}")
        logger.info(f"Description: {target_profile.description}")

        # === STEP 1: LOAD AUDIO ===
        logger.info("\n[1/7] Loading audio...")
        audio, sr = sf.read(input_path, always_2d=True)
        logger.info(f"Loaded: {input_path}")
        logger.info(f"Shape: {audio.shape}, Sample rate: {sr} Hz")

        # Resample if needed
        if sr != self.sample_rate:
            logger.info(f"Resampling to {self.sample_rate} Hz...")
            audio = self._resample(audio, sr, self.sample_rate)
            sr = self.sample_rate

        # === STEP 2: SOURCE SEPARATION ===
        logger.info("\n[2/7] Separating sources (this may take a moment)...")
        stems = self._separate_stems(audio)
        logger.info(f"Separated into {len(stems)} stems: {list(stems.keys())}")

        # === STEP 3: ANALYZE STEMS ===
        logger.info("\n[3/7] Analyzing stems...")
        analyses = self.analyzer.analyze_all_stems(stems)

        for name, analysis in analyses.items():
            logger.info(f"  {name}: LUFS-I={analysis.lufs_integrated:.1f}, "
                       f"DR={analysis.dynamic_range:.1f}, "
                       f"Centroid={analysis.spectral_centroid:.0f}Hz")

        # === STEP 4: BUILD TARGET MAP ===
        logger.info("\n[4/7] Building target map from profile...")
        targets = {
            "drums": target_profile.drums,
            "bass": target_profile.bass,
            "vocals": target_profile.vocals,
            "other": target_profile.other,
        }

        # === STEP 5: REBALANCE STEMS ===
        logger.info("\n[5/7] Rebalancing stems with corrective DSP...")
        processed_stems = self.rebalancer.rebalance_all_stems(
            stems=stems,
            analyses=analyses,
            targets=targets,
            balance_gains=target_profile.stem_balance,
        )

        # === STEP 6: REMIX STEMS ===
        logger.info("\n[6/7] Remixing stems...")
        master_mix = self._remix_stems(processed_stems)
        logger.info(f"Mixed shape: {master_mix.shape}")

        # === STEP 7: FINAL LIMITING ===
        logger.info("\n[7/7] Applying final transparent limiter...")
        limiter = TransparentLimiter(
            sample_rate=self.sample_rate,
            target_lufs=target_profile.final_lufs_integrated,
            true_peak_ceiling=target_profile.final_true_peak,
        )

        if multiband_limiting:
            final_master = limiter.process_with_multiband(master_mix, bands=3)
        else:
            final_master = limiter.process(master_mix)

        # === EXPORT ===
        logger.info(f"\nExporting master to: {output_path}")
        sf.write(output_path, final_master, self.sample_rate, subtype='PCM_24')

        logger.info("=" * 60)
        logger.info("MASTERING COMPLETE!")
        logger.info("=" * 60)

        # Return metadata
        return {
            "input_file": input_path,
            "output_file": output_path,
            "profile": target_profile.name,
            "stems_analyzed": list(stems.keys()),
            "final_lufs": target_profile.final_lufs_integrated,
            "analyses": {k: v.__dict__ for k, v in analyses.items()},
        }

    def _separate_stems(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Separate stereo audio into stems using Demucs

        Args:
            audio: Stereo audio (samples, 2)

        Returns:
            Dictionary with keys: 'drums', 'bass', 'vocals', 'other'
        """
        # Lazy-load Demucs
        if self._separator is None:
            logger.info(f"Loading Demucs model: {self.demucs_model_name}...")
            try:
                from demucs.pretrained import get_model
                from demucs.apply import apply_model

                self._separator_model = get_model(self.demucs_model_name)
                self._separator_model.to(self.device)
                self._apply_model = apply_model
                self._separator = True
                logger.info("Demucs loaded successfully")
            except ImportError:
                logger.error(
                    "Demucs not installed. Install with: pip install demucs"
                )
                raise
            except Exception as e:
                logger.error(f"Error loading Demucs: {e}")
                raise

        # Convert to tensor (channels, samples)
        audio_tensor = torch.from_numpy(audio.T).float().unsqueeze(0)
        audio_tensor = audio_tensor.to(self.device)

        # Apply separation
        logger.info("Running Demucs inference...")
        with torch.no_grad():
            sources = self._apply_model(
                self._separator_model,
                audio_tensor,
                device=self.device,
                shifts=1,
                split=True,
                overlap=0.25,
            )

        # Convert back to numpy (stems are ordered: drums, bass, other, vocals)
        stems_tensor = sources[0].cpu().numpy()  # (4, channels, samples)

        stems = {
            "drums": stems_tensor[0].T,   # (samples, channels)
            "bass": stems_tensor[1].T,
            "other": stems_tensor[2].T,
            "vocals": stems_tensor[3].T,
        }

        return stems

    def _remix_stems(self, stems: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Mix processed stems back into stereo

        Args:
            stems: Dictionary of processed stems

        Returns:
            Stereo mix (samples, 2)
        """
        # Sum all stems
        mixed = sum(stems.values())

        # Ensure stereo
        if mixed.ndim == 1:
            mixed = np.stack([mixed, mixed], axis=1)

        return mixed

    def _resample(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """Resample audio using librosa"""
        import librosa

        # Resample each channel
        if audio.ndim == 2:
            resampled = np.zeros(
                (int(len(audio) * target_sr / orig_sr), audio.shape[1])
            )
            for ch in range(audio.shape[1]):
                resampled[:, ch] = librosa.resample(
                    audio[:, ch],
                    orig_sr=orig_sr,
                    target_sr=target_sr,
                )
        else:
            resampled = librosa.resample(
                audio, orig_sr=orig_sr, target_sr=target_sr
            )

        return resampled

    def export_stems(
        self,
        stems: Dict[str, np.ndarray],
        output_dir: str,
        prefix: str = "stem",
    ) -> None:
        """
        Export separated stems as individual WAV files

        Args:
            stems: Dictionary of stems
            output_dir: Output directory
            prefix: Filename prefix
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for name, audio in stems.items():
            filepath = output_path / f"{prefix}_{name}.wav"
            sf.write(filepath, audio, self.sample_rate, subtype='PCM_24')
            logger.info(f"Exported: {filepath}")
