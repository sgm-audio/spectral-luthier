# Spectral Neural Luthier

Autonomous Stem-Aware Spectral Mastering with Dynamic DSP

## Overview

The Spectral Neural Luthier represents a paradigm shift in audio mastering. Instead of blindly applying EQ and compression to a stereo mixdown, it:

1. **Separates** the stereo file into 4 stems (Drums, Bass, Vocals, Other) using Hybrid Transformer Demucs
2. **Analyzes** each stem's spectral and dynamic characteristics (LUFS, spectral centroid, dynamic range)
3. **Compares** against genre-specific target profiles
4. **Rebalances** each stem using corrective EQ, compression, and gain to match targets
5. **Remixes** the stems back to stereo with proper balance
6. **Limits** transparently to achieve target loudness without artifacts

This approach fixes mix issues (e.g., "snare too loud", "bass muddy") **without access to the original multitrack**.

## Key Innovations (2025/2026)

- **Source Separation** - Uses latest Demucs Hybrid Transformer (4.0+) for clean stem extraction
- **Spectral Repair** - Goes beyond simple EQ matching to fix fundamental spectral balance issues
- **Stem-Level Processing** - Applies DSP where it's needed, not across the entire mix
- **Transparent Limiting** - True-peak limiting with lookahead and soft-knee for maximum loudness without clipping
- **Profile-Driven** - Genre-specific targets (Modern Metal, Cinematic Pop, etc.) guide the mastering decisions

## Installation

```bash
cd spectral-luthier
pip install -e .
```

### Dependencies

- **PyTorch** - Deep learning inference (CUDA recommended for speed)
- **Demucs** - Hybrid Transformer source separation
- **Pedalboard** (Spotify) - Professional audio DSP
- **librosa** - Spectral analysis
- **pyloudnorm** - LUFS measurement

All dependencies install automatically with `pip install -e .`

## Usage

### Command Line

```bash
# Basic usage
spectral-luthier master input.wav -o mastered.wav -p Modern_Metal

# With multiband limiter (more transparent)
spectral-luthier master song.wav -o master.wav -p Cinematic_Pop --multiband

# Verbose output
spectral-luthier master track.wav -o out.wav -p Modern_Metal -v

# List available profiles
spectral-luthier profiles

# Show system info
spectral-luthier info
```

### Python API

```python
from spectral_luthier import StemMaster

# Initialize
master = StemMaster(device="cuda")  # or "cpu"

# Process
result = master.process(
    input_path="my_song.wav",
    profile="Modern_Metal",
    output_path="mastered.wav",
    multiband_limiting=True,
)

print(f"Final LUFS: {result['final_lufs']}")
```

### Advanced Usage

```python
from spectral_luthier import StemMaster, TargetProfile, StemTarget

# Create custom profile
custom_profile = TargetProfile(
    name="My_Custom_Master",
    description="Custom mastering profile",
    final_lufs_integrated=-10.0,
    final_true_peak=-0.5,
    drums=StemTarget(
        lufs_integrated=-12.0,
        dynamic_range=8.0,
        spectral_centroid=3000.0,
        eq_curve={
            "60": 2.0,    # Kick boost
            "3000": 3.0,  # Snare crack
        },
        compression_ratio=4.0,
        compression_threshold=-18.0,
    ),
    # ... define bass, vocals, other ...
    stem_balance={
        "drums": 0.0,
        "bass": -2.0,
        "vocals": 2.0,
        "other": -1.0,
    }
)

# Use custom profile
master = StemMaster()
result = master.process(
    "input.wav",
    profile=custom_profile,  # Pass object instead of string
    output_path="output.wav",
)
```

## Available Profiles

### Modern_Metal
- **Target LUFS:** -8.0 (very loud)
- **Characteristics:** Aggressive, punchy, tight low end
- **Best for:** Metal, metalcore, djent, hard rock
- **Stems:** Heavily compressed drums, tight bass, upfront vocals

### Cinematic_Pop
- **Target LUFS:** -11.0 (moderate)
- **Characteristics:** Wide, dynamic, lush vocals
- **Best for:** Pop, cinematic, ballads, indie
- **Stems:** More dynamic drums, deep bass, natural vocal dynamics

## Technical Details

### Pipeline Architecture

```
Input WAV (Stereo)
    ↓
[1] Source Separation (Demucs HT)
    ↓
4 Stems: Drums, Bass, Vocals, Other
    ↓
[2] Analysis (per stem)
    - LUFS-I (Integrated)
    - LUFS-S (Short-term)
    - Dynamic Range (PLR)
    - Spectral Centroid
    - Spectral Rolloff
    - Crest Factor
    ↓
[3] Compare to Target Profile
    ↓
[4] Rebalancing (per stem)
    - Gain adjustment (LUFS matching)
    - Compression (dynamic control)
    - EQ (spectral shaping)
    ↓
[5] Remix (sum stems with balance)
    ↓
[6] Final Limiting
    - True-peak limiter
    - Target LUFS achievement
    - Optional multiband mode
    ↓
Output Master WAV (Stereo, 24-bit)
```

### Performance

- **GPU (CUDA):** ~30-60 seconds for 4-minute song
- **CPU:** ~2-5 minutes for 4-minute song
- **RAM:** ~4-8 GB (Demucs model + audio buffers)
- **VRAM:** ~2-4 GB (GPU mode)

### Demucs Models

- **htdemucs** (default) - Hybrid Transformer, best quality/speed tradeoff
- **htdemucs_ft** - Fine-tuned on more data, slightly better quality
- **mdx_extra** - Experimental, highest quality but slower

Change model:
```python
master = StemMaster(demucs_model="htdemucs_ft")
```

## Integration with MAAPS

The Spectral Luthier can be integrated into the MAAPS ecosystem as a new agent:

```python
# See: maaps-integration/agents/spectral_mastering_agent.py

class SpectralMasteringAgent(BaseAgent):
    """MAAPS agent wrapper for Spectral Luthier"""

    def setup(self):
        self.subscribe("audio.mastering.request")
        self.stem_master = StemMaster()

    def handle_message(self, message):
        if message.topic == "audio.mastering.request":
            result = self.stem_master.process(
                input_path=message.data["input_file"],
                profile=message.data.get("profile", "Modern_Metal"),
                output_path=message.data["output_file"],
            )
            self.publish("audio.mastering.complete", result)
```

## Examples

See `examples/` directory:
- `basic_usage.py` - Simple mastering example
- `custom_profile.py` - Create and use custom profiles
- `stem_export.py` - Export separated stems
- `batch_processing.py` - Process multiple files

## Troubleshooting

### CUDA Out of Memory
- Use smaller Demucs model: `StemMaster(demucs_model="htdemucs")`
- Process on CPU: `StemMaster(device="cpu")`
- Process shorter segments

### Poor Separation Quality
- Use `htdemucs_ft` for better quality
- Ensure input is high-quality (24-bit recommended)
- Check for phase issues in source material

### Limiting Artifacts
- Enable multiband limiter: `multiband_limiting=True`
- Reduce target LUFS: e.g., -14.0 instead of -8.0
- Adjust true peak ceiling: e.g., -1.5 dBTP instead of -0.5

## Citation

If you use this in research or production:

```bibtex
@software{spectral_luthier_2026,
  title={Spectral Neural Luthier: Autonomous Stem-Aware Spectral Mastering},
  author={Agentic Systems 2026},
  year={2026},
  url={https://github.com/...}
}
```

## License

MIT License

## Acknowledgments

- **Demucs** (Meta AI) - Source separation
- **Pedalboard** (Spotify) - Audio DSP
- **librosa** - Audio analysis
- **pyloudnorm** - LUFS metering
