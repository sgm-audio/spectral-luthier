# spectral-luthier

An experiment in mastering from a stereo mix by working on stems. It splits the mix into drums, bass, vocals and other with Demucs, measures each stem, nudges each one toward a genre target profile with gain, EQ and compression (Spotify's pedalboard), sums them back, and runs a limiter to hit a target loudness.

The idea is to fix things like "snare too loud" or "bass muddy" when you only have the bounce, not the multitrack. It's early research code. I haven't run a listening test or a before/after measurement on real material yet, so there are no quality claims here.

## Pipeline

1. Separate with Demucs (`htdemucs` by default; `htdemucs_ft` and `mdx_extra` can be passed in). Runs on CUDA if available, otherwise CPU.
2. Per stem, measure integrated, short-term and momentary loudness (pyloudnorm), dynamic range, spectral centroid, rolloff and bandwidth (`analyzer.py`).
3. Compare against the profile's per-stem targets.
4. Rebalance each stem: gain to the loudness target, a pedalboard compressor and an EQ chain from the profile (`rebalancer.py`).
5. Sum the stems with the profile's balance offsets.
6. Limit (`limiter.py`): pedalboard's `Limiter` plus gain toward the target LUFS. True peak is measured with a 4x resample, but the limiter itself has no lookahead or soft knee. `--multiband` splits into 3 bands and limits each one separately. I haven't checked how close that lands to the loudness target.

Two profiles ship in `profiles.py`: `Modern_Metal` (-8 LUFS integrated) and `Cinematic_Pop` (-11 LUFS). You can build your own `TargetProfile` in Python.

## Install

```bash
git clone https://github.com/sgm-audio/spectral-luthier.git
cd spectral-luthier
pip install -e .
```

This pulls in torch, torchaudio and demucs. `stem_master.py` imports torch when the package loads, so they aren't optional. On a machine without a GPU you can install the CPU wheels first to save the download:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

Demucs downloads its model weights the first time it runs.

## Usage

```bash
spectral-luthier master input.wav -o mastered.wav -p Modern_Metal
spectral-luthier master song.wav -o master.wav -p Cinematic_Pop --multiband
spectral-luthier profiles
spectral-luthier info
```

```python
from spectral_luthier import StemMaster

result = StemMaster(device="cpu").process(
    input_path="my_song.wav",
    profile="Modern_Metal",
    output_path="mastered.wav",
)
print(result["final_lufs"])
```

`examples/basic_usage.py` is the only example so far.

## Status

- Until this branch, `pip install -e .` didn't install torch, demucs, pyloudnorm or pydantic, so the package failed to import. They're now listed in `setup.py` and `requirements.txt`.
- No tests yet.
- No speed or memory numbers yet. I haven't timed it on a known machine.

## License

MIT. See [LICENSE](LICENSE).

Uses Demucs (Meta), pedalboard (Spotify), librosa and pyloudnorm, each under its own licence.
