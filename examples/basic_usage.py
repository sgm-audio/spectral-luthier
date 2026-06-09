"""
Basic usage example for Spectral Neural Luthier
"""

from spectral_luthier import StemMaster
from pathlib import Path


def main():
    print("Spectral Neural Luthier - Basic Example")
    print("=" * 50)

    # Initialize (auto-detects CUDA if available)
    master = StemMaster(device="auto")

    # Example input file
    input_file = "input_song.wav"

    if not Path(input_file).exists():
        print(f"\nPlease provide an input file: {input_file}")
        print("This should be a stereo WAV file of your mixed (but unmastered) song.")
        return

    # Process with Modern Metal profile
    print("\nProcessing with Modern_Metal profile...")
    result = master.process(
        input_path=input_file,
        profile="Modern_Metal",
        output_path="mastered_metal.wav",
        multiband_limiting=False,  # Set to True for more transparent limiting
    )

    # Display results
    print("\n" + "=" * 50)
    print("RESULTS:")
    print("=" * 50)
    print(f"Output file: {result['output_file']}")
    print(f"Profile used: {result['profile']}")
    print(f"Target LUFS: {result['final_lufs']:.1f}")
    print(f"\nStems analyzed: {', '.join(result['stems_analyzed'])}")

    print("\nPer-stem analysis:")
    for stem_name, analysis in result['analyses'].items():
        print(f"\n{stem_name.upper()}:")
        print(f"  LUFS-I: {analysis['lufs_integrated']:.1f}")
        print(f"  Dynamic Range: {analysis['dynamic_range']:.1f} dB")
        print(f"  Spectral Centroid: {analysis['spectral_centroid']:.0f} Hz")

    print("\n" + "=" * 50)
    print("Mastering complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
