from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="spectral-luthier",
    version="0.1.0",
    author="Scott Mills",
    author_email="scott@sgmstudios.ca",
    description="Autonomous Stem-Aware Spectral Mastering with Dynamic DSP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/scottmills306/spectral-luthier",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.10.0",
        "soundfile>=0.12.1",
        "librosa>=0.10.0",
        "pedalboard>=0.9.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "loguru>=0.7.0",
        # Optional for stem separation: "demucs>=4.0.1", "torch>=2.2.0", "torchaudio>=2.2.0"
    ],
    entry_points={
        "console_scripts": [
            "spectral-luthier=spectral_luthier.cli:app",
        ],
    },
)
