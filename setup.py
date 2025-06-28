from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="human-voice-ai",
    version="0.1.0",
    author="Jaden Fix",
    author_email="jadenfix123@gmail.com",
    description="Real-time emotion-aware voice agent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jadenfix/HumanVoiceAI",
    project_urls={
        "Bug Tracker": "https://github.com/jadenfix/HumanVoiceAI/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.9.0",
        "torchaudio>=0.9.0",
        "librosa>=0.8.1",
        "sounddevice>=0.4.2",
        "streamlit>=1.0.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=0.24.0",
        "shap>=0.40.0",
        "pyyaml>=5.4.1",
        "tqdm>=4.60.0",
    ],
    entry_points={
        "console_scripts": [
            "human-voice-ai=human_voice_ai.__main__:main",
        ],
    },
)
