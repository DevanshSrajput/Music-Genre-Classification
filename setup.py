"""Setup script for the Music Genre Classification package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="music-genre-classification",
    version="1.0.0",
    author="DevanshSrajput",
    author_email="devansh@example.com",
    description="A deep learning application for automatic music genre classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DevanshSrajput/music-genre-classification",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "librosa>=0.10.1",
        "tensorflow>=2.13.0",
        "streamlit>=1.25.0",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.2",
        "plotly>=5.15.0",
        "soundfile>=0.12.1",
        "pydub>=0.25.1",
    ],
    entry_points={
        "console_scripts": [
            "genre-predict=predict:main",
            "genre-train=train_model:main",
        ],
    },
)