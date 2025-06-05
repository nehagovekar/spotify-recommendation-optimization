"""
Setup configuration for Spotify Hit Predictor & A/B Testing Platform
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="spotify-hit-predictor",
    version="1.0.0",
    author="Neha Govekar",
    author_email="nehagovekar2198@gmail.com",
    description="ML-powered song success prediction and A/B testing platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nehagovekar/spotify-recommendation-optimization",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Data Scientists",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "pylint>=2.14.0",
            "pre-commit>=2.20.0",
        ],
        "api": [
            "fastapi>=0.78.0",
            "uvicorn>=0.18.0",
        ],
        "dashboard": [
            "streamlit>=1.11.0",
            "dash>=2.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "spotify-train=scripts.train_models:main",
            "spotify-test=scripts.run_ab_test:main",
            "spotify-pipeline=scripts.run_full_pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "config/*.json"],
    },
)