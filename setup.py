from pathlib import Path

from setuptools import setup, find_packages

_here = Path(__file__).parent

with (_here / "requirements.txt").open() as _f:
    _install_requires = [
        line.strip()
        for line in _f
        if line.strip() and not line.strip().startswith("#")
    ]

setup(
    name="clicklens",
    version="0.1.0",
    description="YouTube Thumbnail Click-Through Predictor",
    author="Lindsay Gross, Arnav Mahale, Sharmil Nanjappa",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=_install_requires,
)
