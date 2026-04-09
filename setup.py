from setuptools import setup, find_packages

setup(
    name="clicklens",
    version="0.1.0",
    description="YouTube Thumbnail Click-Through Predictor",
    author="Lindsay Gross, Arnav Mahale, Sharmil Nanjappa",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
)
