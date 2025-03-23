"""
Setup script for wake word detection engine
"""
from setuptools import setup, find_packages

setup(
    name="wakeword",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "PyAudio>=0.2.11",
        "torch>=1.9.0",
        "librosa>=0.8.0",
        "PySimpleGUI>=4.45.0",
        "pystray>=0.19.0",
        "Pillow>=8.0.0",
        "scikit-learn>=0.24.0",
    ],
    entry_points={
        "console_scripts": [
            "wakeword=wakeword.main:main",
        ],
    },
    author="Wake Word Engine Developer",
    author_email="developer@example.com",
    description="Offline Wake Word Detection Engine",
    keywords="wake word, speech recognition, voice assistant",
    python_requires=">=3.9",
)
