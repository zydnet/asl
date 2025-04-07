#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Setup script for SignSync package
"""

import os
from setuptools import setup, find_packages


def read_requirements():
    """Read requirements from requirements.txt"""
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def read_long_description():
    """Read long description from README.md"""
    with open("README.md", "r") as f:
        return f.read()


setup(
    name="signsync",
    version="0.1.0",
    description="Real-Time Sign Language Translation Suite",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="SignSync Team",
    author_email="info@signsync.example.com",
    url="https://github.com/signsync/signsync",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "signsync=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
    ],
)
