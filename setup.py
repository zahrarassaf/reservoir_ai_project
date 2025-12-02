"""
Setup script for Reservoir Simulation Framework
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="reservoir-simulator",
    version="2.0.0",
    author="Reservoir Engineering Team",
    author_email="contact@example.com",
    description="Professional Reservoir Simulation Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/reservoir-simulator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
        "plot": [
            "plotly>=5.10.0",
            "seaborn>=0.11.0",
        ],
        "web": [
            "streamlit>=1.12.0",
            "flask>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "reservoir-sim=run_simulation:main",
        ],
    },
)
