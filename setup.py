from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="reservoir-simulation-framework",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Professional Reservoir Simulation Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Zahrarasaf/reservoir_ai_project/tree/main",
    packages=find_packages(),
    package_dir={
        '': '.',
        'data_parser': 'data_parser',
        'analysis': 'analysis',
        'src': 'src',
        'tests': 'tests',
        'utils': 'utils'
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
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
    },
    entry_points={
        "console_scripts": [
            "run-simulation=run_simulation:main",
        ],
    },
)
