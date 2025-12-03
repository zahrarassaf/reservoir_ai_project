"""
Documentation generator with API reference.
"""

import subprocess
import sys
from pathlib import Path


def generate_api_docs():
    """Generate API documentation using Sphinx."""
    docs_dir = Path("docs")
    
    # Run sphinx-apidoc
    cmd = [
        "sphinx-apidoc",
        "-f",  # Force overwrite
        "-o", str(docs_dir / "api"),
        "../src",
        "../src/tests/*",  # Exclude tests
        "../src/benchmarks/*",  # Exclude benchmarks
    ]
    
    print("Generating API documentation...")
    subprocess.run(cmd, check=True)
    
    # Build HTML documentation
    build_cmd = [
        "sphinx-build",
        "-b", "html",
        str(docs_dir),
        str(docs_dir / "_build" / "html"),
    ]
    
    print("Building HTML documentation...")
    subprocess.run(build_cmd, check=True)
    
    print("Documentation generated successfully!")


if __name__ == "__main__":
    generate_api_docs()
