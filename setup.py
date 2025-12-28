"""Setup script for VitalDB AKI prediction package."""

from pathlib import Path

from setuptools import find_packages, setup

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="vitaldb-aki",
    version="0.1.0",
    description="Acute Kidney Injury prediction from VitalDB intraoperative vital signs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="VitalDB AKI Team",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "torch>=1.10.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
        "pyyaml>=5.4.0",
        "requests>=2.26.0",
        "urllib3>=1.26.0",
    ],
    entry_points={
        "console_scripts": [
            "vitaldb-preprocess=vitaldb_aki.scripts.preprocess:main",
            "vitaldb-train=vitaldb_aki.scripts.train:main",
            "vitaldb-evaluate=vitaldb_aki.scripts.evaluate:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

