"""Install FedSER's source and its native Flower application."""

from pathlib import Path
from setuptools import find_packages, setup

setup(
    name="fedser", version="0.2.0",
    description="Privacy-preserving Speech Emotion Recognition through Semi-Supervised Federated Learning",
    long_description=Path("README.md").read_text() if Path("README.md").exists() else "FedSER speech emotion recognition.",
    long_description_content_type="text/markdown", python_requires=">=3.11,<3.14",
    install_requires=["torch==2.8.0", "flwr[simulation]==1.40.0", "numpy>=2,<3",
                      "scipy>=1.14,<2", "soundfile>=0.13,<0.14", "scikit-learn>=1.5,<2"],
    package_dir={"": "src"}, packages=find_packages("src", include=["fedser", "fedser.*"]),
    data_files=[("share/fedser", ["setup.py"])],
    entry_points={"console_scripts": ["fedser-experiment=fedser.experiment:main"]},
)
