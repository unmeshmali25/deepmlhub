from setuptools import find_packages, setup

setup(
    name="llm-retard-lab",
    version="0.1.0",
    description="End-to-end LLM training pipeline following Karpathy's nanoGPT philosophy",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.24.0",
        "transformers>=4.40.0",
        "tiktoken>=0.7.0",
        "trl>=0.8.0",
        "mlflow>=2.13.0",
        "pyyaml>=6.0",
        "google-cloud-storage>=2.10.0",
        "tensorboard>=2.15.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "ruff>=0.1.0",
            "mypy>=1.7.0",
        ],
    },
)
