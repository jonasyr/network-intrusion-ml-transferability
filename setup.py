from setuptools import setup, find_packages

setup(
    name="ml-network-anomaly-detection",
    version="1.0.0",
    author="Your Name",
    description="Cross-dataset evaluation of ML models for network intrusion detection",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.2.0",
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.11.0",
    ],
    python_requires=">=3.8",
)
