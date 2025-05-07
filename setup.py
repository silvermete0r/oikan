from setuptools import setup, find_packages

setup(
    name="oikan",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "scikit-learn",
        "tqdm"  # Add tqdm for progress bars
    ]
)
