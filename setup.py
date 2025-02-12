from setuptools import setup, find_packages

setup(
    name="oikan",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "sympy",
        "scipy",
        "matplotlib"
    ]
)
