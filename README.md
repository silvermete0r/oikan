# OIKAN

## OIKAN v0.0.2 is under development and not yet released!!! 

## Coming Soon!!!

Optimized Interpretable Kolmogorov-Arnold Networks (OIKAN)  
A deep learning framework for interpretable neural networks using advanced basis functions.

[![PyPI version](https://badge.fury.io/py/oikan.svg)](https://badge.fury.io/py/oikan)
[![PyPI Downloads per month](https://img.shields.io/pypi/dm/oikan.svg)](https://pypistats.org/packages/oikan)
[![PyPI Total Downloads](https://static.pepy.tech/badge/oikan)](https://pepy.tech/projects/oikan)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub issues](https://img.shields.io/github/issues/silvermete0r/OIKAN.svg)](https://github.com/silvermete0r/oikan/issues)
[![Docs](https://img.shields.io/badge/docs-passing-brightgreen)](https://silvermete0r.github.io/oikan/)

## Key Features
- 🚀 Efficient Implementation ~ Optimized KAN architecture with SVD projection
- 📊 Advanced Basis Functions ~ B-spline and Fourier basis transformations
- 🎯 Multi-Task Support ~ Both regression and classification capabilities
- 🔍 Interpretability Tools ~ Extract and visualize symbolic formulas
- 📈 Interactive Visualizations ~ Built-in plotting and analysis tools
- 🧮 Symbolic Mathematics ~ LaTeX formula extraction and symbolic approximations

## Installation

### Method 1: Via PyPI (Recommended)
```bash
pip install oikan
```

### Method 2: Local Development
```bash
git clone https://github.com/silvermete0r/OIKAN.git
cd OIKAN
pip install -e .  # Install in development mode
```

## Quick Start

### Regression Example
```python
from oikan.model import OIKAN
from oikan.trainer import train
from oikan.visualize import visualize_regression
from oikan.symbolic import extract_symbolic_formula, plot_symbolic_formula, extract_latex_formula

model = OIKAN(input_dim=2, output_dim=1)
train(model, (X_train, y_train))

visualize_regression(model, X, y)

formula = extract_symbolic_formula(model, X_test, mode='regression')
print("Extracted formula:", formula)

plot_symbolic_formula(model, X_test, mode='regression')

latex_formula = extract_latex_formula(model, X_test, mode='regression')
print("LaTeX:", latex_formula)
```

### Classification Example
```python
from oikan.model import OIKAN
from oikan.trainer import train_classification
from oikan.visualize import visualize_classification
from oikan.symbolic import extract_symbolic_formula, plot_symbolic_formula, extract_latex_formula

model = OIKAN(input_dim=2, output_dim=2)
train_classification(model, (X_train, y_train))

visualize_classification(model, X_test, y_test)

formula = extract_symbolic_formula(model, X_test, mode='classification')
print("Extracted formula:", formula)

plot_symbolic_formula(model, X_test, mode='classification')

latex_formula = extract_latex_formula(model, X_test, mode='classification')
print("LaTeX:", latex_formula)
```

## Usage
- Explore the `oikan/` folder for model architectures, training routines, and symbolic extraction.
- Check the `examples/` directory for complete usage examples for both regression and classification.

## Contributing
Contributions are welcome! Submit a Pull Request with your improvements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.