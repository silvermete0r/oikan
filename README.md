# OIKAN

Optimized Interpretable Kolmogorov-Arnold Networks (OIKAN)  
A deep learning framework for interpretable neural networks using advanced basis functions.

[![PyPI version](https://badge.fury.io/py/oikan.svg)](https://badge.fury.io/py/oikan)
[![PyPI downloads](https://img.shields.io/pypi/dm/oikan.svg)](https://pypistats.org/packages/oikan)

## Key Features
- EfficientKAN layer implementation
- Built-in visualization tools
- Support for both regression and classification tasks
- Symbolic formula extraction
- Easy-to-use training interface
- LaTeX-formatted formula extraction

## Installation

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
from oikan.symbolic import extract_symbolic_formula

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