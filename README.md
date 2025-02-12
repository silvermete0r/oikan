# OIKAN Library

[![PyPI version](https://badge.fury.io/py/oikan.svg)](https://badge.fury.io/py/oikan)
[![PyPI downloads](https://img.shields.io/pypi/dm/oikan.svg)](https://pypistats.org/packages/oikan)

OIKAN (Optimized Implementation of Kolmogorov-Arnold Networks) is a PyTorch-based library for creating interpretable neural networks. It implements the KAN architecture to provide both accurate predictions and interpretable results.

## Key Features

- EfficientKAN layer implementation
- Built-in visualization tools
- Support for both regression and classification tasks
- Symbolic formula extraction
- Easy-to-use training interface

## Installation

```bash
git clone https://github.com/yourusername/OIKAN.git
cd OIKAN
pip install -r requirements.txt
```

## Quick Start

### Regression Example
```python
from oikan.model import OIKAN
from oikan.trainer import train

# Create and train model
model = OIKAN(input_dim=2, output_dim=1)
train(model, train_loader)

# Extract interpretable formula
formula = extract_symbolic_formula_regression(model, X)
```

### Classification Example
```python
model = OIKAN(input_dim=2, output_dim=2)
train_classification(model, train_loader)
visualize_classification(model, X, y)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.