<!-- logo in the center -->
<div align="center">
<img src="https://raw.githubusercontent.com/silvermete0r/oikan/main/docs/media/oikan_logo.png" alt="OIKAN Logo" width="200"/>

<h1>OIKAN: Optimized Interpretable Kolmogorov-Arnold Networks</h1>
</div>

## Overview
OIKAN (Optimized Interpretable Kolmogorov-Arnold Networks) is a neuro-symbolic ML framework that combines modern neural networks with classical Kolmogorov-Arnold representation theory. It provides interpretable machine learning solutions through automatic extraction of symbolic mathematical formulas from trained models.

[![PyPI version](https://badge.fury.io/py/oikan.svg)](https://badge.fury.io/py/oikan)
[![PyPI Downloads per month](https://img.shields.io/pypi/dm/oikan.svg)](https://pypistats.org/packages/oikan)
[![PyPI Total Downloads](https://static.pepy.tech/badge/oikan)](https://pepy.tech/projects/oikan)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub issues](https://img.shields.io/github/issues/silvermete0r/OIKAN.svg)](https://github.com/silvermete0r/oikan/issues)
[![Docs](https://img.shields.io/badge/docs-passing-brightgreen)](https://silvermete0r.github.io/oikan/)

## Key Features
- 🧠 **Neuro-Symbolic ML**: Combines neural network learning with symbolic mathematics
- 📊 **Automatic Formula Extraction**: Generates human-readable mathematical expressions
- 🎯 **Scikit-learn Compatible**: Familiar `.fit()` and `.predict()` interface
- 🚀 **Production-Ready**: Export symbolic formulas for lightweight deployment
- 📈 **Multi-Task**: Supports both regression and classification problems

## Scientific Foundation

OIKAN is based on Kolmogorov's superposition theorem, which states that any multivariate continuous function can be represented as a composition of single-variable functions. We leverage this theory by:

1. Using neural networks to learn optimal basis functions
2. Employing SVD projection for dimensionality reduction
3. Applying symbolic regression to extract interpretable formulas

## Quick Start

### Installation

#### Method 1: Via PyPI (Recommended)
```bash
pip install -qU oikan
```

#### Method 2: Local Development
```bash
git clone https://github.com/silvermete0r/OIKAN.git
cd OIKAN
pip install -e .  # Install in development mode
```

### Regression Example
```python
from oikan.model import OIKANRegressor
from sklearn.model_selection import train_test_split

# Initialize model with optimal architecture
model = OIKANRegressor(
    hidden_dims=[16, 8],  # Network architecture
    num_basis=10,         # Number of basis functions
    degree=3,             # Polynomial degree
    dropout=0.1           # Regularization
)

# Fit model (sklearn-style)
model.fit(X_train, y_train, epochs=200, lr=0.01)

# Get predictions
y_pred = model.predict(X_test)

# Save interpretable formula to file with auto-generated guidelines
# The output file will contain:
# - Detailed symbolic formulas for each feature
# - Instructions for practical implementation
# - Recommendations for production deployment
model.save_symbolic_formula("regression_formula.txt")
```

*Example of the saved symbolic formula instructions: [outputs/regression_symbolic_formula.txt](outputs/regression_symbolic_formula.txt)*


### Classification Example
```python
from oikan.model import OIKANClassifier

# Similar sklearn-style interface for classification
model = OIKANClassifier(hidden_dims=[16, 8])
model.fit(X_train, y_train)
probas = model.predict_proba(X_test)

# Save classification formulas with implementation guidelines
# The output file will contain:
# - Decision boundary formulas for each class
# - Softmax application instructions
# - Production deployment recommendations
model.save_symbolic_formula("classification_formula.txt")
```

*Example of the saved symbolic formula instructions: [outputs/classification_symbolic_formula.txt](outputs/classification_symbolic_formula.txt)*

## Architecture Details

OIKAN's architecture consists of three main components:

1. **Basis Function Layer**: Learns optimal single-variable transformations
   - B-spline bases for smooth function approximation
   - Trigonometric bases for periodic patterns
   - Polynomial bases for algebraic relationships

2. **Neural Composition Layer**: Combines transformed features
   - SVD projection for dimensionality reduction
   - Dropout for regularization
   - Skip connections for gradient flow

3. **Symbolic Extraction Layer**: Generates interpretable formulas
   - L1 regularization for sparse representations
   - Symbolic regression for formula extraction
   - LaTeX export for documentation

## Contributing

We welcome contributions! Key areas of interest:

- Model architecture improvements
- Novel basis function implementations
- Improved symbolic extraction algorithms
- Real-world case studies and applications
- Performance optimizations

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use OIKAN in your research, please cite:

```bibtex
@software{oikan2025,
  title = {OIKAN: Optimized Interpretable Kolmogorov-Arnold Networks},
  author = {Zhalgasbayev, Arman},
  year = {2025},
  url = {https://github.com/silvermete0r/OIKAN}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.