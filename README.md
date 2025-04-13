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

> **Important Disclaimer**: OIKAN is an experimental research project. It is not intended for production use or real-world applications. This framework is designed for research purposes, experimentation, and academic exploration of neuro-symbolic machine learning concepts.

## Key Features
- 🧠 **Neuro-Symbolic ML**: Combines neural network learning with symbolic mathematics
- 📊 **Automatic Formula Extraction**: Generates human-readable mathematical expressions
- 🎯 **Scikit-learn Compatible**: Familiar `.fit()` and `.predict()` interface
- 🔬 **Research-Focused**: Designed for academic exploration and experimentation
- 📈 **Multi-Task**: Supports both regression and classification problems

## Scientific Foundation

OIKAN implements the Kolmogorov-Arnold Representation Theorem through a novel neural architecture:

1. **Theorem Background**: Any continuous multivariate function f(x1,...,xn) can be represented as:
   ```
   f(x1,...,xn) = ∑(j=0 to 2n){ φj( ∑(i=1 to n) ψij(xi) ) }
   ```
   where φj and ψij are continuous single-variable functions.

2. **Neural Implementation**:
   ```python
   # Pseudo-implementation of KAN architecture
   class KANLayer:
       def __init__(self, input_dim, output_dim):
           self.edges = [SymbolicEdge() for _ in range(input_dim * output_dim)]
           self.weights = initialize_weights(input_dim, output_dim)
   
       def forward(self, x):
           # Transform each input through basis functions
           edge_outputs = [edge(x_i) for x_i, edge in zip(x, self.edges)]
           # Combine using learned weights
           return combine_weighted_outputs(edge_outputs, self.weights)
   ```

3. **Basis functions**
```python
# Edge activation contains interpretable basis functions
ADVANCED_LIB = {
      'x': (lambda x: x),          # Linear
      'x^2': (lambda x: x**2),     # Quadratic
      'sin(x)': np.sin,            # Periodic
      'tanh(x)': np.tanh          # Bounded
}
```

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

# Initialize model 
model = OIKANRegressor()

# Fit model (sklearn-style)
model.fit(X_train, y_train, epochs=100, lr=0.01)

# Get predictions
y_pred = model.predict(X_test)

# Save interpretable formula to file with auto-generated guidelines
# The output file will contain:
# - Detailed symbolic formulas for each feature
# - Instructions for practical implementation
# - Recommendations for testing and validation
model.save_symbolic_formula("regression_formula.txt")
```

*Example of the saved symbolic formula instructions: [outputs/regression_symbolic_formula.txt](outputs/regression_symbolic_formula.txt)*


### Classification Example
```python
from oikan.model import OIKANClassifier

# Similar sklearn-style interface for classification
model = OIKANClassifier()
model.fit(X_train, y_train, epochs=100, lr=0.01)
probas = model.predict_proba(X_test)

# Save classification formulas with implementation guidelines
# The output file will contain:
# - Decision boundary formulas for each class
# - Softmax application instructions
# - Recommendations for testing and validation
model.save_symbolic_formula("classification_formula.txt")
```

*Example of the saved symbolic formula instructions: [outputs/classification_symbolic_formula.txt](outputs/classification_symbolic_formula.txt)*

### Architecture Diagram

![Architecture Diagram](https://raw.githubusercontent.com/silvermete0r/oikan/main/docs/media/oikan_model_architecture_v0.0.2.2.png)

### Key Design Principles

1. **Interpretability First**: All transformations maintain clear mathematical meaning
2. **Scikit-learn Compatibility**: Familiar `.fit()` and `.predict()` interface
3. **Symbolic Formula Exporting**: Export formulas as lightweight mathematical expressions
4. **Automatic Simplification**: Remove insignificant terms (|w| < 1e-4)


### Key Model Components

1. **EdgeActivation Layer**:
   - Implements interpretable basis function transformations
   - Automatically prunes insignificant terms
   - Maintains mathematical transparency

2. **Formula Extraction**:
   - Combines edge transformations with learned weights
   - Applies symbolic simplification
   - Generates human-readable expressions

3. **Training Process**:
   - Gradient-based optimization of edge weights
   - Automatic feature importance detection
   - Complexity control through regularization

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