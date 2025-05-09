<!-- logo in the center -->
<div align="center">
<img src="https://raw.githubusercontent.com/silvermete0r/oikan/main/docs/media/oikan_logo.png" alt="OIKAN Logo" width="200"/>

<h1>OIKAN: Neuro-Symbolic ML for Scientific Discovery</h1>
</div>

## Overview

OIKAN is a neuro-symbolic machine learning framework inspired by Kolmogorov-Arnold representation theorem. It combines the power of modern neural networks with techniques for extracting clear, interpretable symbolic formulas from data. OIKAN is designed to make machine learning models both accurate and Interpretable.

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

OIKAN implements a modern interpretation of the Kolmogorov-Arnold Representation Theorem through a hybrid neural architecture:

1. **Theoretical Foundation**: The Kolmogorov-Arnold theorem states that any continuous n-dimensional function can be decomposed into a combination of single-variable functions:

   ```
   f(x₁,...,xₙ) = ∑(j=0 to 2n){ φⱼ( ∑(i=1 to n) ψᵢⱼ(xᵢ) ) }
   ```

   where φⱼ and ψᵢⱼ are continuous univariate functions.

2. **Neural Implementation**: OIKAN uses a specialized architecture combining:
   - Feature transformation layers with interpretable basis functions
   - Symbolic regression for formula extraction
   - Automatic pruning of insignificant terms
   
   ```python
   class OIKANRegressor:
       def __init__(self, hidden_sizes=[64, 64], activation='relu',
                    polynomial_degree=2, alpha=0.1):
           # Neural network for learning complex patterns
           self.neural_net = TabularNet(input_size, hidden_sizes, activation)
           # Symbolic regression for interpretable formulas
           self.symbolic_model = None

   ```

3. **Basis Functions**: Core set of interpretable transformations:
   ```python
   SYMBOLIC_FUNCTIONS = {
       'linear': 'x',           # Direct relationships
       'quadratic': 'x^2',      # Non-linear patterns
       'interaction': 'x_i x_j', # Feature interactions
       'higher_order': 'x^n'    # Polynomial terms
   }
   ```

4. **Formula Extraction Process**:
   - Train neural network on raw data
   - Generate augmented samples for better coverage
   - Perform L1-regularized symbolic regression
   - Prune terms with coefficients below threshold
   - Export human-readable mathematical expressions

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
from sklearn.metrics import mean_squared_error

# Initialize model
model = OIKANRegressor(
    hidden_sizes=[32, 32], # Hidden layer sizes
    activation='relu', # Activation function (other options: 'tanh', 'leaky_relu', 'elu', 'swish', 'gelu')
    augmentation_factor=5, # Augmentation factor for data generation
    polynomial_degree=2, # Degree of polynomial basis functions
    alpha=0.1, # L1 regularization strength
    sigma=0.1, # Standard deviation of Gaussian noise for data augmentation
    epochs=100, # Number of training epochs
    lr=0.001, # Learning rate
    batch_size=32, # Batch size for training
    verbose=True # Verbose output during training
)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Get symbolic formula
formula = model.get_formula()
print("Symbolic Formula:", formula)

# Get feature importances
importances = model.feature_importances()
print("Feature Importances:", importances)

# Save the model (optional)
model.save("outputs/model.json")

# Load the model (optional)
loaded_model = OIKANRegressor()
loaded_model.load("outputs/model.json")
```

*Example of the saved symbolic formula (regression model): [outputs/california_housing_model.json](outputs/california_housing_model.json)*


### Classification Example
```python
from oikan.model import OIKANClassifier
from sklearn.metrics import accuracy_score

# Initialize model
model = OIKANClassifier(
    hidden_sizes=[32, 32], # Hidden layer sizes
    activation='relu', # Activation function (other options: 'tanh', 'leaky_relu', 'elu', 'swish', 'gelu')
    augmentation_factor=10, # Augmentation factor for data generation
    polynomial_degree=2, # Degree of polynomial basis functions
    alpha=0.1, # L1 regularization strength
    sigma=0.1, # Standard deviation of Gaussian noise for data augmentation
    epochs=100, # # Number of training epochs
    lr=0.001, # Learning rate
    batch_size=32, # Batch size for training
    verbose=True # Verbose output during training
)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Get symbolic formulas for each class
formulas = model.get_formula()
for i, formula in enumerate(formulas):
    print(f"Class {i} Formula:", formula)
   
# Get feature importances
importances = model.feature_importances()
print("Feature Importances:", importances)

# Save the model (optional)
model.save("outputs/model.json")

# Load the model (optional)
loaded_model = OIKANClassifier()
loaded_model.load("outputs/model.json")
```

*Example of the saved symbolic formula (classification model): [outputs/iris_model.json](outputs/iris_model.json)*

### Architecture Diagram

![OIKAN v0.0.3(1) Architecture](https://raw.githubusercontent.com/silvermete0r/oikan/main/docs/media/oikan-v0.0.3(1)-architecture-oop.png)

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
  title = {OIKAN: Neuro-Symbolic ML for Scientific Discovery},
  author = {Zhalgasbayev, Arman},
  year = {2025},
  url = {https://github.com/silvermete0r/OIKAN}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.