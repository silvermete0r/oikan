'''
OIKAN v0.0.3 | 2025

OIKAN is a neuro-symbolic machine learning framework inspired by Kolmogorov-Arnold representation theory. It combines the power of modern neural networks with techniques for extracting clear, interpretable symbolic formulas from data. OIKAN is designed to make machine learning models both accurate and understandable.

GitHub: https://github.com/silvermete0r/oikan
PyPI: https://pypi.org/project/oikan/
Docs: https://silvermete0r.github.io/oikan/
'''

from .model import OIKAN, OIKANClassifier, OIKANRegressor
from .neural import TabularNet
from .elasticnet import ElasticNet

__all__ = ['OIKAN', 'OIKANClassifier', 'OIKANRegressor', 'TabularNet', 'ElasticNet']
__version__ = '0.0.3'