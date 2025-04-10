from .exceptions import *
import torch
import torch.nn as nn
import numpy as np

# Core basis functions with explicit variable notation
ADVANCED_LIB = {
    'x':    ('x', lambda x: x),
    'x^2':  ('x^2', lambda x: np.clip(x**2, -100, 100)),
    'x^3':  ('x^3', lambda x: np.clip(x**3, -100, 100)),
    'exp':  ('exp(x)', lambda x: np.exp(np.clip(x, -10, 10))),
    'log':  ('log(x)', lambda x: np.log(np.abs(x) + 1)),
    'sqrt': ('sqrt(x)', lambda x: np.sqrt(np.abs(x))),
    'tanh': ('tanh(x)', lambda x: np.tanh(x)),
    'sin':  ('sin(x)', lambda x: np.sin(np.clip(x, -10*np.pi, 10*np.pi))),
    'abs':  ('abs(x)', lambda x: np.abs(x))
}

class EdgeActivation(nn.Module):
    """Learnable edge-based activation function."""
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(len(ADVANCED_LIB)))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        features = []
        for _, func in ADVANCED_LIB.values():
            feat = torch.tensor(func(x.detach().cpu().numpy()), 
                              dtype=torch.float32).to(x.device)
            features.append(feat)
        features = torch.stack(features, dim=-1)
        return torch.matmul(features, self.weights.unsqueeze(0).T) + self.bias
    
    def get_symbolic_repr(self, threshold=1e-4):
        """Get symbolic representation of the activation function."""
        significant_terms = []
        
        for (notation, _), weight in zip(ADVANCED_LIB.values(), 
                                       self.weights.detach().cpu().numpy()):
            if abs(weight) > threshold:
                significant_terms.append(f"{weight:.4f}*{notation}")
                
        if abs(self.bias.item()) > threshold:
            significant_terms.append(f"{self.bias.item():.4f}")
                
        return " + ".join(significant_terms) if significant_terms else "0"