from .exceptions import *
import torch
import torch.nn as nn
import numpy as np

def ensure_tensor(x):
    """Helper function to ensure input is a PyTorch tensor."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    elif isinstance(x, (int, float)):
        return torch.tensor([x], dtype=torch.float32)
    elif isinstance(x, torch.Tensor):
        return x.float()
    else:
        raise ValueError(f"Unsupported input type: {type(x)}")

# Updated to handle numpy arrays and scalars
ADVANCED_LIB = {
    'x': ('x', lambda x: ensure_tensor(x)),
    'x^2': ('x^2', lambda x: torch.pow(ensure_tensor(x), 2)),
    'sin': ('sin(x)', lambda x: torch.sin(ensure_tensor(x))),
    'tanh': ('tanh(x)', lambda x: torch.tanh(ensure_tensor(x)))
}

class EdgeActivation(nn.Module):
    """Learnable edge-based activation function with improved gradient flow."""
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(len(ADVANCED_LIB)))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        x_tensor = ensure_tensor(x)
        features = []
        for _, func in ADVANCED_LIB.values():
            feat = func(x_tensor)
            features.append(feat)
        features = torch.stack(features, dim=-1)
        return torch.matmul(features, self.weights.unsqueeze(0).T) + self.bias
    
    def get_symbolic_repr(self, threshold=1e-4):
        """Get symbolic representation of the activation function."""
        weights_np = self.weights.detach().cpu().numpy()
        significant_terms = []
        
        for (notation, _), weight in zip(ADVANCED_LIB.values(), weights_np):
            if abs(weight) > threshold:
                significant_terms.append(f"{weight:.4f}*{notation}")
                
        if abs(self.bias.item()) > threshold:
            significant_terms.append(f"{self.bias.item():.4f}")
                
        return " + ".join(significant_terms) if significant_terms else "0"