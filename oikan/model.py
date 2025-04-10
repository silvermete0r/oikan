import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from .utils import BSplineBasis, ADVANCED_LIB
from .exceptions import *

class SymbolicEdge(nn.Module):
    """Edge-based activation function learner"""
    def __init__(self, input_dim=1, num_basis=10):
        super().__init__()
        self.input_dim = input_dim
        # One weight per advanced function plus bias
        self.weights = nn.Parameter(torch.randn(len(ADVANCED_LIB)))
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        # Apply each advanced function and combine with weights
        features = []
        for _, func in ADVANCED_LIB.values():
            feat = torch.tensor(func(x.detach().cpu().numpy()), 
                              dtype=torch.float32).to(x.device)
            features.append(feat)
        features = torch.stack(features, dim=-1)
        return torch.matmul(features, self.weights.unsqueeze(0).T) + self.bias
    
    def get_symbolic_repr(self, threshold=1e-4):
        """Get interpretable symbolic representation with explicit variable notation"""
        terms = []
        for w, (notation, _) in zip(self.weights, ADVANCED_LIB.items()):
            if abs(w.item()) > threshold:
                terms.append(f"{w.item():.4f}*{notation[0]}")
        if abs(self.bias.item()) > threshold:
            terms.append(f"{self.bias.item():.4f}")
        return " + ".join(terms) if terms else "0"

class KANLayer(nn.Module):
    """Kolmogorov-Arnold Network layer with interpretable edges"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.edges = nn.ModuleList([
            nn.ModuleList([SymbolicEdge(input_dim=1) for _ in range(output_dim)])
            for _ in range(input_dim)
        ])
        
        self.combination_weights = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1)
    
    def forward(self, x):
        # Vectorized computation using split and stacking instead of nested loops
        x_split = x.split(1, dim=1)  # list of (batch, 1) tensors for each input feature
        edge_outputs = torch.stack([
            torch.stack([edge(x_i).squeeze() for edge in edge_list], dim=1)
            for x_i, edge_list in zip(x_split, self.edges)
        ], dim=1)  # shape: (batch, input_dim, output_dim)
        combined = edge_outputs * self.combination_weights.unsqueeze(0)
        return combined.sum(dim=1)
    
    def get_symbolic_formula(self):
        """Extract interpretable formulas for each output"""
        formulas = []
        for j in range(self.output_dim):
            terms = []
            for i in range(self.input_dim):
                weight = self.combination_weights[i, j].item()
                if abs(weight) > 1e-4:
                    edge_formula = self.edges[i][j].get_symbolic_repr()
                    if edge_formula != "0":
                        terms.append(f"({weight:.4f} * ({edge_formula}))")
            formulas.append(" + ".join(terms) if terms else "0")
        return formulas

class BaseOIKAN(BaseEstimator):
    """Base OIKAN model implementing common functionality"""
    def __init__(self, hidden_dims=[64, 32], num_basis=10, degree=3, dropout=0.1):
        self.hidden_dims = hidden_dims
        self.num_basis = num_basis
        self.degree = degree
        self.dropout = dropout  # Dropout probability for uncertainty quantification
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Auto device chooser
        self.model = None
        self._is_fitted = False
    
    def _build_network(self, input_dim, output_dim):
        layers = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(KANLayer(prev_dim, hidden_dim))
            layers.append(nn.Dropout(self.dropout))  # Apply dropout for uncertainty quantification
            prev_dim = hidden_dim
        layers.append(KANLayer(prev_dim, output_dim))
        return nn.Sequential(*layers).to(self.device)
    
    def _validate_data(self, X, y=None):
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        if y is not None and not isinstance(y, torch.Tensor):
            y = torch.FloatTensor(y)
        return X.to(self.device), (y.to(self.device) if y is not None else None)

    def get_symbolic_formula(self):
        """Generate and cache symbolic formulas for production‐ready inference."""
        if not self._is_fitted:
            raise NotFittedError("Model must be fitted before extracting formulas")
        if hasattr(self, "symbolic_formula"):
            return self.symbolic_formula
        if hasattr(self, 'classes_'):  # Classifier
            n_features = self.model[0].input_dim
            n_classes = len(self.classes_)
            formulas = [[None for _ in range(n_classes)] for _ in range(n_features)]
            first_layer = self.model[0]
            for i in range(n_features):
                for j in range(n_classes):
                    weight = first_layer.combination_weights[i, j].item()
                    if abs(weight) > 1e-4:
                        edge_formula = first_layer.edges[i][j].get_symbolic_repr()
                        terms = []
                        for term in edge_formula.split(" + "):
                            if term and term != "0":
                                if "*" in term:
                                    coef, rest = term.split("*", 1)
                                    coef = float(coef) * weight
                                    terms.append(f"{coef:.4f}*{rest}")
                                else:
                                    terms.append(f"{float(term)*weight:.4f}")
                        formulas[i][j] = " + ".join(terms) if terms else "0"
                    else:
                        formulas[i][j] = "0"
            self.symbolic_formula = formulas
            return formulas
        else:  # Regressor
            formulas = []
            first_layer = self.model[0]
            for i in range(first_layer.input_dim):
                formula = first_layer.edges[i][0].get_symbolic_repr()
                formulas.append(formula)
            self.symbolic_formula = formulas
            return formulas

    def save(self, filename="outputs/symbolic_formula.txt"):
        """Save the cached symbolic formulas to file for efficient production use."""
        sym = self.get_symbolic_formula()
        with open(filename, "w") as f:
            if hasattr(self, 'classes_'):
                for i, feature in enumerate(sym):
                    for j, formula in enumerate(feature):
                        f.write(f"Feature {i} - Class {j}: {formula}\n")
            else:
                for i, formula in enumerate(sym):
                    f.write(f"Feature {i}: {formula}\n")
        print(f"Symbolic formulas saved to {filename}")

    def get_feature_scores(self):
        """Get feature importance scores based on edge weights."""
        if not self._is_fitted:
            raise NotFittedError("Model must be fitted before computing scores")
        
        weights = self.model[0].combination_weights.detach().cpu().numpy()
        return np.mean(np.abs(weights), axis=1)

    def _eval_formula(self, formula, x):
        """Helper to evaluate a symbolic formula for an input vector x using ADVANCED_LIB basis functions."""
        import re
        total = 0
        pattern = re.compile(r"(-?\d+\.\d+)\*?([\w\(\)\^]+)")
        matches = pattern.findall(formula)
        for coef_str, func_name in matches:
            try:
                coef = float(coef_str)
                for key, (notation, func) in ADVANCED_LIB.items():
                    if notation.strip() == func_name.strip():
                        total += coef * func(x)
                        break
            except Exception:
                continue
        return total

    def symbolic_predict(self, X):
        """Predict using only the extracted symbolic formula (regressor)."""
        if not self._is_fitted:
            raise NotFittedError("Model must be fitted before prediction")
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        formulas = self.get_symbolic_formula()  # For regressor: list of formula strings.
        predictions = np.zeros((X.shape[0], 1))
        for i, formula in enumerate(formulas):
            x = X[:, i]
            predictions[:, 0] += self._eval_formula(formula, x)
        return predictions

    def save_symbolic_formula(self, filename="output/final_symbolic_formula.txt"):
        formulas = self.get_symbolic_formula()
        all_formula = ""
        # For regression, formulas is a list of strings
        if not hasattr(self, 'classes_'):
            for i, form in enumerate(formulas):
                all_formula += f"Feature {i}: {form}\n"
        else:
            # For classifier: formulas is a nested list
            for i, feature_formulas in enumerate(formulas):
                for j, form in enumerate(feature_formulas):
                    all_formula += f"Feature {i} - Class {j}: {form}\n"
        with open(filename, "w") as f:
            f.write(all_formula)
        print(f"Symbolic formulas saved to {filename}")
    
    def compile_symbolic_formula(self, filename="output/final_symbolic_formula.txt"):
        import re
        from .utils import ADVANCED_LIB  # needed to retrieve basis functions
        with open(filename, "r") as f:
            content = f.read()
        # Regex to extract coefficient and function notation.
        # Matches patterns like: "(-?\d+\.\d+)\*?([\w\(\)\^]+)"
        matches = re.findall(r"(-?\d+\.\d+)\*?([\w\(\)\^]+)", content)
        compiled_terms = []
        for coef_str, func_name in matches:
            try:
                coef = float(coef_str)
                # Search for a matching basis function in ADVANCED_LIB (e.g. 'x', 'x^2', etc.)
                for key, (notation, func) in ADVANCED_LIB.items():
                    if notation.strip() == func_name.strip():
                        compiled_terms.append((coef, func))
                        break
            except Exception:
                continue
        def prediction_function(x):
            pred = 0
            for coef, func in compiled_terms:
                pred += coef * func(x)
            return pred
        return prediction_function

class OIKANRegressor(BaseOIKAN, RegressorMixin):
    """OIKAN implementation for regression tasks"""
    def fit(self, X, y, epochs=100, lr=0.01, batch_size=32, verbose=True):
        X, y = self._validate_data(X, y)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        if self.model is None:
            self.model = self._build_network(X.shape[1], y.shape[1])
            
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.model(X)
            loss = criterion(y_pred, y)
            
            if torch.isnan(loss):
                print("Warning: NaN loss detected, reinitializing model...")
                self.model = None
                return self.fit(X, y, epochs, lr/10, batch_size, verbose)
                
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise NotFittedError("Model must be fitted before prediction")
            
        X = self._validate_data(X)[0]
        self.model.eval()
        with torch.no_grad():
            return self.model(X).cpu().numpy()

class OIKANClassifier(BaseOIKAN, ClassifierMixin):
    """OIKAN implementation for classification tasks"""
    def fit(self, X, y, epochs=100, lr=0.01, batch_size=32, verbose=True):
        X, y = self._validate_data(X, y)
        self.classes_ = torch.unique(y)
        n_classes = len(self.classes_)
        
        if self.model is None:
            self.model = self._build_network(X.shape[1], 1 if n_classes == 2 else n_classes)
            
        criterion = (nn.BCEWithLogitsLoss() if n_classes == 2 
                    else nn.CrossEntropyLoss())
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = self.model(X)
            if n_classes == 2:
                y_tensor = y.float()
                logits = logits.squeeze()
            else:
                y_tensor = y.long()
            loss = criterion(logits, y_tensor)
            loss.backward()
            optimizer.step()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
        self._is_fitted = True
        return self

    def predict_proba(self, X):
        if not self._is_fitted:
            raise NotFittedError("Model must be fitted before prediction")
            
        X = self._validate_data(X)[0]
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            if len(self.classes_) == 2:
                probs = torch.sigmoid(logits)
                return np.column_stack([1 - probs.cpu().numpy(), probs.cpu().numpy()])
            else:
                return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def symbolic_predict_proba(self, X):
        """Predict class probabilities using only the extracted symbolic formula."""
        if not self._is_fitted:
            raise NotFittedError("Model must be fitted before prediction")
            
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            
        # Scale input data similar to training
        X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
            
        formulas = self.get_symbolic_formula()
        n_classes = len(self.classes_)
        predictions = np.zeros((X.shape[0], n_classes))
        
        # Evaluate each feature's contribution to each class
        for i in range(X.shape[1]):  # For each feature
            x = X_scaled[:, i]  # Use scaled data
            for j in range(n_classes):  # For each class
                formula = formulas[i][j]
                if formula and formula != "0":
                    predictions[:, j] += self._eval_formula(formula, x)
        
        # Apply softmax with temperature for better separation
        temperature = 1.0
        exp_preds = np.exp(predictions / temperature)
        probas = exp_preds / exp_preds.sum(axis=1, keepdims=True)
        
        # Clip probabilities to avoid numerical issues
        probas = np.clip(probas, 1e-7, 1.0)
        probas = probas / probas.sum(axis=1, keepdims=True)
        
        return probas
    
    def get_symbolic_formula(self):
        """Extract symbolic formulas for all features and outputs."""
        if not self._is_fitted:
            raise NotFittedError("Model must be fitted before extracting formulas")
        
        n_features = self.model[0].input_dim
        n_classes = len(self.classes_)
        formulas = [[[] for _ in range(n_classes)] for _ in range(n_features)]
        
        first_layer = self.model[0]
        for i in range(n_features):
            for j in range(n_classes):
                edge = first_layer.edges[i][j]
                weight = first_layer.combination_weights[i, j].item()
                
                if abs(weight) > 1e-4:
                    # Get the edge formula and scale by the weight
                    edge_formula = edge.get_symbolic_repr()
                    terms = []
                    for term in edge_formula.split(" + "):
                        if term and term != "0":
                            if "*" in term:
                                coef, rest = term.split("*", 1)
                                coef = float(coef) * weight
                                terms.append(f"{coef:.4f}*{rest}")
                            else:
                                terms.append(f"{float(term) * weight:.4f}")
                    
                    formulas[i][j] = " + ".join(terms) if terms else "0"
                else:
                    formulas[i][j] = "0"
        
        return formulas

    def symbolic_predict(self, X):
        """Predict classes using only the extracted symbolic formula."""
        proba = self.symbolic_predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]