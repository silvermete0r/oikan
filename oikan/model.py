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
        
        # Create edge functions for each input-output connection
        self.edges = nn.ModuleList([
            nn.ModuleList([
                SymbolicEdge(input_dim=1) 
                for _ in range(output_dim)
            ])
            for _ in range(input_dim)
        ])
        
        # Initialize combination weights with small random values
        self.combination_weights = nn.Parameter(
            torch.randn(input_dim, output_dim) * 0.1
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        edge_outputs = torch.zeros(batch_size, self.input_dim, 
                                 self.output_dim).to(x.device)
        
        # Process each input dimension separately
        for i in range(self.input_dim):
            x_i = x[:, i:i+1]  # Keep dimension for broadcasting
            for j in range(self.output_dim):
                edge_outputs[:, i, j] = self.edges[i][j](x_i).squeeze()
        
        # Combine edge outputs with weights
        combined = edge_outputs * self.combination_weights.unsqueeze(0)
        return torch.sum(combined, dim=1)
    
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
    def __init__(self, hidden_dims=[64, 32], num_basis=10, degree=3):
        self.hidden_dims = hidden_dims
        self.num_basis = num_basis
        self.degree = degree
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._is_fitted = False
    
    def _build_network(self, input_dim, output_dim):
        layers = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(KANLayer(prev_dim, hidden_dim))
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
        """Extract symbolic formulas for all features and outputs."""
        if not self._is_fitted:
            raise NotFittedError("Model must be fitted before extracting formulas")
        
        if hasattr(self, 'classes_'):  # For classifier
            n_classes = len(self.classes_)
            formulas = []
            first_layer = self.model[0]
            for i in range(first_layer.input_dim):
                class_formulas = []
                for j in range(first_layer.output_dim):
                    formula = first_layer.edges[i][j].get_symbolic_repr()
                    class_formulas.append(formula)
                formulas.append(class_formulas)
            return formulas
        else:  # For regressor
            return [self.model[0].edges[i][0].get_symbolic_repr() 
                   for i in range(self.model[0].input_dim)]

    def get_feature_scores(self):
        """Get feature importance scores based on edge weights."""
        if not self._is_fitted:
            raise NotFittedError("Model must be fitted before computing scores")
        
        weights = self.model[0].combination_weights.detach().cpu().numpy()
        return np.mean(np.abs(weights), axis=1)

    def symbolic_predict(self, X):
        """Predict using only the extracted symbolic formula."""
        if not self._is_fitted:
            raise NotFittedError("Model must be fitted before prediction")
        
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            
        formulas = self.get_symbolic_formula()
        predictions = np.zeros((X.shape[0], 1))  # Default shape for regression
        
        if hasattr(self, 'classes_'):  # For classifier
            predictions = np.zeros((X.shape[0], len(self.classes_)))
            for i, feature_formulas in enumerate(formulas):
                for j, formula in enumerate(feature_formulas):
                    # Evaluate formula for each feature and class
                    x = X[:, i]
                    for name, (_, func) in ADVANCED_LIB.items():
                        if name in formula:
                            try:
                                term_value = eval(formula.replace(name, 'func(x)'))
                                predictions[:, j] += term_value
                            except:
                                continue
            return self.classes_[np.argmax(predictions, axis=1)]
        else:  # For regressor
            for i, formula in enumerate(formulas):
                x = X[:, i]
                for name, (_, func) in ADVANCED_LIB.items():
                    if name in formula:
                        try:
                            term_value = eval(formula.replace(name, 'func(x)'))
                            predictions[:, 0] += term_value
                        except:
                            continue
            return predictions

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
                feature_formulas = formulas[i][j].split(" + ")
                for term in feature_formulas:
                    if not term or term == "0":
                        continue
                        
                    for name, (notation, func) in ADVANCED_LIB.items():
                        if notation in term:
                            try:
                                # Extract coefficient
                                coef = float(term.split("*")[0])
                                # Apply transformation
                                term_value = coef * func(x)
                                predictions[:, j] += term_value
                            except:
                                continue
        
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