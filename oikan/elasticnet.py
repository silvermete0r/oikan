import torch.nn as nn
import torch
import numpy as np

class ElasticNet(nn.Module):
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=False, max_iter=1000, tol=1e-4, random_state=None):
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n_samples, n_features = X.shape
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_targets = y.shape[1]

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        W = torch.zeros((n_features, n_targets), requires_grad=True, dtype=torch.float32)
        if self.fit_intercept:
            b = torch.zeros(n_targets, requires_grad=True, dtype=torch.float32)
        else:
            b = None

        optimizer = torch.optim.Adam([W] + ([b] if b is not None else []), lr=0.05)

        prev_loss = None
        for _ in range(self.max_iter):
            optimizer.zero_grad()
            pred = X_tensor @ W
            if b is not None:
                pred = pred + b
            mse = torch.mean((pred - y_tensor) ** 2)
            l1 = torch.sum(torch.abs(W))
            l2 = torch.sum(W ** 2)
            loss = mse + self.alpha * (self.l1_ratio * l1 + (1 - self.l1_ratio) * l2)
            loss.backward()
            optimizer.step()
            if prev_loss is not None and abs(prev_loss - loss.item()) < self.tol:
                break
            prev_loss = loss.item()

        self.coef_ = W.detach().cpu().numpy().T if n_targets > 1 else W.detach().cpu().numpy().flatten()
        if b is not None:
            self.intercept_ = b.detach().cpu().numpy()
        else:
            self.intercept_ = np.zeros(n_targets) if n_targets > 1 else 0.0
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.coef_ is None:
            raise RuntimeError("Model not fitted yet.")
        W = self.coef_.T if self.coef_.ndim == 2 else self.coef_
        y_pred = X @ W
        if self.intercept_ is not None:
            y_pred += self.intercept_
        return y_pred