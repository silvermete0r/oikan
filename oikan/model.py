import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from abc import ABC, abstractmethod
import json
from .neural import TabularNet
from .utils import evaluate_basis_functions, get_features_involved
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import sys

class OIKAN(ABC):
    """
    Base class for the OIKAN neuro-symbolic framework.
    
    Parameters:
    -----------
    hidden_sizes : list, optional (default=[64, 64])
        List of hidden layer sizes for the neural network.
    activation : str, optional (default='relu')
        Activation function for the neural network ('relu', 'tanh', 'leaky_relu', 'elu', 'swish', 'gelu').
    augmentation_factor : int, optional (default=10)
        Number of augmented samples per original sample.
    polynomial_degree : int, optional (default=2)
        Maximum degree of polynomial features for symbolic regression.
    alpha : float, optional (default=0.1)
        L1 regularization strength for Lasso in symbolic regression.
    sigma : float, optional (default=0.1)
        Standard deviation of Gaussian noise for data augmentation.
    epochs : int, optional (default=100)
        Number of epochs for neural network training.
    lr : float, optional (default=0.001)
        Learning rate for neural network optimization.
    batch_size : int, optional (default=32)
        Batch size for neural network training.
    verbose : bool, optional (default=False)
        Whether to display training progress.
    evaluate_nn : bool, optional (default=False)
        Whether to evaluate neural network performance before full training.
    """
    def __init__(self, hidden_sizes=[64, 64], activation='relu', augmentation_factor=10, 
                 polynomial_degree=2, alpha=0.1, sigma=0.1, epochs=100, lr=0.001, batch_size=32, 
                 verbose=False, evaluate_nn=False):
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.augmentation_factor = augmentation_factor
        self.polynomial_degree = polynomial_degree
        self.alpha = alpha
        self.sigma = sigma
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.verbose = verbose
        self.evaluate_nn = evaluate_nn
        self.neural_net = None
        self.symbolic_model = None
        self.evaluation_done = False

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def get_formula(self):
        """Returns the symbolic formula(s) as a string (regression) or list of strings (classification)."""
        if self.symbolic_model is None:
            raise ValueError("Model not fitted yet.")
        basis_functions = self.symbolic_model['basis_functions']
        if 'coefficients' in self.symbolic_model:
            coefficients = self.symbolic_model['coefficients']
            formula = " + ".join([f"{coefficients[i]:.3f}*{basis_functions[i]}" 
                                for i in range(len(coefficients)) if coefficients[i] != 0])
            return formula if formula else "0"
        else:
            formulas = []
            for c, coef in enumerate(self.symbolic_model['coefficients_list']):
                formula = " + ".join([f"{coef[i]:.3f}*{basis_functions[i]}" 
                                    for i in range(len(coef)) if coef[i] != 0])
                formulas.append(f"Class {self.classes_[c]}: {formula if formula else '0'}")
            return formulas

    def feature_importances(self):
        """
        Computes the importance of each original feature based on the symbolic model.
        
        Returns:
        --------
        numpy.ndarray : Normalized feature importances.
        """
        if self.symbolic_model is None:
            raise ValueError("Model not fitted yet.")
        basis_functions = self.symbolic_model['basis_functions']
        n_features = self.symbolic_model['n_features']
        importances = np.zeros(n_features)
        
        # Handle regression case
        if 'coefficients' in self.symbolic_model:
            coefficients = self.symbolic_model['coefficients']
            for i, func in enumerate(basis_functions):
                if coefficients[i] != 0:
                    features_involved = get_features_involved(func)
                    for idx in features_involved:
                        importances[idx] += np.abs(coefficients[i])
        # Handle classification case with multiple coefficient sets
        else:
            for coef in self.symbolic_model['coefficients_list']:
                for i, func in enumerate(basis_functions):
                    if coef[i] != 0:
                        features_involved = get_features_involved(func)
                        for idx in features_involved:
                            importances[idx] += np.abs(coef[i])
        
        total = importances.sum()
        return importances / total if total > 0 else importances

    def save(self, path):
        """
        Saves the symbolic model to a .json file.
        
        Parameters:
        -----------
        path : str
            File path to save the model. Should end with .json
        """
        if self.symbolic_model is None:
            raise ValueError("Model not fitted yet.")
            
        if not path.endswith('.json'):
            path = path + '.json'
            
        # Convert numpy arrays and other non-serializable types to lists
        model_data = {
            'n_features': self.symbolic_model['n_features'],
            'degree': self.symbolic_model['degree'],
            'basis_functions': self.symbolic_model['basis_functions']
        }
        
        if 'coefficients' in self.symbolic_model:
            model_data['coefficients'] = self.symbolic_model['coefficients']
        else:
            model_data['coefficients_list'] = [coef for coef in self.symbolic_model['coefficients_list']]
            if hasattr(self, 'classes_'):
                model_data['classes'] = self.classes_.tolist()
        
        with open(path, 'w') as f:
            json.dump(model_data, f, indent=2)

    def load(self, path):
        """
        Loads the symbolic model from a .json file.
        
        Parameters:
        -----------
        path : str
            File path to load the model from. Should end with .json
        """
        if not path.endswith('.json'):
            path = path + '.json'
            
        with open(path, 'r') as f:
            model_data = json.load(f)
            
        self.symbolic_model = {
            'n_features': model_data['n_features'],
            'degree': model_data['degree'],
            'basis_functions': model_data['basis_functions']
        }
        
        if 'coefficients' in model_data:
            self.symbolic_model['coefficients'] = model_data['coefficients']
        else:
            self.symbolic_model['coefficients_list'] = model_data['coefficients_list']
            if 'classes' in model_data:
                self.classes_ = np.array(model_data['classes'])

    def _evaluate_neural_net(self, X, y, output_size, loss_fn):
        """Evaluates neural network performance on train-test split."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        input_size = X.shape[1]
        self.neural_net = TabularNet(input_size, self.hidden_sizes, output_size, self.activation)
        optimizer = optim.Adam(self.neural_net.parameters(), lr=self.lr)
        
        # Train on the training set
        self._train_neural_net(X_train, y_train, output_size, loss_fn)
        
        # Evaluate on test set
        self.neural_net.eval()
        with torch.no_grad():
            y_pred = self.neural_net(torch.tensor(X_test, dtype=torch.float32))
            if output_size == 1:  # Regression
                y_pred = y_pred.numpy()
                score = r2_score(y_test, y_pred)
                metric_name = "R² Score"
            else:  # Classification
                y_pred = torch.argmax(y_pred, dim=1).numpy()
                y_test = torch.argmax(y_test, dim=1).numpy()
                score = accuracy_score(y_test, y_pred)
                metric_name = "Accuracy"
        
        print(f"\nNeural Network Evaluation:")
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        print(f"{metric_name}: {score:.4f}")
        
        # Ask user for confirmation
        response = input("\nProceed with full training and symbolic regression? [Y/n]: ").lower()
        if response not in ['y', 'yes']:
            sys.exit("Training cancelled by user.")

        # Retrain on full dataset
        self._train_neural_net(X, y, output_size, loss_fn)

    def _train_neural_net(self, X, y, output_size, loss_fn):
        """Trains the neural network on the input data."""
        if self.evaluate_nn and not self.evaluation_done:
            self.evaluation_done = True
            self._evaluate_neural_net(X, y, output_size, loss_fn)
            return
            
        input_size = X.shape[1]
        if self.neural_net is None:
            self.neural_net = TabularNet(input_size, self.hidden_sizes, output_size, self.activation)
        optimizer = optim.Adam(self.neural_net.parameters(), lr=self.lr)
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), 
                                               torch.tensor(y, dtype=torch.float32))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.neural_net.train()

        if self.verbose:
            from tqdm import tqdm
            epoch_iterator = tqdm(range(self.epochs), desc="Training")
        else:
            epoch_iterator = range(self.epochs)

        for epoch in epoch_iterator:
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.neural_net(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if self.verbose:
                epoch_iterator.set_postfix({'loss': f'{total_loss/len(loader):.4f}'})

    def _generate_augmented_data(self, X):
        """Generates augmented data by adding Gaussian noise."""
        n_samples = X.shape[0]
        X_aug = []
        for _ in range(self.augmentation_factor):
            noise = np.random.normal(0, self.sigma, X.shape)
            X_perturbed = X + noise
            X_aug.append(X_perturbed)
        return np.vstack(X_aug)

    def _perform_symbolic_regression(self, X, y):
        """Performs symbolic regression using polynomial features and Lasso."""
        poly = PolynomialFeatures(degree=self.polynomial_degree, include_bias=True)
        X_poly = poly.fit_transform(X)
        model = Lasso(alpha=self.alpha, fit_intercept=False)
        model.fit(X_poly, y)
        if len(y.shape) == 1 or y.shape[1] == 1:
            coef = model.coef_.flatten()
            selected_indices = np.where(np.abs(coef) > 1e-6)[0]
            self.symbolic_model = {
                'n_features': X.shape[1],
                'degree': self.polynomial_degree,
                'basis_functions': poly.get_feature_names_out()[selected_indices].tolist(),
                'coefficients': coef[selected_indices].tolist()
            }
        else:
            coefficients_list = []
            # Note: Using the same basis functions across classes for simplicity
            selected_indices = set()
            for c in range(y.shape[1]):
                coef = model.coef_[c]
                indices = np.where(np.abs(coef) > 1e-6)[0]
                selected_indices.update(indices)
            selected_indices = list(selected_indices)
            basis_functions = poly.get_feature_names_out()[selected_indices].tolist()
            for c in range(y.shape[1]):
                coef = model.coef_[c]
                coef_selected = coef[selected_indices].tolist()
                coefficients_list.append(coef_selected)
            self.symbolic_model = {
                'n_features': X.shape[1],
                'degree': self.polynomial_degree,
                'basis_functions': basis_functions,
                'coefficients_list': coefficients_list
            }

class OIKANRegressor(OIKAN):
    """OIKAN model for regression tasks."""
    def fit(self, X, y):
        """
        Fits the regressor to the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        """
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        self._train_neural_net(X, y, output_size=1, loss_fn=nn.MSELoss())
        if self.verbose:
            print(f"Original data: features shape: {X.shape} | target shape: {y.shape}")
        X_aug = self._generate_augmented_data(X)
        self.neural_net.eval()
        with torch.no_grad():
            y_aug = self.neural_net(torch.tensor(X_aug, dtype=torch.float32)).detach().numpy()
        if self.verbose:
            print(f"Augmented data: features shape: {X_aug.shape} | target shape: {y_aug.shape}")
        self._perform_symbolic_regression(X_aug, y_aug)

    def predict(self, X):
        """
        Predicts target values for the input data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        if self.symbolic_model is None:
            raise ValueError("Model not fitted yet.")
        X = np.asarray(X)
        X_transformed = evaluate_basis_functions(X, self.symbolic_model['basis_functions'], 
                                               self.symbolic_model['n_features'])
        return np.dot(X_transformed, self.symbolic_model['coefficients'])

class OIKANClassifier(OIKAN):
    """OIKAN model for classification tasks."""
    def fit(self, X, y):
        """
        Fits the classifier to the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels.
        """
        X = np.asarray(X)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.classes_ = le.classes_
        n_classes = len(self.classes_)
        y_onehot = nn.functional.one_hot(torch.tensor(y_encoded), num_classes=n_classes).float()
        self._train_neural_net(X, y_onehot, output_size=n_classes, loss_fn=nn.CrossEntropyLoss())
        if self.verbose:
            print(f"Original data: features shape: {X.shape} | target shape: {y.shape}")
        X_aug = self._generate_augmented_data(X)
        self.neural_net.eval()
        with torch.no_grad():
            logits_aug = self.neural_net(torch.tensor(X_aug, dtype=torch.float32)).detach().numpy()
        if self.verbose:
            print(f"Augmented data: features shape: {X_aug.shape} | target shape: {logits_aug.shape}")
        self._perform_symbolic_regression(X_aug, logits_aug)

    def predict(self, X):
        """
        Predicts class labels for the input data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        if self.symbolic_model is None:
            raise ValueError("Model not fitted yet.")
        X = np.asarray(X)
        X_transformed = evaluate_basis_functions(X, self.symbolic_model['basis_functions'], 
                                               self.symbolic_model['n_features'])
        logits = np.dot(X_transformed, np.array(self.symbolic_model['coefficients_list']).T)
        probabilities = nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
        return self.classes_[np.argmax(probabilities, axis=1)]