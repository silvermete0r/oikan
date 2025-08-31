import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import PolynomialFeatures
from abc import ABC, abstractmethod
import json
from .elasticnet import ElasticNet
from .neural import TabularNet
from .utils import evaluate_basis_functions, get_features_involved, sympify_formula, get_latex_formula
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from .exceptions import *
import sys
from tqdm import tqdm

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
    alpha : float, optional (default=1.0)
        ElasticNet regularization strength.
    l1_ratio: float, optional (default=0.5)
        ElasticNet mixing parameter (0 <= l1_ratio <= 1).
        0 is equivalent to Ridge regression, 1 is equivalent to Lasso.
    sigma : float, optional (default=5.0)
        Standard deviation of Gaussian noise for data augmentation.
    top_k : int, optional (default=5)
        Number of top features to select in hierarchical symbolic regression.
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
    random_state: int, optional (default=None)
        Random seed for reproducibility.
    """
    def __init__(self, hidden_sizes=[64, 64], activation='relu', augmentation_factor=10, 
                 alpha=1.0, l1_ratio=0.5, sigma=5.0, epochs=100, lr=0.001, batch_size=32, 
                 verbose=False, evaluate_nn=False, top_k=5, random_state=None):
        if not isinstance(hidden_sizes, list) or not all(isinstance(x, int) and x > 0 for x in hidden_sizes):
            raise InvalidParameterError("hidden_sizes must be a list of positive integers")
        if activation not in ['relu', 'tanh', 'leaky_relu', 'elu', 'swish', 'gelu']:
            raise InvalidParameterError(f"Unsupported activation function: {activation}")
        if not isinstance(augmentation_factor, int) or augmentation_factor < 1:
            raise InvalidParameterError("augmentation_factor must be a positive integer")
        if not isinstance(top_k, int) or top_k < 1:
            raise InvalidParameterError("top_k must be a positive integer")
        if not 0 < lr < 1:
            raise InvalidParameterError("Learning rate must be between 0 and 1")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise InvalidParameterError("batch_size must be a positive integer")
        if not isinstance(epochs, int) or epochs < 1:
            raise InvalidParameterError("epochs must be a positive integer")
        if not 0 <= alpha <= 1:
            raise InvalidParameterError("alpha must be between 0 and 1")
        if not 0 <= l1_ratio <= 1:
            raise InvalidParameterError("l1_ratio must be between 0 and 1")
        if sigma <= 0:
            raise InvalidParameterError("sigma must be positive")
        
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.augmentation_factor = augmentation_factor
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.sigma = sigma
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.verbose = verbose
        self.evaluate_nn = evaluate_nn
        self.top_k = top_k
        self.neural_net = None
        self.symbolic_model = None
        self.evaluation_done = False
        self.random_state = random_state
        
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def get_formula(self, type='original'): 
        """
        Returns the symbolic formula(s) as a string (regression) or list of strings (classification). 
        
        Parameter:
        --------
        type : str, optional (default='original') other options: 'sympy', 'latex'
            'original' returns the original formula with coefficients, 'sympy' returns sympy simplified formula.
        """
        if type.lower() not in ['original', 'sympy', 'latex']:
            raise InvalidParameterError("Invalid type. Choose 'original', 'sympy', 'latex'.")
        if self.symbolic_model is None:
            raise ValueError("Model not fitted yet.")
        basis_functions = self.symbolic_model['basis_functions']
        if type.lower() == 'original':
            if 'coefficients' in self.symbolic_model:
                coefficients = self.symbolic_model['coefficients']
                formula = " + ".join([f"{coefficients[i]:.6f}*{basis_functions[i]}" 
                                    for i in range(len(coefficients)) if coefficients[i] != 0])
                return formula if formula else "0"
            else:
                formulas = []
                for c, coef in enumerate(self.symbolic_model['coefficients_list']):
                    formula = " + ".join([f"{coef[i]:.6f}*{basis_functions[i]}" 
                                        for i in range(len(coef)) if coef[i] != 0])
                    formulas.append(f"Class {self.classes_[c]}: {formula if formula else '0'}")
                return formulas
        elif type.lower() == 'sympy':
            if 'coefficients' in self.symbolic_model:
                formula = sympify_formula(self.symbolic_model['basis_functions'], self.symbolic_model['coefficients'], self.symbolic_model['n_features'])
                return formula
            else: 
                formulas = []
                for c, coef in enumerate(self.symbolic_model['coefficients_list']):
                    formula = sympify_formula(self.symbolic_model['basis_functions'], coef, self.symbolic_model['n_features'])
                    formulas.append(f"Class {self.classes_[c]}: {formula}")
                return formulas
        else:
            if 'coefficients' in self.symbolic_model:
                formula = get_latex_formula(self.symbolic_model['basis_functions'], self.symbolic_model['coefficients'], self.symbolic_model['n_features'])
                return formula
            else: 
                formulas = []
                for c, coef in enumerate(self.symbolic_model['coefficients_list']):
                    formula = get_latex_formula(self.symbolic_model['basis_functions'], coef, self.symbolic_model['n_features'])
                    formulas.append(f"Class {self.classes_[c]}: {formula}")
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
            raise ModelNotFittedError("Model must be fitted before saving")
        
        if not path.endswith('.json'):
            path = path + '.json'
        
        try:    
            # Convert numpy arrays and other non-serializable types to lists
            model_data = {
                'n_features': self.symbolic_model['n_features'],
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
        except Exception as e:
            raise ModelSerializationError(f"Failed to save model: {str(e)}")
        
        if self.verbose:
            print(f"Model saved to {path}")

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
        
        try:
            with open(path, 'r') as f:
                model_data = json.load(f)
                
            self.symbolic_model = {
                'n_features': model_data['n_features'],
                'basis_functions': model_data['basis_functions']
            }
            
            if 'coefficients' in model_data:
                self.symbolic_model['coefficients'] = model_data['coefficients']
            else:
                self.symbolic_model['coefficients_list'] = model_data['coefficients_list']
                if 'classes' in model_data:
                    self.classes_ = np.array(model_data['classes'])
        except Exception as e:
            raise ModelSerializationError(f"Failed to load model: {str(e)}")
        
        if self.verbose:
            print(f"Model loaded from {path}")

    def _evaluate_neural_net(self, X, y, output_size, loss_fn):
        """Evaluates neural network performance on train-test split."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        input_size = X.shape[1]
        self.neural_net = TabularNet(input_size, self.hidden_sizes, output_size, self.activation)
        
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
                                         y.clone().detach())
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
        if self.augmentation_factor == 1:
            return np.array([]).reshape(0, X.shape[1])
            
        X_aug = []
        for _ in range(self.augmentation_factor - 1):
            noise = np.random.normal(0, self.sigma, X.shape)
            X_perturbed = X + noise
            X_aug.append(X_perturbed)

        return np.vstack(X_aug)

    def _perform_symbolic_regression(self, X, y):
        """
        Performs hierarchical symbolic regression using a two-stage approach.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : array-like of shape (n_samples,) or (n_samples, n_classes)
            Target values or logits.
        """
        n_features = X.shape[1]
        self.top_k = min(self.top_k, n_features)
        
        if self.top_k < 1:
            raise InvalidParameterError("top_k must be at least 1")

        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise NumericalInstabilityError("Input data contains NaN values")

        if np.any(np.isinf(X)) or np.any(np.isinf(y)):
            raise NumericalInstabilityError("Input data contains infinite values")

        if self.verbose:
            print("\nStage 1: Coarse Model Fitting")
            
        coarse_degree = 2  # Fixed low degree for coarse model
        poly_coarse = PolynomialFeatures(degree=coarse_degree, include_bias=True)
        
        if self.verbose:
            print("Generating polynomial features...")
        X_poly_coarse = poly_coarse.fit_transform(X)
        
        if self.verbose:
            print("Fitting coarse elastic net model...")
        model_coarse = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, fit_intercept=False, random_state=self.random_state)
        model_coarse.fit(X_poly_coarse, y)

        if self.verbose:
            print("Computing feature importances...")
        basis_functions_coarse = poly_coarse.get_feature_names_out()
        if len(y.shape) == 1 or y.shape[1] == 1:
            coef_coarse = model_coarse.coef_.flatten()
        else:
            coef_coarse = np.sum(np.abs(model_coarse.coef_), axis=0)

        importances = np.zeros(X.shape[1])
        for i, func in enumerate(tqdm(basis_functions_coarse, disable=not self.verbose, desc="Analyzing features")):
            features_involved = get_features_involved(func)
            for idx in features_involved:
                importances[idx] += np.abs(coef_coarse[i])
        
        if np.all(importances == 0):
            raise FeatureExtractionError("Failed to compute feature importances - all values are zero")

        # Select top K features
        top_k_indices = np.argsort(importances)[::-1][:self.top_k]

        if self.verbose:
            print(f"\nStage 2: Refined Model with top {self.top_k} features")
            print("Generating additional non-linear features...")

        additional_features = []
        additional_names = []
        for i in tqdm(top_k_indices, disable=not self.verbose, desc="Generating features"):
            # Higher-degree polynomial
            additional_features.append(X[:, i]**3)
            additional_names.append(f'x{i}^3')
            # Non-linear transformations
            additional_features.append(np.log1p(np.abs(X[:, i])))
            additional_names.append(f'log1p_x{i}')
            additional_features.append(np.exp(np.clip(X[:, i], -10, 10)))
            additional_names.append(f'exp_x{i}')
            additional_features.append(np.sin(X[:, i]))
            additional_names.append(f'sin_x{i}')
        
        if self.verbose:
            print("Combining features and fitting final model...")
        X_additional = np.column_stack(additional_features)
        X_refined = np.hstack([X_poly_coarse, X_additional])
        basis_functions_refined = list(basis_functions_coarse) + additional_names

        model_refined = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, fit_intercept=False, random_state=self.random_state)
        model_refined.fit(X_refined, y)

        if self.verbose:
            print("Building final symbolic model...")

        # Store symbolic model
        if len(y.shape) == 1 or y.shape[1] == 1:
            # Regression
            coef_refined = model_refined.coef_.flatten()
            selected_indices = np.where(np.abs(coef_refined) > 1e-6)[0]
            self.symbolic_model = {
                'n_features': X.shape[1],
                'basis_functions': [basis_functions_refined[i] for i in selected_indices],
                'coefficients': coef_refined[selected_indices].tolist()
            }
        else:
            # Classification
            coefficients_list = []
            selected_indices = set()
            for c in tqdm(range(y.shape[1]), disable=not self.verbose, desc="Processing classes"):
                coef = model_refined.coef_[c]
                indices = np.where(np.abs(coef) > 1e-6)[0]
                selected_indices.update(indices)
            selected_indices = list(selected_indices)
            basis_functions = [basis_functions_refined[i] for i in selected_indices]
            for c in range(y.shape[1]):
                coef = model_refined.coef_[c]
                coef_selected = coef[selected_indices].tolist()
                coefficients_list.append(coef_selected)
            self.symbolic_model = {
                'n_features': X.shape[1],
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
        
        if self.augmentation_factor > 1:
            self._train_neural_net(X, y, output_size=1, loss_fn=nn.MSELoss())
            
            if self.verbose:
                print(f"Original data: features shape: {X.shape} | target shape: {y.shape} | size: {X.nbytes / (1024 * 1024):.2f} MB")
            
            X_aug = self._generate_augmented_data(X)
            
            self.neural_net.eval()
            with torch.no_grad():
                y_aug = self.neural_net(torch.tensor(X_aug, dtype=torch.float32)).detach().numpy()
            
            if self.verbose:
                print(f"Augmented data: features shape: {X_aug.shape} | target shape: {y_aug.shape} | size: {X_aug.nbytes / (1024 * 1024):.2f} MB")
            
            X_combined = np.vstack([X, X_aug])
            y_combined = np.vstack([y, y_aug])
        else:
            if self.verbose:
                print("Skipping neural network training (augmentation_factor=1)")
                print(f"Data: features shape: {X.shape} | target shape: {y.shape} | size: {X.nbytes / (1024 * 1024):.2f} MB")
            X_combined = X
            y_combined = y
            
        self._perform_symbolic_regression(X_combined, y_combined)
        if self.verbose:
            print("OIKANRegressor model training completed successfully!")

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
        
        if self.augmentation_factor > 1:
            self._train_neural_net(X, y_onehot, output_size=n_classes, loss_fn=nn.CrossEntropyLoss())
            
            if self.verbose:
                print(f"Original data: features shape: {X.shape} | target shape: {y.shape} | size: {X.nbytes / (1024 * 1024):.2f} MB")
            
            X_aug = self._generate_augmented_data(X)
            
            self.neural_net.eval()
            with torch.no_grad():
                logits_aug = self.neural_net(torch.tensor(X_aug, dtype=torch.float32)).detach().numpy()
            
            if self.verbose:
                print(f"Augmented data: features shape: {X_aug.shape} | target shape: {logits_aug.shape} | size: {X_aug.nbytes / (1024 * 1024):.2f} MB")
            
            X_combined = np.vstack([X, X_aug])
            y_combined = np.vstack([y_onehot.numpy(), logits_aug])
        else:
            if self.verbose:
                print("Skipping neural network training (augmentation_factor=1)")
                print(f"Data: features shape: {X.shape} | target shape: {y.shape} | size: {X.nbytes / (1024 * 1024):.2f} MB")
            X_combined = X
            y_combined = y_onehot.numpy()
            
        self._perform_symbolic_regression(X_combined, y_combined)
        if self.verbose:
            print("OIKANClassifier model training completed successfully!")

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