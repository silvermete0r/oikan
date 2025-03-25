import numpy as np
from oikan.model import OIKAN
from oikan.visualize import visualize_regression
from oikan.metrics import evaluate_regression

if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.rand(1000, 2)
    y = X[:, 0]**2 + np.sin(X[:, 1])
    
    model = OIKAN(mode='regression')
    model.fit(X, y, epochs=100, lr=0.01)

    y_pred = model.predict(X)

    evaluate_regression(y, y_pred)

    visualize_regression(X, y, y_pred)

    print('OIKAN Symbolic Formula:', model.extract_symbolic_formula(X))

    print('OIKAN LaTeX-formatted Symbolic Formula:', model.extract_latex_formula(X))

    model.test_symbolic_formula(X)

    model.plot_symbolic_formula(X)