import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    m, n = X.shape
    y = y.reshape(-1, 1)  # Ensure y is a column vector
    theta = np.zeros((n, 1))  # Initialize weights to zeros

    for _ in range(iterations):
        predict = X @ theta
        errors = predict - y;
        gradient = 1/m * X.T @ errors
        theta = theta - alpha * gradient

    return np.round(theta.flatten(), 4)