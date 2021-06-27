import numpy as np

class LeastSquaresClassifier:
    def __init__(self, penalty=0.0):
        self.coefficients = None
        self.intercept = None
        self.fitted = False
        self.penalty = penalty

    def fit(self, X, y):
        self.fitted = True
        I = np.eye(X.shape[1])
        self.coefficients = np.dot(np.linalg.pinv(np.dot(X.T, X) + self.penalty * I), np.dot(X.T, y))
        self.intercept = np.mean(y, axis=0) - np.dot(self.coefficients.T, np.mean(X, axis=0))
        pass

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Trying to predict using an unfitted model")
        return np.dot(X, self.coefficients) + self.intercept
