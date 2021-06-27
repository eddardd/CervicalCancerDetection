import numpy as np
from utils.evaluation import accuracy


class LogisticRegression:
    def __init__(self, batch_size=256, learning_rate=1e-2, momentum=0.99, n_epochs=100, verbose=False, penalty=0.0):
        self.coefficients = None
        self.intercept = None
        self.fitted = False
        self.memory = None
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.penalty = penalty

    def __softmax(self, v):
        tmp = np.exp(v)
        return tmp / tmp.sum(axis=0)

    def __loss(self, y_true, y_pred):
        return - np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def fit(self, X, y):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        try:
            n_classes = y.shape[1]
        except IndexError:
            n_classes = 1
        n_batches = n_samples // self.batch_size
        self.history = {
            'loss': [],
            'acc': []
        }

        best_acc = - np.infty
        best_coefficients = None
        best_intercept = None
        self.best_epoch = -1

        self.coefficients = np.sqrt(1 / (n_features * n_classes)) * np.random.randn(n_features, n_classes)
        self.intercept = np.sqrt(1 / n_classes) * np.random.randn(n_classes)
        self.memory = [np.zeros_like(self.coefficients), np.zeros_like(self.intercept)]

        if self.verbose: print("|{:^25}|{:^25}|{:^25}|".format("Epoch", "Loss", "Accuracy"))
        for epoch in range(self.n_epochs):
            losses = []
            accs = []
            for i in range(n_batches):
                x_batch = X[i * self.batch_size: (i + 1) * self.batch_size]
                y_batch = y[i * self.batch_size: (i + 1) * self.batch_size]

                v = np.dot(x_batch, self.coefficients) + self.intercept
                y_pred = self.__softmax(v)

                g_coeff = (np.dot(x_batch.T, (y_batch - y_pred)) + self.penalty * self.coefficients) / self.batch_size
                g_intercept = np.mean(y_batch - y_pred, axis=0)

                V_prev_coeff, V_prev_intercept = self.memory
                d_coeff = self.momentum * V_prev_coeff + (1 - self.momentum) * g_coeff
                d_intercept = self.momentum * V_prev_intercept + (1 - self.momentum) * g_intercept
                self.memory = [d_coeff, d_intercept]
                
                self.coefficients = self.coefficients + self.learning_rate * d_coeff
                self.intercept = self.intercept + self.learning_rate * d_intercept

                losses.append(self.__loss(y_batch, y_pred))
                accs.append(accuracy(y_batch.argmax(axis=1), y_pred.argmax(axis=1)))
            if np.mean(accs) > best_acc:
                best_coefficients = self.coefficients
                best_intercept = self.intercept
                self.best_epoch = epoch
                best_acc = np.mean(accs)
            self.history['loss'].append(np.mean(losses))
            self.history['acc'].append(np.mean(accs))
            print("|{:^25}|{:^25}|{:^25}|".format(epoch, np.mean(losses), np.mean(accs)))
        self.fitted = True
        if best_coefficients is not None: self.coefficients = best_coefficients
        if best_intercept is not None: self.intercept = best_intercept

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Trying to predict using an unfitted model")
        v = np.dot(X, self.coefficients) + self.intercept
        return self.__softmax(v)