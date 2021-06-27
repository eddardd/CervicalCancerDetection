def accuracy(y_true, y_pred):
    return sum(1 * (y_pred == y_true)) / len(y_true)
