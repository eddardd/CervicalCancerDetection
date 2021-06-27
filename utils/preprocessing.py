import numpy as np

from scipy.stats import skew
from scipy.stats import boxcox


def apply_boxcox(X, skewThr=0.1):
    """Applies Box-Cox transformtion on columns of X (features) whose skewness is greater than skewThr.

    Parameters
    ----------
    X : :class:`numpy.ndarray`
        Features matrix of shape (n, d).
    skewThr : float
        Threshold for the application of Box-Cox transformation
    """
    _X = X.copy()
    predictor_skewness = [skew(X[:, i]) for i in range(X.shape[1])]
    predictors_for_log_transform = [
        j for j, gamma_j in enumerate(predictor_skewness) if abs(gamma_j) > skewThr
    ]

    for j in predictors_for_log_transform:
        _X[:, j], _ = boxcox(_X[:, j])

    return _X


def standardize_data(X, eps=1e-9):
    """Subtracts mean and divides by standard deviation

    Parameters
    ----------
    X : :class:`numpy.ndarray`
        Features matrix of shape (n, d).
    eps : float
        Value for preventing division by zero.
    """
    _X = X.copy()
    mean = np.mean(_X, axis=0)
    std = np.std(_X, axis=0)

    return (_X - mean) / (std + eps)


def pipeline(X, transformations, **kwargs):
    """Pipeline of preprocessing methods on features. Each preprocessing function should accept X as its first argument,
    and should output a single array, that corresponds to the preprocessed data.

    Parameters
    ----------
    X : :class:`numpy.ndarray`
        Features matrix of shape (n, d).
    transformations : list
        List of preprocessing functions
    """
    _X = X.copy()
    for T in transformations:
        _X = T(_X, **kwargs)

    return _X


def one_hot_encode(y, n_classes):
    """Uses one-hot encoding for encoding multi-class vectors.

    Parameters
    ----------
    y : :class:`numpy.ndarray`
        Vector of shape (n,) containing integer values corresponding to classes on :math:`{0, \cdots, K-1}`
    """
    T = np.zeros([y.shape[0], n_classes])

    for i, yi in enumerate(y):
        T[i, yi] = 1

    return T