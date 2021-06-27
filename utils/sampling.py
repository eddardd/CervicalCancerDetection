import numpy as np


def stratified_sampling(X, y, proportion=0.8, shuffle=True):
    """Stratified Sampling function. Performs sampling on a dataset (x, y) using categories y.

    Parameters
    ----------
    X : :class:`numpy.ndarray`
        Features matrix of shape (n, d).
    y : :class:`numpy.ndarray`
        Label vector of shape (n,). Each element :math:`y_i` belongs to :math:`{0, 1, \cdots, K-1}`
    proportion : float
        Float value on the range [0, 1], specifying how many samples will be used for training, and how many
        will be used for testing.
    """
    # Initialize lists
    Xtr, ytr = [], []
    Xts, yts = [], []
    # Loops over each category
    for yi in np.unique(y):
        # Gets indices corresponding to i-th category
        ind_yi = np.where(y == yi)[0]

        # Extracts only samples belonging to i-th category
        Xi = X[ind_yi]

        # Determines how many samples will be used for training
        ntr = int(proportion * len(Xi))
        
        # Samples training and test data
        Xtr_i, Xts_i, _ = random_sampling(Xi, n_samples=ntr)

        # Append to lists
        Xtr.append(Xtr_i)
        ytr.append([yi] * len(Xtr_i))
        Xts.append(Xts_i)
        yts.append([yi] * len(Xts_i))
    # Concatenate lists
    Xtr, ytr = np.concatenate(Xtr, axis=0), np.concatenate(ytr, axis=0)
    Xts, yts = np.concatenate(Xts, axis=0), np.concatenate(yts, axis=0)

    if shuffle:
        ind_tr = np.arange(len(Xtr))
        np.random.shuffle(ind_tr)
        Xtr, ytr = Xtr[ind_tr], ytr[ind_tr]
    
        ind_ts = np.arange(len(Xts))
        np.random.shuffle(ind_ts)
        Xts, yts = Xts[ind_ts], yts[ind_ts]
    return Xtr, ytr, Xts, yts


def random_sampling(X, n_samples):
    """Random sampling function. Randomly samples 'n_samples' samples from array X

    Parameters
    ----------
    X : :class:`numpy.ndarray`
        Features matrix of shape (n, d).
    n_samples : int
        Integer specifying how many samples are taken for training. Note: (1 - n_samples) are taken for test
    """
    ind = np.arange(len(X))
    np.random.shuffle(ind)
    return X[ind[:n_samples]], X[ind[n_samples:]], ind
