import numpy as np


def load_dataset(filepath):
    """
    Loads dataset into np.array.

    Parameters
    ----------
    filepath: str
        filepath where dataset is located

    Returns
    -------

    np.array:
        2 dimensional np.array (n x k) with k columns[:-1] as features, n rows as samples, column[-1] as label
    """
    # Load data
    data = np.loadtxt(filepath)

    # Check for nans
    if np.isnan(data).sum() == 0:
        return data
    else:
        print('Dataset contains nans. Please pre-process')


def split_dataset_x_y(data):
    """
    Split features x from labels y of given dataset with dimensions (n x k)

    Parameters
    ----------
    data: np.array
        2 dimensional np.array with columns[:-1] as features, rows as samples, column[-1] as label

    Returns
    -------
    x: np.array
        2 dimensional np.array (n x k-1) with columns as features, rows as samples
    y: (n,) dimensional np.array holding the label for each sample n
    """
    x = data[:, :-1]
    y = data[:, -1]
    return x, y

