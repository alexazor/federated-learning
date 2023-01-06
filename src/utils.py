import numpy as np
from numpy._typing import NDArray
from sklearn.datasets import load_diabetes


def get_data(n_clients):

    diabetes = load_diabetes()
    y = diabetes.target
    X = diabetes.data

    # Add constant to emulate intercept
    X = np.c_[X, np.ones(X.shape[0])]

    # The features are already preprocessed
    # Shuffle
    # Is this really useful ? As the representative test set is already sampled randomly (see below)
    # and that we want to induce a bias on the splited distributions ?
    perm = np.random.permutation(X.shape[0])
    X, y = X[perm, :], y[perm]

    # Select test at random
    test_size = 50
    test_idx = np.random.choice(X.shape[0], size=test_size, replace=False)
    train_idx = np.ones(X.shape[0], dtype=bool)
    train_idx[test_idx] = False
    X_test, y_test = X[test_idx, :], y[test_idx]
    X_train, y_train = X[train_idx, :], y[train_idx]

    # Split train among multiple clients.
    # The selection is not at random but by slice. We simulate the fact that each client
    # sees a potentially very different sample of patients. ??? Not really because the original dataset has been suffled.
    X, y = [], []
    step = int(X_train.shape[0] / n_clients)
    for c in range(n_clients):
        X.append(X_train[step * c: step * (c + 1), :])
        y.append(y_train[step * c: step * (c + 1)])

    return X, y, X_test, y_test


def encrypt_vector(pubKey, vector, encrypting_function):
    return [encrypting_function(pubKey, vector[i]) for i in range(vector.shape[0])]


def encrypt_matrix(pubKey, matrix: NDArray, encrypting_function):
    if matrix.ndim == 1:
        return encrypt_vector(pubKey, matrix, encrypting_function)
    return [encrypt_vector(pubKey, row, encrypting_function) for row in matrix]


def decrypt_vector(privKey, pubKey, vector, decrypting_function):
    return np.array([decrypting_function(privKey, pubKey, vector[i]) for i in vector])


def decrypt_matrix(privKey, pubKey, matrix, decrypting_function):
    if not isinstance(matrix[0], list):
        return decrypt_vector(privKey, matrix)
    return np.array([decrypt_vector(privKey, pubKey, row, decrypting_function) for row in x])


def sum_encrypted_vectors(x, y):
    if len(x) != len(y):
        raise Exception('Encrypted vectors must have the same size')
    return [x[i] + y[i] for i in range(len(x))]


def sum_encrypted_matrix(x, y):
    if not isinstance(x[0], list):
        return sum_encrypted_vectors(x, y)
    return [[x[i][j] + y[i][j] for j in range(len(x[0]))] for i in range(len(x))]


def mean_square_error(y_pred, y):
    return np.mean((y - y_pred) ** 2)
