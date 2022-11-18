import phe as paillier
import numpy as np
from sklearn.datasets import load_diabetes
from datetime import datetime
import math

def get_data(n_clients):

    diabetes = load_diabetes()
    y = diabetes.target
    X = diabetes.data

    # Add constant to emulate intercept
    X = np.c_[X, np.ones(X.shape[0])]

    # The features are already preprocessed
    # Shuffle
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
    # sees a potentially very different sample of patients.
    X, y = [], []
    step = int(X_train.shape[0] / n_clients)
    for c in range(n_clients):
        X.append(X_train[step * c: step * (c + 1), :])
        y.append(y_train[step * c: step * (c + 1)])

    return X, y, X_test, y_test

def encrypt_vector(pubkey, x):
    return [pubkey.encrypt(x[i]) for i in range(x.shape[0])]

def encrypt_vector_2(pubkey, x):
    return [pubkey.encrypt(k) for k in x]

def decrypt_vector(privkey, x):
    return np.array([privkey.decrypt(i) for i in x])

def sum_encrypted_vectors(x, y):
    if len(x) != len(y):
        raise Exception('Encrypted vectors must have the same size')
    return [x[i] + y[i] for i in range(len(x))]

def mean_square_error(y_pred, y):
    return np.mean((y - y_pred) ** 2)