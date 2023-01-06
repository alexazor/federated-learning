from numpy.typing import NDArray
from helper import ModelAdapter
from models import MLP
from utils import encrypt_vector, encrypt_matrix, sum_encrypted_vectors, sum_encrypted_matrix
import numpy as np
import torch.nn as nn
import torch


class Client:
    """Run linear regression either with local data or by gradient steps,
    where gradients can be sent from remotely.
    Hold the public key and can encrypt gradients to send remotely.
    """

    def __init__(self, name, X, y, pubkey, n_iter=100, lr=0.01):
        self.name = name
        self.pubkey = pubkey
        self.X, self.y = X, y
        self.weights = np.zeros(X.shape[1])
        self.n_iter = n_iter
        self.lr = lr

    def fit(self):
        """Linear regression for n_iter"""

        for _ in range(self.n_iter):
            gradient = self.compute_gradient()
            self.gradient_step(gradient)

    def gradient_step(self, gradient):
        """Update the model with the given gradient"""

        self.weights -= self.lr * gradient

    def compute_gradient(self):
        """Return the gradient computed at the current model on all training
        set"""

        delta = self.predict(self.X) - self.y
        return delta.dot(self.X)

    def predict(self, X):
        """Score test data"""
        X = np.array(X)
        return X.dot(self.weights)

    def encrypted_gradient(self, sum_to=None):
        """Compute gradient. Encrypt it.
        When `sum_to` is given, sum the encrypted gradient to it, assumed
        to be another vector of the same size
        """

        gradient = encrypt_vector(self.pubkey, self.compute_gradient())

        if sum_to is not None:
            if len(sum_to) != len(gradient):
                raise Exception('Encrypted vectors must have the same size')
            return sum_encrypted_vectors(sum_to, gradient)
        else:
            return gradient


class MLPClient:
    """Run linear regression either with local data or by gradient steps,
    where gradients can be sent from remotely.
    Hold the public key and can encrypt gradients to send remotely.
    """

    def __init__(self, name, X: NDArray, y: NDArray, pubkey, model: ModelAdapter, n_epoch=100):
        self.model = model
        self.name = name
        self.pubkey = pubkey
        self.X, self.y = X, y
        self.n_epoch = n_epoch

    def fit(self):
        """Linear regression for n_iter"""
        for _ in range(self.n_epoch):
            gradients = self.compute_gradient()
            self.model.update_model(gradients)
            print(self.model.compute_loss(self.X, self.y))
            # gradients = self.compute_gradient()
            # self.model.update_model(gradients)
        pass

    def gradient_step(self, gradients):
        self.model.update_model(gradients)
        pass

    def compute_gradient(self):
        self.model.forward_backward(self.X, self.y)
        return self.model.get_gradient()

    def predict(self, X: NDArray) -> float:
        """Score test data"""
        return self.model.predict(X)

    def encrypted_gradient(self, sum_to=None) -> list[NDArray]:
        """Compute gradient. Encrypt it.
        When `sum_to` is given, sum the encrypted gradient to it, assumed
        to be another vector of the same size
        """
        gradients = [encrypt_matrix(self.pubkey, gradient)
                     for gradient in self.compute_gradient()]

        if sum_to is not None:
            if len(sum_to) != len(gradients):
                raise Exception('Encrypted vectors must have the same size')
            return [sum_encrypted_matrix(grad_sum_to, grad) for grad_sum_to, grad in zip(sum_to, gradients)]
        else:
            return gradients


def get_mlp_client(name, X: NDArray, y: NDArray, pubkey, lr=0.01, **kwargs):
    model = MLP(X.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return MLPClient(name, X, y, pubkey, ModelAdapter(model, criterion, optimizer), **kwargs)


def get_lreg_client(name, X, y, pubkey, **kwargs):
    return Client(name, X, y, pubkey, **kwargs)
