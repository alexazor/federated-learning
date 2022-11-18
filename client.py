from utils import encrypt_vector, sum_encrypted_vectors
import numpy as np

class Client:
    """Run linear regression either with local data or by gradient steps,
    where gradients can be send from remotely.
    Hold the public key and can encrypt gradients to send remotely.
    """

    def __init__(self, name, X, y, pubkey, n_iter = 100, eta = 0.01):
        self.name = name
        self.pubkey = pubkey
        self.X, self.y = X, y
        self.weights = np.zeros(X.shape[1])
        self.n_iter = n_iter
        self.eta = eta

    def fit(self):
        """Linear regression for n_iter"""

        for _ in range(self.n_iter):
            gradient = self.compute_gradient()
            self.gradient_step(gradient)

    def gradient_step(self, gradient):
        """Update the model with the given gradient"""
        
        self.weights -= self.eta * gradient

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