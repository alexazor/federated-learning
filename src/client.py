import numpy as np
from utils import encrypt_vector


class Client:
    def __init__(self, id, pubkey, model, data, target: NDArray, encrypt):

        self.id = id
        self.pubkey = pubkey
        self.model = model
        self.data = data
        self.target = target
        self.encrypt = encrypt  # encrypt function

    def local_fit(self):
        for _ in range(self.n_iter):
            gradient = self.compute_gradient()
            self.gradient_step(gradient)

    def gradient_step(self, gradient):
        self.weights -= self.lr * gradient

    def encrypt_gradient(self):

        encrypted_gradient = encrypt_vector(
            self.pubkey, self.compute_gradient(), self.encrypt)
        return encrypted_gradient
