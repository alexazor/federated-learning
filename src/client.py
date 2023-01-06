from numpy._typing import NDArray
from utils import encrypt_matrix, sum_encrypted_matrix


class Client:
    def __init__(self, id, pubkey, model, data, target: NDArray, encrypt, nb_iter=100):
        self.id = id
        self.pubkey = pubkey
        self.model = model
        self.data = data
        self.target = target
        self.encrypt = encrypt  # encrypt function
        self.nb_iter = nb_iter

    def local_fit(self):
        for _ in range(self.nb_iter):
            gradients = self.compute_gradient()
            self.model.update_model(gradients)
            print(self.model.compute_loss(self.data, self.target))
        pass

    def gradient_step(self, gradients):
        self.model.update_model(gradients)
        pass

    def compute_gradient(self):
        return self.model.compute_gradient(self.data, self.target)

    def predict(self, X: NDArray) -> float:
        """Score test data"""
        return self.model.predict(X)

    def encrypted_gradient(self, sum_to=None) -> list[NDArray]:
        """Compute gradient. Encrypt it.
        When `sum_to` is given, sum the encrypted gradient to it, assumed
        to be another vector of the same size
        """
        gradients = [encrypt_matrix(self.pubkey, gradient, self.encrypt)
                     for gradient in self.compute_gradient()]

        if sum_to is not None:
            if len(sum_to) != len(gradients):
                raise Exception('Encrypted vectors must have the same size')
            return [sum_encrypted_matrix(grad_sum_to, grad) for grad_sum_to, grad in zip(sum_to, gradients)]
        else:
            return gradients
