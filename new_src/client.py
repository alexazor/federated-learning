from numpy._typing import NDArray

from cryptosystem import CKKS, PaillierEnc
from model import get_model_lin_reg, get_model_mlp


class Client:
    def __init__(self, id, model, encryptClass, nb_iter):
        self.id = id
        self.model = model
        self.data = model.get_data()
        self.target = model.get_label()
        self.nb_iter = nb_iter
        self.encrypt = encryptClass  # encrypt class : PaillierEnc or CKKS
        self.key = self.encrypt.get_pubkey()

    def get_id(self):
        return self.id

    def get_data(self):
        return self.data

    def get_label(self):
        return self.label

    def set_data(self, data: NDArray):
        self.model.set_data(data)

    def set_label(self, label: NDArray):
        self.model.set_label(label)

    def reset_weights(self):
        self.model.reset_weights()

    def local_fit(self):
        for _ in range(self.nb_iter):
            gradients = self.compute_gradient()
            self.gradient_step(gradients)
        pass

    def gradient_step(self, gradients):
        self.model.update_model(gradients)
        pass

    def compute_gradient(self):
        return self.model.compute_gradient()

    def predict(self, data: NDArray) -> float:
        """Score test data"""
        return self.model.predict(data)

    def compute_loss(self, data, label):
        return self.model.compute_loss(data, label)

    def encrypted_gradient(self, sum_to=None) -> list[NDArray]:
        """Compute gradient. Encrypt it.
        When `sum_to` is given, sum the encrypted gradient to it, assumed
        to be another vector of the same size
        """
        
        gradients = self.encrypt.encrypt_tensor(self.compute_gradient())

        if sum_to is not None:
            try:
                diff_size = sum_to.size() - gradients.size()
            except:
                diff_size = len(sum_to) - len(gradients)
            if diff_size != 0:
                raise Exception('Encrypted vectors must have the same size')
            return self.encrypt.sum_encrypted_tensor(sum_to, gradients)
        else:
            return gradients

def get_client(id, cl_type, data, target, encryptClass, nb_iter=100, lr=0.01, **kwargs):
    if cl_type == "MLP":
        model = get_model_mlp(data.shape[1], data, target, lr=lr)
    elif cl_type == "lreg":
        model = get_model_lin_reg(data.shape[1], data, target)
    return Client(id, model, encryptClass, nb_iter, **kwargs)

