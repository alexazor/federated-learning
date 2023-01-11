from numpy._typing import NDArray

from cryptosystem import CKKS, PaillierEnc
from model import get_model_lin_reg, get_model_mlp


class Client:
    def __init__(self, id, model, encryptClass, nb_iter):
        """Client Class: Object representing a client.
        It has its own model and encrypting class to encrypt and send gradients.

        Args:
            id (int): ID of the client
            model (Model): Model of the client
            encryptClass (EncClass): Encrypting class of the client
            nb_iter (int): Number of iterations to train the model
        """
        self.id = id
        self.model = model
        self.data = model.get_data()
        self.target = model.get_label()
        self.nb_iter = nb_iter
        self.encrypt = encryptClass  # encrypt class : PaillierEnc or CKKSEnc

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

    def reset_weights(self, seed):
        """Resets the model weights
        """
        self.model.reset_weights(seed)

    def local_fit(self):
        """Local training of the client's model
        """
        for i in range(self.nb_iter):
            gradients = self.compute_gradient()
            self.gradient_step(gradients)
            if i % 250 == 0:
                print(f"Iteration : {i}")
        pass

    def gradient_step(self, gradients):
        """Applies gradient step to the model

        Args:
            gradients (list[NDarray]): model's gradients
        """
        self.model.update_model(gradients)
        pass

    def compute_gradient(self):
        """Computes model's gradients on training data

        Returns:
            list[NDarray]: Gradients of the model
        """
        return self.model.compute_gradient()

    def predict(self, data: NDArray):
        """Predicts label on data

        Args:
            data (NDArray): Data

        Returns:
            NDarray: Predicted label
        """
        return self.model.predict(data)

    def compute_loss(self, data, label):
        """Computes loss on data vs label

        Args:
            data (NDarray): data
            label (NDarray): labels

        Returns:
            float: loss value
        """
        return self.model.compute_loss(data, label)

    def encrypted_gradient(self, sum_to=None) -> list[NDArray]:
        """Compute gradient and encrypt it.
        When `sum_to` is given, sum the encrypted gradient to it, assumed
        to be another vector of the same size

        Args:
            sum_to (list[NDarray], optional): Sum of gradients of previous models. Defaults to None.

        Returns:
            list[NDArray]: Encrypted sum of gradients
        """
        gradients = self.encrypt.encrypt_gradients(self.compute_gradient())

        if sum_to is not None:
            """ Inutile
            try:
                diff_size = sum_to.size() - gradients.size()
            except:
                diff_size = len(sum_to) - len(gradients)
            if diff_size != 0:
                raise Exception('Encrypted vectors must have the same size')
            """
            return self.encrypt.sum_encrypted_gradients(sum_to, gradients)
        else:
            return gradients

def get_client(id, cl_type, data, target, encryptClass, nb_iter=100, seed=7, lr=0.01, **kwargs):
    """Returns a client object initialized with the given parameters

    Args:
        id (int): ID of client
        cl_type (string): Model type : "MLP" or "lreg"
        data (NDarray): Train data
        target (NDarray): Train labels
        encryptClass (EncClass): Encrypting class
        nb_iter (int, optional): Number of iterations for training. Defaults to 100.
        seed (int, optional): Seed to initialize weights of the models. Defaults to 7.
        lr (float, optional): Learning rate for the model. Defaults to 0.01.

    Returns:
        Client: Client object initialized
    """
    if cl_type == "MLP":
        model = get_model_mlp(data.shape[1], data, target, seed, lr=lr)
    elif cl_type == "lreg":
        model = get_model_lin_reg(data.shape[1], data, target, lr)
    return Client(id, model, encryptClass, nb_iter, **kwargs)

