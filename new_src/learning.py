import math
import time

import numpy as np

from client import get_client
from cryptosystem import CKKS, Paillier
from server import get_server
from utils import get_data, mean_square_error


class FederatedLearning:
    def __init__(self, n_clients, cl_type, crypto, nb_iter=500, seed=None) -> None:
        """FederatedLearning Class : Framework class to simulate federated learning 
        and evaluate performances.

        Args:
            n_clients (int): Number of clients
            cl_type (string): Model type : "MLP" or "lreg"
            crypto (string): Cryptosystem type : "ckks" or "paillier"
            nb_iter (int, optional): Number of iterations for the training. Defaults to 500.
            seed (int, optional): Seed for the initialization of the model weights. Defaults to None.
        """
        # cl_type : "MLP" or "lreg"
        # crypto : "paillier" or "ckks"
        self.data, self.label, self.testdata, self.testlabel = get_data(n_clients)
        self.nb_iter = nb_iter
        self.n_clients = n_clients
        if seed==None:
            self.seed = np.random.randint(0,2**31)
        else:
            self.seed = seed
        self.__generate_cryptosystem(crypto)
        self.__generate_clients(self.data, self.label, self.system.get_encryption(), cl_type, self.nb_iter)
        self.__generate_server(self.system.get_decryption())

    def __generate_cryptosystem(self, crypto):
        """Generates cryptosystem for federated learning and set the class attribute

        Args:
            crypto (Cryptosystem): Cryptosystem for encrypting and decrypting

        Raises:
            Exception: if crypto system provided doesn't exist
        """
        if crypto == "paillier":
            self.system = Paillier()
        elif crypto == "ckks":
            self.system = CKKS()
        else:
            raise Exception("Crypto system not available")

    def __generate_clients(self, X, y, encryptClass, cl_type, nb_iter):
        """Generates clients for federated learning

        Args:
            X (NDarray): data
            y (NDarray): label
            encryptClass (EncClass): Encrypting class
            cl_type (string): Model type : "MLP" or "lreg"
            nb_iter (int): number of iterations for the models
            seed (int, optional): seed for the initials model weights. 
                Defaults to None which will generate a random int as seed.
        """
        n_clients = len(X)
        self.clients = []
        for i in range(n_clients):
            data = X[i]
            target = y[i]
            self.clients.append(get_client(i+1, cl_type, data, target, encryptClass, nb_iter, self.seed))

    def __generate_server(self, decryptClass):
        """Generate server for federated learning

        Args:
            decryptClass (DecClass): Decrypting class
        """
        self.server = get_server(decryptClass)

    def reset_weights(self):
        """Reset weights of the models
        """
        for cl in self.clients:
            cl.reset_weights(self.seed)

    def local_learning(self):
        """Local learning of every client: each learns on its own data
        """
        for cl in self.clients:
            cl.local_fit()

    def federated_learning(self):
        """Federated learning process
        """
        # Federated Learning with encryption
        for i in range(self.nb_iter):
            if i % 50 == 0:
                print(f"Iteration {i}")
            # Compute gradients, encrypt and aggregate
            encrypt_aggr = None
            for cl in self.clients:
                encrypt_aggr = cl.encrypted_gradient(sum_to=encrypt_aggr)
            # Send aggregate to server and decrypt it
            aggr = self.server.give_clear_gradient(encrypt_aggr, self.n_clients)
            # Take gradient steps
            for cl in self.clients:
                cl.gradient_step(aggr)

    def mse(self, verbose = True):
        """MSE of the current client's model on the test data

        Args:
            verbose (bool, optional): Print MSE values. Defaults to True.

        Returns:
            dict: dict containing the MSE values for each client
        """
        mses = {}
        mean = 0
        for cl in self.clients:
            y_pred = cl.predict(self.testdata)
            cmse = mean_square_error(y_pred, self.testlabel)
            mses[cl.get_id()] = cmse
            mean += cmse
            if verbose:
                print(f"MSE {cl.get_id()}: {mses[cl.get_id()]}")
        mean /= self.n_clients
        if verbose:
            print(f"Mean MSE : {mean}")
        return mses

    def loss(self, verbose=True):
        """Loss of the current client's model on the test data

        Args:
            verbose (bool, optional): Print Loss values. Defaults to True.

        Returns:
            dict: dict containing the Loss values for each client
        """
        losses = {}
        for cl in self.clients:
            losses[cl.get_id()] = cl.compute_loss(self.testdata, self.testlabel)
            if verbose:
                print(f"Loss {cl.get_id()}: {losses[cl.get_id()]}")
        return losses
        
    def local_learning_mse(self, verbose = True):
        """Run local_learning() to train clients and then run mse()
        to evaluate performances.

        Args:
            verbose (bool, optional): Print MSE values. Defaults to True.

        Returns:
            dict: dict containing the MSE values for each client
        """
        start = time.time()
        self.local_learning()
        mses = self.mse(verbose)
        end = time.time()
        print(f"Execution Time : {end-start}")
        return mses
    
    def federated_learning_mse(self, verbose = True):
        """Run federated_learning() to train clients and then run mse()
        to evaluate performances.

        Args:
            verbose (bool, optional): Print MSE values. Defaults to True.

        Returns:
            dict: dict containing the MSE values for each client
        """
        start = time.time()
        self.federated_learning()
        mses = self.mse(verbose)
        end = time.time()
        print(f"Execution Time : {end-start}")
        return mses