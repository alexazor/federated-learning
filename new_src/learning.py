import math
import time

import numpy as np

from client import get_client
from cryptosystem import CKKS, Paillier
from server import get_server
from utils import get_data, mean_square_error


class FederatedLearning:
    def __init__(self, n_clients, cl_type, crypto, nb_iter=100) -> None:
        self.data, self.label, self.testdata, self.testlabel = get_data(n_clients)
        self.nb_iter = nb_iter
        self.system = self.__generate_cryptosystem(crypto)
        self.clients = self.__generate_clients(self.data, self.label, self.system.get_encryption(), cl_type, self.nb_iter)
        self.server = self.__generate_server(self.system.get_decryption())

    def __generate_cryptosystem(crypto):
        if crypto == "paillier":
            system = Paillier()
        elif crypto == "ckks":
            system = CKKS()
        else:
            raise Exception("Crypto system not available")
        
        return system

    def __generate_clients(X, y, encryptClass, cl_type, nb_iter=100):
        # cl_type : MLP or lreg
        n_clients = len(X)
        clients = []
        for i in range(n_clients):
            data = X[i]
            target = y[i]
            client = get_client(i+1, cl_type, data, target, encryptClass, nb_iter)
            clients[i] = client
        
        return clients

    def __generate_server(decryptClass):
        return get_server(decryptClass)

    def reset_weights(self):
        for cl in self.clients:
            cl.reset_weights()

    def local_learning(self):
        for cl in self.clients:
            cl.local_fit()

    def federated_learning(self):
        # Federated Learning with encryption
        for i in range(self.nb_iter):
            if i % 50 == 0:
                print(f"Iteration {i}")
            # Compute gradients, encrypt and aggregate
            encrypt_aggr = None
            for cl in self.clients:
                encrypt_aggr = cl.encrypted_gradient(sum_to=encrypt_aggr)
            # Send aggregate to server and decrypt it
            aggr = self.server.give_clear_gradient(encrypt_aggr)
            # Take gradient steps
            for cl in self.clients:
                cl.gradient_step(aggr)

    def mse(self, verbose = True):
        mses = {}
        for cl in self.clients:
            y_pred = cl.predict(self.testdata)
            mses[cl.get_id()] = mean_square_error(y_pred, self.testlabel)
            if verbose:
                print(f"MSE {cl.get_id()}: {mses[cl.get_id()]}")
        return mses

    def loss(self, verbose=True):
        losses = {}
        for cl in self.clients:
            losses[cl.get_id()] = cl.compute_loss(self.testdata, self.testlabel)
            if verbose:
                print(f"Loss {cl.get_id()}: {losses[cl.get_id()]}")
        return losses
        
    def local_learning_mse(self, verbose = True):
        start = time.time()
        self.local_learning()
        mses = self.mse(verbose)
        end = time.time()
        print(f"Execution Time : {end-start}")
        return mses
    
    def federated_learning_mse(self, verbose = True):
        start = time.time()
        self.federated_learning()
        mses = self.mse(verbose)
        end = time.time()
        print("Execution Time : " + str(end-start))
        return mses