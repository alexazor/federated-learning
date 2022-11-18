from client import Client
from server import Server
from utils import mean_square_error
from datetime import datetime
import math
import numpy as np

def local_learning(names,n_clients, X, y, n_iter = 100, eta = 0.01):
    clients = []
    for i in range(n_clients):
        clients.append(Client(names[i], X[i], y[i], None, n_iter, eta))
    
    for c in clients:
        c.fit()
    
    return clients

def federated_learning(names, n_clients, X, y, n_iter = 100, eta = 0.01):
    server = Server(key_length=1024)
    clients = []
    for i in range(n_clients):
        clients.append(Client(names[i], X[i], y[i], server.pubkey, None, eta))
        
    # Federated Learning with encryption
    for i in range(n_iter):
        if i % 50 == 0:
            print(i)
        # Compute gradients, encrypt and aggregate
        encrypt_aggr = None
        for i in range(n_clients):
            encrypt_aggr = clients[i].encrypted_gradient(sum_to=encrypt_aggr)
        # Send aggregate to server and decrypt it
        aggr = server.decrypt_aggregate(encrypt_aggr, n_clients)
        # Take gradient steps
        for c in clients:
            c.gradient_step(aggr)
            
    return clients

def mse(clients, X_test, y_test, verbose = True):
    mses = {}
    for c in clients:
        y_pred = c.predict(X_test)
        mses[c.name] = mean_square_error(y_pred, y_test)
        if verbose:
            print(f"{c.name}: {mses[c.name]}")
    return mses
        
def local_learning_mse(names,n_clients, X, y, X_test, y_test, n_iter = 100, eta = 0.01, verbose = True):
    start = datetime.now()
    clients = local_learning(names,n_clients, X, y, n_iter, eta)
    mses = mse(clients, X_test, y_test, verbose)
    end = datetime.now()
    if verbose:
        print("Execution Time : " + str(end-start))
    return mses
    
def federated_learning_mse(names,n_clients, X, y, X_test, y_test, n_iter = 100, eta = 0.01):
    start = datetime.now()
    clients = federated_learning(names,n_clients, X, y, n_iter, eta)
    mse(clients, X_test, y_test)
    end = datetime.now()
    print("Execution Time : " + str(end-start))

def find_best_eta(names,n_clients, X, y, X_test, y_test, n_iter, eta_list):
    dic = {}
    best_mse, best_eta = math.inf, -1
    for eta in eta_list:
        res = local_learning_mse(names,n_clients, X, y, X_test, y_test, n_iter, eta, False)
        m = np.array(list(res.values())).mean()
        dic[eta] = m
        if m < best_mse:
            best_mse = m
            best_eta = eta
    print("Best Eta : " + str(best_eta) + " with MSE : " + str(best_mse))
    return dic