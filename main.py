import numpy as np
from utils import get_data
from model import local_learning_mse, find_best_eta, federated_learning_mse

# Seed for reproductability
seed = 42
np.random.seed(seed)

# Initialize data
n_iter, eta = 100, 0.01
names = ['Hospital 1', 'Hospital 2', 'Hospital 3']
n_clients = len(names)
X, y, X_test, y_test = get_data(n_clients=n_clients)
print(type(X))
res = local_learning_mse(names,n_clients, X, y, X_test, y_test, n_iter, eta)
print(res)

dic = find_best_eta(names,n_clients, X, y, X_test, y_test, n_iter, [0.1, 0.05, 0.02, 0.01, 0.008, 0.006, 0.004, 0.002, 0.001, 0.0005, 0.0001])
print(dic)

federated_learning_mse(names,n_clients, X, y, X_test, y_test, n_iter, eta)

