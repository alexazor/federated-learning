from src.learning import federated_learning
from utils import get_data, mean_square_error
from clients import get_mlp_client



if __name__ == '__main__':
    names = ['Hospital 1', 'Hospital 2', 'Hospital 3']
    n_clients = len(names)
    X, y, X_test, y_test = get_data(n_clients)
    # clients = local_learning(names, n_clients, X, y, get_mlp_client, n_iter = 1000)

    clients = federated_learning(names, n_clients, X, y, get_mlp_client, n_iter=100)
    for client, xx, yy in zip(clients, X_test, y_test):
        print("")
        pred = client.predict(xx)
        print(mean_square_error(pred, yy))

# Initialize data
# names = ['Hospital 1', 'Hospital 2', 'Hospital 3']
# n_clients = len(names)
# X, y, X_test, y_test = get_data(n_clients=n_clients)
# print(len(y))

# Initialize server
# server = Server(key_length=1024)
# pubkey = server.pubkey

# Initialize clients
# clients = []
# for name, data, target in zip(names,X,y):
#    clients.append(Client(name, data, target, pubkey))
