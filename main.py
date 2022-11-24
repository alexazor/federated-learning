from utils import get_data
from client import Client
from server import Server

# Initialize data
names = ['Hospital 1', 'Hospital 2', 'Hospital 3']
n_clients = len(names)
X, y, X_test, y_test = get_data(n_clients=n_clients)
print(len(y))

# Initialize server
server = Server(key_length=1024)
pubkey = server.pubkey

# Initialize clients
clients = []
for name, data, target in zip(names,X,y):
    clients.append(Client(name, data, target, pubkey))
