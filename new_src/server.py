

class Server:
    """Holds the private key. Decrypt the average gradient"""

    def __init__(self, decryptClass):
        self.decrypt = decryptClass # Decrypt Class : PaillerDec or CKKSDec

    def give_clear_gradient(self, encrypted_gradient, n_clients):
        gradients = self.decrypt.decrypt_gradients(encrypted_gradient)
        n_grad = len(gradients)
        for i in range(n_grad):
            gradients[i] = gradients[i]/n_clients
        return gradients

def get_server(decryptClass):
    server = Server(decryptClass)
    return server