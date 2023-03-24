

class Server:
    """Holds the decrypting class. 
    Allows to decrypt and average the gradients"""

    def __init__(self, decryptClass):
        self.decrypt = decryptClass # Decrypt Class : PaillerDec or CKKSDec

    def give_clear_gradient(self, encrypted_gradient, n_clients):
        """Returns clear gradients and averages them

        Args:
            encrypted_gradient (list[encrypted_gradient]): encrypted gradients of the model
            n_clients (int): number of clients

        Returns:
            list[NDarray]: gradients of the model
        """
        gradients = self.decrypt.decrypt_gradients(encrypted_gradient)
        n_grad = len(gradients)
        for i in range(n_grad):
            gradients[i] = gradients[i]/n_clients
        return gradients

def get_server(decryptClass):
    """Returns a server object

    Args:
        decryptClass (DecClass): Decrypting class

    Returns:
        Server: a server object with the decryptClass as decrypting system
    """
    server = Server(decryptClass)
    return server