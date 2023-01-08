

class Server:
    """Holds the private key. Decrypt the average gradient"""

    def __init__(self, decryptClass):
        self.decrypt = decryptClass # Decrypt Class : PaillerDec or CKKS

    def give_clear_gradient(self, encrypted_gradient):
        return self.decrypt.decrypt_tensor(encrypted_gradient)

def get_server(decryptClass):
    server = Server(decryptClass)
    return server