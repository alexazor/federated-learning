import phe as paillier
from utils import decrypt_vector

class Server:
    """Hold the private key. Decrypt the average gradient"""

    def __init__(self, key_length=1024):
        self.pubkey, self.privkey = paillier.generate_paillier_keypair(n_length=key_length)

    def decrypt_aggregate(self, input_model, n_clients):
        return decrypt_vector(self.privkey, input_model) / n_clients