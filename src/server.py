from utils import decrypt_matrix


class Server:
    """Holds the private key. Decrypt the average gradient"""

    def __init__(self, preivKey, pubKey, decrypt):
        self.pubKey = pubKey
        self.privKey = privKey
        self.decrypt = decrypt

    def give_clear_gradient(self, encrypted_gradient):
        """_summary_

        Args:
            encrypted_gradient (list of encrypted vectors (list) or encrypted matrixes (list of list)):

        Returns:
            list: _description_
        """
        return [decrypt_matrix(self.privKey, self.pubKey, element) for element in encrypted_gradient]
