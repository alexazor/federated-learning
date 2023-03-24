from abc import ABC, abstractmethod

import numpy as np
import phe as paillier
import tenseal as ts
from numpy._typing import NDArray

############
# Paillier #
############

class Cryptosystem(ABC):
    @abstractmethod
    def get_pubkey(self):
        """Returns public key
        """
        pass

    @abstractmethod
    def get_privkey(self):
        """Returns private key
        """
        pass

    @abstractmethod
    def get_encryption(self):
        """Returns encryption class
        """
        pass

    @abstractmethod
    def get_decryption(self):
        """Returns decryption class
        """
        pass

class EncClass(ABC):
    @abstractmethod
    def encrypt_gradients(self, gradients):
        """Returns list of encrypted gradients
        Args:
            gradients (list[NDArray]): model's gradients
        """
        pass

    @abstractmethod
    def sum_encrypted_gradients(self, x, y):
        """Sums list of encrypted gradients
        Args:
            x (list[NDArray]): model's gradients
            y (list[NDArray]): model's gradients
        """
        pass

class DecClass(ABC):
    @abstractmethod
    def decrypt_gradients(self, enc_gradients):
        """Returns list of decrypted gradients
        Args:
            enc_gradients (list[Encrypted_NDArray]): encrypted model's gradients
        """
        pass

class Paillier(Cryptosystem):
    def __init__(self) -> None:
        """Paillier cryptosystem with encrypting and decrypting class
        """
        pubkey, privkey = paillier.generate_paillier_keypair(n_length=1024)
        self.pubkey = pubkey
        self.privkey = privkey

    def get_pubkey(self):
        return self.pubkey
    
    def get_privkey(self):
        return self.privkey

    def get_encryption(self):
        enc = PaillierEnc(self.pubkey)
        return enc

    def get_decryption(self):
        dec = PaillierDec(self.privkey)
        return dec

class PaillierEnc(EncClass):
    def __init__(self, pubKey) -> None:
        """Paillier encrypting class

        Args:
            pubKey (Paillier.Pubkey): Paillier public key
        """
        self.pubKey = pubKey

    def encrypt_value(self, message):
        """Encrypt value : int of float

        Args:
            message (int, float): Value to encrypt

        Returns:
            Encrypted_value: Encrypted value
        """
        return self.pubKey.encrypt(message)

    def encrypt_vector(self, vector):
        """Encrypt vector

        Args:
            vector (list, NDArray): Vector of values to encrypt

        Returns:
            list[Encrypted_value]: Vector of encrypted values
        """
        return [self.encrypt_value(v) for v in vector]

    def encrypt_matrix(self, matrix: NDArray):
        """Encrypt matrix

        Args:
            matrix (NDArray): Matrix of values to encrypt

        Returns:
            list[list[Encrypted_value]]: Matrix of encrypted values
        """
        if isinstance(matrix[0], (int, float)):
            return self.encrypt_vector(matrix)
        return [self.encrypt_vector(row) for row in matrix]

    def encrypt_gradients(self, gradients):
        """Returns list of encrypted gradients
        Args:
            gradients (list[NDArray]): model's gradients

        Returns:
            list[Encrypted_gradient]: Encrypted gradients
        """
        enc_grads = []
        for grad in gradients:
            enc_grads.append(self.encrypt_matrix(grad))
        return enc_grads

    def sum_encrypted_vectors(self, x, y):
        """Sum of two encrypted vectors

        Args:
            x (list[Encrypted_value]): Vector of encrypted values
            y (list[Encrypted_value]): Vector of encrypted values

        Raises:
            Exception: If vectors are not of the same size

        Returns:
            list[Encrypted_value]: Sum vector
        """
        if len(x) != len(y):
            raise Exception('Encrypted vectors must have the same size')
        return [x[i] + y[i] for i in range(len(x))]

    def sum_encrypted_matrix(self, x, y):
        """Sum of two encrypted matrix

        Args:
            x (list[list[Encrypted_value]]): Matrix of encrypted values
            y (list[list[Encrypted_value]]): Matrix of encrypted values

        Returns:
            list[list[Encrypted_value]]: Sum matrix
        """
        if not isinstance(x[0], list):
            return self.sum_encrypted_vectors(x, y)
        return [[x[i][j] + y[i][j] for j in range(len(x[0]))] for i in range(len(x))]

    def sum_encrypted_gradients(self, x, y):
        """Sums list of encrypted gradients
        Args:
            x (list[NDArray]): model's gradients
            y (list[NDArray]): model's gradients

        Raises:
            Exception: If X and Y are not of the same size

        Returns:
            list[Encrypted_gradient]: Sum of gradients
        """
        if len(x) != len(y):
            raise Exception('Encrypted vectors must have the same size')
        return [self.sum_encrypted_matrix(x[i], y[i]) for i in range(len(x))]

class PaillierDec(DecClass):
    def __init__(self, privKey) -> None:
        """Paillier decrypting class

        Args:
            privKey (Paillier.Privkey): Paillier private key
        """
        self.privKey = privKey
        
    def decrypt_value(self, enc_message):
        """Decrypt value with paillier

        Args:
            enc_message (Encrypted_value): Encrypted value with paillier

        Returns:
            (int, float): Decrypted value
        """
        return self.privKey.decrypt(enc_message)

    def decrypt_vector(self, enc_vector):
        """Decrypt vector of encrypted values

        Args:
            enc_vector (list[Encrypted_value]): Vector of encrypted values

        Returns:
            NDArray[Encrypted_value]: Vector of decrypted values
        """
        return np.array([self.decrypt_value(ev) for ev in enc_vector])

    def decrypt_matrix(self, enc_matrix):
        """Decrypt matrix of encrypted values

        Args:
            enc_matrix (list[list[Encrypted_value]]): Matrix of encrypted values

        Returns:
            NDArray[NDArray[Encrypted_value]]: Matrix of decrypted values
        """
        if not isinstance(enc_matrix[0], list):
            return self.decrypt_vector(enc_matrix)
        return np.array([self.decrypt_vector(row) for row in enc_matrix])

    def decrypt_gradients(self, enc_gradients):
        """Returns list of decrypted gradients
        Args:
            enc_gradients (list[NDArray[Encrypted_value]): encrypted model's gradients

        Returns:
            list[gradient]: Model's gradients
        """
        gradients = []
        for grad in enc_gradients:
            gradients.append(self.decrypt_matrix(grad))
        return gradients

########
# CKKS #
########

class CKKS(Cryptosystem):
    def __init__(self) -> None:
        """CKKS cryptosystem with encrypting and decrypting class
        """
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
          )
        self.context.generate_galois_keys()
        self.context.global_scale = 2**40
        self.privkey = self.context.secret_key()
        self.context.make_context_public()

    def get_pubkey(self):
        return self.context.public_key()

    def get_privkey(self):
        return self.privkey

    def get_context(self):
        return self.context

    def get_encryption(self):
        enc = CKKSEnc(self.context)
        return enc

    def get_decryption(self):
        dec = CKKSDec(self.privkey)
        return dec

class CKKSEnc(EncClass):
    def __init__(self, context) -> None:
        """CKKS encrypting class

        Args:
            context (tenseal.Context): Tenseal context for encrypting
        """
        self.context = context

    def encrypt_vector(self, vector):
        """Encrypt vector

        Args:
            vector (list, NDArray): Vector of values to encrypt

        Returns:
            tenseal.Encrypted_vector: Vector of encrypted values
        """
        return ts.ckks_vector(self.context, vector)

    def encrypt_tensor(self, matrix):
        """Encrypt tensor or matrix

        Args:
            matrix (NDArray, tensor): Matrix or tensor of values to encrypt

        Returns:
            tenseal.Encrypted_tensor: Tensor of encrypted values
        """
        if isinstance(matrix[0], (int, float)):
            return self.encrypt_vector(matrix)
        return ts.ckks_tensor(self.context, matrix)

    def encrypt_gradients(self, gradients):
        """Encrypt model's gradients

        Args:
            gradients (list[NDArray]): Model's gradients

        Returns:
            list[tenseal.Encrypted_tensor]: Encrypted gradients
        """
        enc_grads = []
        for grad in gradients:
            enc_grads.append(ts.ckks_tensor(self.context, grad))
        return enc_grads

    def sum(self, x, y):
        """Sum of vectors or matrix or tensor

        Args:
            x (tenseal.Encrypted_vector, tenseal.Encrypted_tensor): Iterable to sum
            y (tenseal.Encrypted_vector, tenseal.Encrypted_tensor): Iterable to sum

        Returns:
            (tenseal.Encrypted_vector, tenseal.Encrypted_tensor): Sum
        """
        return x + y

    def sum_encrypted_gradients(self, x, y):
        """_summary_

        Args:
            x (list[tenseal.Encrypted_tensor]): Model's gradients
            y (list[tenseal.Encrypted_tensor]): Model's gradients

        Returns:
            tenseal.Encrypted_tensor: Sum of model's gradients
        """
        return [x[i] + y[i] for i in range(len(x))]

class CKKSDec(DecClass):
    def __init__(self, privkey) -> None:
        """Decrypting class for CKKS

        Args:
            privkey (tenseal.Secret_key): tenseal private key for decrypting
        """
        self.privkey = privkey

    def decrypt(self, enc):
        """Decrypt vector or matrix or tensor

        Args:
            enc (tenseal.Encrypted_vector, tenseal.Encrypted_tensor): Encrypted iterable

        Returns:
            (list, NDArray, tensor): Decrypted iterable
        """
        return enc.decrypt(self.privkey)

    def decrypt_gradients(self, enc_gradients):
        """Decrypt model's gradients

        Args:
            enc_gradients (list[tenseal.Encrypted_tensor]): Encrypted model's gradients

        Returns:
            list[NDArray]: Model's gradients
        """
        gradients = []
        for enc_grad in enc_gradients:
            gradients.append(np.array(self.decrypt(enc_grad).tolist()))
        return gradients