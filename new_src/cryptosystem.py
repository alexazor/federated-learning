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
        pass

    @abstractmethod
    def get_encryption(self):
        pass

    @abstractmethod
    def get_decryption(self):
        pass

class Paillier(Cryptosystem):
    def __init__(self) -> None:
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

class PaillierEnc:
    def __init__(self, pubKey) -> None:
        self.pubKey = pubKey

    def encrypt_value(self, message):
        return self.pubKey.encrypt(message)

    def encrypt_vector(self, vector):
        return [self.encrypt_value(v) for v in vector]

    def encrypt_matrix(self, matrix: NDArray):
        if isinstance(matrix[0], (int, float)):
            return self.encrypt_vector(matrix)
        return [self.encrypt_vector(row) for row in matrix]

    def encrypt_gradients(self, gradients):
        enc_grads = []
        for grad in gradients:
            enc_grads.append(self.encrypt_matrix(grad))
        return enc_grads

    def sum_encrypted_vectors(self, x, y):
        if len(x) != len(y):
            raise Exception('Encrypted vectors must have the same size')
        return [x[i] + y[i] for i in range(len(x))]

    def sum_encrypted_matrix(self, x, y):
        if not isinstance(x[0], list):
            return self.sum_encrypted_vectors(x, y)
        return [[x[i][j] + y[i][j] for j in range(len(x[0]))] for i in range(len(x))]

    def sum_encrypted_gradients(self, x, y):
        if len(x) != len(y):
            raise Exception('Encrypted vectors must have the same size')
        return [self.sum_encrypted_matrix(x[i], y[i]) for i in range(len(x))]

class PaillierDec:
    def __init__(self, privKey) -> None:
        self.privKey = privKey
        
    def decrypt_value(self, enc_message):
        return self.privKey.decrypt(enc_message)

    def decrypt_vector(self, enc_vector):
        return np.array([self.decrypt_value(ev) for ev in enc_vector])

    def decrypt_matrix(self, enc_matrix):
        if not isinstance(enc_matrix[0], list):
            return self.decrypt_vector(enc_matrix)
        return np.array([self.decrypt_vector(row) for row in enc_matrix])

    def decrypt_gradients(self, enc_gradients):
        gradients = []
        for grad in enc_gradients:
            gradients.append(self.decrypt_matrix(grad))
        return gradients

########
# CKKS #
########

class CKKS(Cryptosystem):
    def __init__(self) -> None:
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
          )
        self.context.generate_galois_keys()
        self.context.global_scale = 2**40

    def get_pubkey(self):
        return self.context.public_key()

    def get_context(self):
        return self.context

    def get_encryption(self):
        enc = CKKSEnc(self.context)
        return enc

    def get_decryption(self):
        dec = CKKSDec(self.context)
        return dec

class CKKSEnc:
    def __init__(self, context) -> None:
        self.context = context

    def encrypt_vector(self, vector):
        return ts.ckks_vector(self.context, vector)

    def encrypt_gradients(self, gradients):
        enc_grads = []
        for grad in gradients:
            enc_grads.append(ts.ckks_tensor(self.context, grad))
        return enc_grads

    def encrypt_tensor(self, matrix):
        if isinstance(matrix[0], (int, float)):
            return self.encrypt_vector(matrix)
        return ts.ckks_tensor(self.context, matrix)

    def sum(self, x, y):
        return x + y

    def sum_encrypted_gradients(self, x, y):
        return [x[i] + y[i] for i in range(len(x))]

class CKKSDec:
    def __init__(self, context) -> None:
        self.context = context

    def decrypt(self, enc):
        return enc.decrypt()

    def decrypt_gradients(self, enc_gradients):
        gradients = []
        for enc_grad in enc_gradients:
            gradients.append(np.array(self.decrypt(enc_grad).tolist()))
        return gradients