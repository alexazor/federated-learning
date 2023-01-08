import numpy as np
import phe as paillier
import tenseal as ts
from numpy._typing import NDArray

############
# Paillier #
############

class Paillier:
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
        if matrix.ndim == 1:
            return self.encrypt_vector(matrix)
        return [self.encrypt_vector(row) for row in matrix]

    def encrypt_tensor(self, tensor):
        return [self.encrypt_matrix(col) for col in tensor]

    def sum_encrypted_vectors(self, x, y):
        if len(x) != len(y):
            raise Exception('Encrypted vectors must have the same size')
        return [x[i] + y[i] for i in range(len(x))]

    def sum_encrypted_matrix(self, x, y):
        if not isinstance(x[0], list):
            return self.sum_encrypted_vectors(x, y)
        return [[x[i][j] + y[i][j] for j in range(len(x[0]))] for i in range(len(x))]

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

    def decrypt_tensor(self, enc_tensor):
        if not isinstance(enc_tensor[0], list):
            return self.decrypt_vector(enc_tensor)
        if not isinstance(enc_tensor[1], list):
            return self.decrypt_matrix(enc_tensor)
        return np.array([self.decrypt_matrix(col) for col in enc_tensor])

########
# CKKS #
########

class CKKS:
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

    def encrypt_tensor(self, matrix: NDArray):
        if matrix.ndim == 1:
            return self.encrypt_vector(matrix)
        return ts.ckks_tensor(self.context, matrix)

    def sum(self, x, y):
        return x + y

    def sum_encrypted_tensor(self, x, y):
        return self.sum(x, y)

class CKKSDec:
    def __init__(self, context) -> None:
        self.context = context

    def decrypt(self, enc):
        return enc.decrypt()

    def decrypt_tensor(self, enc_matrix):
        return self.decrypt(enc_matrix)