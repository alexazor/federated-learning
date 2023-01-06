import phe as paillier

############
# Paillier #
############


def encryptPaillier(pubKey, message):
    return pubKey.encrypt(message)


def decryptPaillier(privKey, message):
    return privKey.decrypt(message)


########
# CKKS #
########


def encryptCKKS(pubKey, message):
    pass


def decryptCKKS(privKey, pubKey, message):
    pass
