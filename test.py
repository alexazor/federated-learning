import numpy as np
import torch.nn as nn
import torch
from numpy.typing import NDArray
from utils import encrypt_vector, sum_encrypted_vectors


class ModelAdapter:
    def __init__(self, model: nn.Module, criterion: nn.MSELoss, optimizer: torch.optim.Optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_losses = []

        self.__init_device()
        self.model.to(device=self.device)

    def __init_device(self):  # Use GPU or CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        self.device = torch.device('cpu')
        pass

    def __compute_output(self, data):
        data = data.to(device=self.device, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            output = self.model(data)
        return output

    def forward_backward(self, data, label):
        self.model.train()

        data = torch.tensor(data)
        label = torch.tensor(label)
        data = data.to(device=self.device, dtype=torch.float32)
        label = label.to(device=self.device, dtype=torch.float32)
        self.optimizer.zero_grad()  # clear the gradients of all optimized variables
        output = self.model(data)  # forward pass: compute predicted outputs by passing inputs to the model
        loss = self.criterion(output, label)
        loss.backward()  # backward pass: compute gradient of the loss with respect to model parameters
        pass

    def get_gradient(self):
        gradients = []
        for p in self.model.parameters():
            gradients.append(p.grad)
        return gradients

    def update_model(self, gradients: list[NDArray]):
        for p, gradient in zip(self.model.parameters(), gradients):
            p.grad = torch.tensor(gradient)  # update model parameters gradient
        self.optimizer.step()  # perform one optimizer step
        pass

    def compute_loss(self, data, label):
        output = self.__compute_output(data)

        label = torch.tensor(label)
        label = label.to(device=self.device, dtype=torch.float32)
        loss = self.criterion(output, label)
        return loss.item()

    def compute_train_loss(self, data, label):
        loss = self.compute_loss(data, label)
        self.train_losses.append(loss)
        return loss

    def predict(self, X_test) -> np.array:
        output = self.__compute_output(X_test)
        return torch.round(output).to('cpu').numpy().reshape(-1)


class MLPClient:
    """Run linear regression either with local data or by gradient steps,
    where gradients can be sent from remotely.
    Hold the public key and can encrypt gradients to send remotely.
    """

    def __init__(self, name, X, y, pubkey, model: ModelAdapter, n_epoch=100, lr=0.01):
        self.model = model
        self.name = name
        self.pubkey = pubkey
        self.X, self.y = X, y
        self.n_epoch = n_epoch
        self.lr = lr

    def fit(self):
        """Linear regression for n_iter"""

        for _ in range(self.n_epoch):
            self.model.forward_backward(self.X, self.y)
            gradients = self.model.get_gradient()
            self.model.update_model(gradients)

    def predict(self, X):
        """Score test data"""
        return self.model.predict(X)

    def compute_loss(self, X, y):
        return self.model.compute_loss(X, y)

    def encrypted_gradient(self, sum_to=None):
        """Compute gradient. Encrypt it.
        When `sum_to` is given, sum the encrypted gradient to it, assumed
        to be another vector of the same size
        """

        gradients = [encrypt_vector(self.pubkey, gradient) for gradient in  self.model.get_gradient()]

        if sum_to is not None:
            if len(sum_to) != len(gradients):
                raise Exception('Encrypted vectors must have the same size')
            return [sum_encrypted_vectors(grad_sum_to, grad) for grad_sum_to, grad in zip(sum_to, gradients)]
        else:
            return gradients

#    def __progress(self):
#        bar_len = 100
#        filled_len = int(round(bar_len * self.fold_index / float(self.n_split)))
#
#        percents = round(100.0 * self.fold_index / float(self.n_split), 1)
#        bar = '=' * filled_len + '-' * (bar_len - filled_len)
#
#        sys.stdout.write('[%s] %s%s \r' % (bar, percents, '%'))
#        sys.stdout.flush()
