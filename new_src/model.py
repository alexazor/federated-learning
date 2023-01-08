from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from numpy._typing import NDArray


class Model(ABC):
    @abstractmethod
    def update_model(self, gradients: list[NDArray]):
        pass

    @abstractmethod
    def predict(self, data: NDArray):
        pass

    @abstractmethod
    def compute_gradient(self):
        pass

    @abstractmethod
    def compute_loss(self, data: NDArray, label: NDArray):
        pass

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def get_label(self):
        pass

    @abstractmethod
    def set_data(self):
        pass

    @abstractmethod
    def set_label(self):
        pass

    @abstractmethod
    def reset_weights(self):
        pass

class ModelLinearRegression(Model):
    def __init__(self, nb_input, data, label, lr=0.01):
        self.weights = np.zeros(nb_input)
        self.data = data
        self.label = label
        self.lr = lr

    def compute_gradient(self) -> list[NDArray]:
        delta = self.predict(self.data) - self.label
        return [delta.dot(self.data)]

    def update_model(self, gradients: list[NDArray]):
        self.weights -= self.lr * gradients[0]
        pass

    def predict(self, data: NDArray) -> NDArray:
        data = np.array(data)
        return data.dot(self.weights)

    def compute_loss(self, data: NDArray, label: NDArray) -> float:
        prediction = self.predict(data)
        return np.linalg.norm(prediction - label) ** 2 / label.shape[0]

    def get_data(self):
        return self.data

    def get_label(self):
        return self.label

    def set_data(self, data: NDArray):
        self.data = data

    def set_label(self, label: NDArray):
        self.label = label

    def reset_weights(self):
        nb_input = len(self.weights)
        self.weights = np.zeros(nb_input)

class ModelMLP(Model):
    def __init__(self, model: nn.Module, criterion: nn.MSELoss, optimizer: torch.optim.Optimizer, data: NDArray, label: NDArray):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_losses = []

        self.__init_device()
        self.model.to(device=self.device)

        self.data, self.label = data, label
        if data != None:
            self.data = torch.tensor(data).to(device=self.device, dtype=torch.float32)
        if label != None:
            self.label = torch.tensor(label).to(device=self.device, dtype=torch.float32).unsqueeze(1)

    def __init_device(self):  # Use GPU or CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        pass

    def __compute_output(self, data: NDArray) -> torch.Tensor:
        data = torch.tensor(data).to(device=self.device, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            output = self.model(data)
        return output

    def compute_gradient(self) -> list[NDArray]:
        self.model.train()
        self.optimizer.zero_grad()  # clear the gradients of all optimized variables
        # forward pass: compute predicted outputs by passing inputs to the model
        output = self.model.forward(self.data)
        loss = self.criterion(output, self.label)
        self.train_losses.append(loss.item())
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        gradients = []
        for p in self.model.parameters():
            gradients.append(p.grad.cpu().detach().numpy().astype(np.dtype('float64')))
        return gradients

    def update_model(self, gradients: list[NDArray]):
        for p, gradient in zip(self.model.parameters(), gradients):
            # update model parameters gradient
            p.grad = torch.tensor(gradient, dtype=torch.float32, device=self.device)
        self.optimizer.step()  # perform one optimizer step
        pass

    def compute_loss(self, data: NDArray, label: NDArray) -> float:
        # For evaluation
        output = self.__compute_output(data)
        label = torch.tensor(label).to(device=self.device, dtype=torch.float32).unsqueeze(1)
        loss = self.criterion(output, label)
        return loss.item()

    def predict(self, X_test: NDArray) -> NDArray:
        output = self.__compute_output(X_test)
        return output.to('cpu').numpy().reshape(-1)

    def get_data(self):
        return self.data

    def get_label(self):
        return self.label

    def set_data(self, data: NDArray):
        self.data = torch.tensor(data).to(device=self.device, dtype=torch.float32)

    def set_label(self, label: NDArray):
        self.label = torch.tensor(label).to(device=self.device, dtype=torch.float32).unsqueeze(1)

    def set_training_data(self, data: NDArray, label: NDArray):
        self.set_data(data)
        self.set_label(label)
        pass

    def reset_weights(self):
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class MLP(nn.Module):
    def __init__(self, nb_input: int):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(nb_input, 1)
        # self.l2 = nn.Linear(2, 1)
        # self.l3 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.l1(x)
        # x = self.l2(x)
        # x = self.l3(x)
        return x


def get_model_lin_reg(nb_input, data, label, **kwargs):
    return ModelLinearRegression(nb_input, data, label, **kwargs)


def get_model_mlp(nb_input, data, label, **kwargs):
    model = MLP(nb_input)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), **kwargs)
    return ModelMLP(model, criterion, optimizer, data, label)
