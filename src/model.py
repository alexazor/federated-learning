from abc import ABC, abstractmethod
import numpy as np
import torch.nn as nn
import torch
from numpy._typing import NDArray


class Model(ABC):
    @abstractmethod
    def update_model(self, gradients: list[NDArray]):
        pass

    @abstractmethod
    def predict(self, data: NDArray):
        pass

    @abstractmethod
    def compute_gradient(self, data: NDArray, label: NDArray):
        pass

    @abstractmethod
    def compute_loss(self, data: NDArray, label: NDArray):
        pass


class ModelLinearRegression(Model):
    def __init__(self, nb_input, lr=0.01):
        self.weights = np.zeros(nb_input)
        self.lr = lr

    def compute_gradient(self, data: NDArray, label: NDArray) -> list[NDArray]:
        delta = self.predict(data) - label
        return [delta.dot(data)]

    def update_model(self, gradients: list[NDArray]):
        self.weights -= self.lr * gradients[0]
        pass

    def predict(self, data: NDArray) -> NDArray:
        data = np.array(data)
        return data.dot(self.weights)

    def compute_loss(self, data: NDArray, label: NDArray) -> float:
        prediction = self.predict(data)
        return np.linalg.norm(prediction - label) ** 2 / label.shape[0]


class ModelMLP(Model):
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

    def compute_gradient(self, data: NDArray, label: NDArray) -> list[NDArray]:
        self.model.train()

        data = torch.tensor(data)
        label = torch.tensor(label)

        data = data.to(device=self.device, dtype=torch.float32)
        label = label.to(device=self.device,
                         dtype=torch.float32)  # .unsqueeze(1)

        self.optimizer.zero_grad()  # clear the gradients of all optimized variables
        # forward pass: compute predicted outputs by passing inputs to the model
        output = self.model.forward(data)
        loss = self.criterion(output, label.unsqueeze(1))
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        gradients = []
        for p in self.model.parameters():
            gradients.append(p.grad.cpu().detach(
            ).numpy().astype(np.dtype('float64')))
        return gradients

    def update_model(self, gradients: list[NDArray]):
        for p, gradient in zip(self.model.parameters(), gradients):
            # update model parameters gradient
            p.grad = torch.tensor(
                gradient, dtype=torch.float32, device=self.device)
        self.optimizer.step()  # perform one optimizer step
        pass

    def compute_loss(self, data: NDArray, label: NDArray) -> float:
        output = self.__compute_output(data)

        label = torch.tensor(label)
        label = label.to(device=self.device, dtype=torch.float32)
        loss = self.criterion(output, label.unsqueeze(1))
        return loss.item()

    def predict(self, X_test: NDArray) -> NDArray:
        output = self.__compute_output(X_test)
        return output.to('cpu').numpy().reshape(-1)


class MLP(nn.Module):
    def __init__(self, nb_input: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(nb_input, 1)
        # self.fc2 = nn.Linear(2, 1)
        # self.fc3 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        return x


def get_model_lin_reg(nb_input, **kwargs):
    return ModelLinearRegression(nb_input, **kwargs)


def get_model_mlp(nb_input, **kwargs):
    model = MLP(nb_input)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), **kwargs)
    return ModelMLP(model, criterion, optimizer)
