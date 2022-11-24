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

    def forward_backward(self, data: NDArray, label: NDArray):
        self.model.train()

        data = torch.tensor(data)
        label = torch.tensor(label)

        data = data.to(device=self.device, dtype=torch.float32)
        label = label.to(device=self.device, dtype=torch.float32)  # .unsqueeze(1)

        self.optimizer.zero_grad()  # clear the gradients of all optimized variables
        output = self.model.forward(data)  # forward pass: compute predicted outputs by passing inputs to the model
        loss = self.criterion(output, label.unsqueeze(1))
        loss.backward()  # backward pass: compute gradient of the loss with respect to model parameters
        # self.optimizer.step()
        pass

    def get_gradient(self) -> list[NDArray]:
        gradients = []
        for p in self.model.parameters():
            gradients.append(p.grad.cpu().detach().numpy().astype(np.dtype('float64')))
        return gradients

    def update_model(self, gradients: list[NDArray]):
        for p, gradient in zip(self.model.parameters(), gradients):
            p.grad = torch.tensor(gradient, dtype=torch.float32, device=self.device)  # update model parameters gradient
        self.optimizer.step()  # perform one optimizer step
        pass

    def compute_loss(self, data: NDArray, label: NDArray) -> float:
        output = self.__compute_output(data)

        label = torch.tensor(label)
        label = label.to(device=self.device, dtype=torch.float32)
        loss = self.criterion(output, label.unsqueeze(1))
        return loss.item()

    def compute_train_loss(self, data: NDArray, label: NDArray) -> float:
        loss = self.compute_loss(data, label)
        self.train_losses.append(loss)
        return loss

    def predict(self, X_test: NDArray) -> np.array:
        output = self.__compute_output(X_test)
        return output.to('cpu').numpy().reshape(-1)

