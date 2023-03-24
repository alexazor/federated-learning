from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from numpy._typing import NDArray


class Model(ABC):
    @abstractmethod
    def update_model(self, gradients: list[NDArray]):
        """Applies gradient step to the model

        Args:
            gradients (list[NDArray]): model's gradients
        """
        pass

    @abstractmethod
    def predict(self, data: NDArray):
        """Predicts label on data

        Args:
            data (NDArray): Data

        Returns:
            NDArray: Predicted label
        """
        pass

    @abstractmethod
    def compute_gradient(self):
        """Computes model's gradients on training data

        Returns:
            list[NDArray]: Gradients of the model
        """
        pass

    @abstractmethod
    def compute_loss(self, data: NDArray, label: NDArray):
        """Computes loss on data vs label

        Args:
            data (NDarray): data
            label (NDarray): labels

        Returns:
            float: loss value
        """
        pass

    @abstractmethod
    def get_data(self):
        """Returns data
        """
        pass

    @abstractmethod
    def get_label(self):
        """Returns labels
        """
        pass

    @abstractmethod
    def set_data(self):
        """Set data
        """
        pass

    @abstractmethod
    def set_label(self):
        """Set label
        """
        pass

    @abstractmethod
    def reset_weights(self):
        """Resets the model weights
        """
        pass

class ModelLinearRegression(Model):
    def __init__(self, nb_input, data, label, lr=0.01):
        """Linear regression model

        Args:
            nb_input (int): Number of inputs
            data (NDarray): Data
            label (NDarray): Labels
            lr (float, optional): Learning rate. Defaults to 0.01.
        """
        self.weights = np.zeros(nb_input)
        self.data = data
        self.label = label
        self.lr = lr

    def compute_gradient(self) -> list[NDArray]:
        """Computes model's gradients on training data

        Returns:
            list[NDArray]: Gradients of the model
        """
        delta = self.predict(self.data) - self.label
        return [delta.dot(self.data)]

    def update_model(self, gradients: list[NDArray]):
        """Applies gradient step to the model

        Args:
            gradients (list[NDArray]): model's gradients
        """
        self.weights -= self.lr * gradients[0]
        pass

    def predict(self, data: NDArray) -> NDArray:
        """Predicts label on data

        Args:
            data (NDArray): Data

        Returns:
            NDArray: Predicted label
        """
        #data = np.array(data)
        return data.dot(self.weights)

    def compute_loss(self, data: NDArray, label: NDArray) -> float:
        """Computes loss on data vs label

        Args:
            data (NDarray): data
            label (NDarray): labels

        Returns:
            float: loss value
        """
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
        """Resets the model weights
        """
        nb_input = len(self.weights)
        self.weights = np.zeros(nb_input)

class ModelMLP(Model):
    def __init__(self, model: nn.Module, criterion: nn.MSELoss, optimizer: torch.optim.Optimizer, data: NDArray, label: NDArray):
        """Complete model class.

        Args:
            model (nn.Module): Model
            criterion (nn.MSELoss): Criterion
            optimizer (torch.optim.Optimizer): Optimizer
            data (NDArray): Data
            label (NDArray): Labels
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_losses = []

        self.__init_device()
        self.model.to(device=self.device)

        self.data, self.label = data, label
        if type(data) != type(None):
            self.data = torch.tensor(data).to(device=self.device, dtype=torch.float32)
        if type(label) != type(None):
            self.label = torch.tensor(label).to(device=self.device, dtype=torch.float32).unsqueeze(1)

    def __init_device(self):
        """Divice initialization CPU or GPU
        """
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        pass

    def __compute_output(self, data: NDArray) -> torch.Tensor:
        """Compute output of model on data. Private method.
        See predict() method for prediction.

        Args:
            data (NDArray): Data

        Returns:
            torch.Tensor: Predicted labels
        """
        data = torch.tensor(data).to(device=self.device, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            output = self.model(data)
        return output

    def compute_gradient(self) -> list[NDArray]:
        """Computes model's gradients on training data

        Returns:
            list[NDarray]: Gradients of the model
        """
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
        """Applies gradient step to the model

        Args:
            gradients (list[NDArray]): model's gradients
        """
        for p, gradient in zip(self.model.parameters(), gradients):
            # update model parameters gradient
            p.grad = torch.tensor(gradient, dtype=torch.float32, device=self.device)
        self.optimizer.step()  # perform one optimizer step
        pass

    def compute_loss(self, data: NDArray, label: NDArray) -> float:
        """Computes loss on data vs label

        Args:
            data (NDarray): data
            label (NDarray): labels

        Returns:
            float: loss value
        """
        output = self.__compute_output(data)
        label = torch.tensor(label).to(device=self.device, dtype=torch.float32).unsqueeze(1)
        loss = self.criterion(output, label)
        return loss.item()

    def predict(self, X_test: NDArray) -> NDArray:
        """Predicts label on data

        Args:
            data (NDArray): Data

        Returns:
            NDArray: Predicted label
        """
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

    def reset_weights(self, seed):
        """Resets the model weights
        """
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                torch.manual_seed(seed)
                layer.reset_parameters()

class MLP(nn.Module):
    def __init__(self, nb_input: int):
        """MLP model class

        Args:
            nb_input (int): Number of inputs
        """
        super(MLP, self).__init__()
        self.l1 = nn.Linear(nb_input, 1)
        # self.l2 = nn.Linear(2, 1)
        # self.l3 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.l1(x)
        # x = self.l2(x)
        # x = self.l3(x)
        return x


def get_model_lin_reg(nb_input, data, label, lr=0.01, **kwargs):
    """Returns initialized linear regression object

    Args:
        nb_input (int): Number of inputs
        data (NDArray): Data
        label (NDArray): Labels
        lr (float, optional): Learning rate. Defaults to 0.01.

    Returns:
        ModelLinearRegression: Initialized Linear regression model
    """
    return ModelLinearRegression(nb_input, data, label, lr, **kwargs)


def get_model_mlp(nb_input, data, label, seed=7, **kwargs):
    """Returns initialized MLP object

    Args:
        nb_input (int): Number of inputs
        data (NDArray): Data
        label (NDArray): Labels
        seed (int, optional): Seed for the model weights initialization. Defaults to 7.

    Returns:
        ModelMLP: Initialized MLP model
    """
    torch.manual_seed(seed)
    model = MLP(nb_input)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), **kwargs)
    return ModelMLP(model, criterion, optimizer, data, label)
