import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


class NeuralNetwork(nn.Module):
    """
    A flexible feedforward neural network class with configurable parameters.
    """

    def __init__(self, input_size, output_size, hidden_layers=[64, 64],
                 activation=nn.ReLU, optimizer_type=optim.Adam,
                 loss_function=nn.MSELoss, learning_rate=0.001):
        """
        Initializes the neural network with given architecture and parameters.
        :param input_size: Number of input features.
        :param output_size: Number of output neurons.
        :param hidden_layers: List containing the number of neurons per hidden layer.
        :param activation: Activation function for hidden layers (default: ReLU).
        :param optimizer_type: Optimizer (default: Adam).
        :param loss_function: Loss function (default: MSELoss).
        :param learning_rate: Learning rate for the optimizer.
        """
        super(NeuralNetwork, self).__init__()

        # Construct layers dynamically
        layers = []
        prev_size = input_size
        for hidden in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden))
            layers.append(activation())
            prev_size = hidden
        layers.append(nn.Linear(prev_size, output_size))

        self.model = nn.Sequential(*layers)
        self.loss_function = loss_function()
        self.optimizer = optimizer_type(self.parameters(), lr=learning_rate)

    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)

    def train_model(self, train_loader, epochs=10):
        """
        Trains the model using the given dataset.
        :param train_loader: PyTorch DataLoader for training.
        :param epochs: Number of training epochs.
        """
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.loss_function(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.6f}")

    def predict(self, x):
        """
        Makes predictions using the trained model.
        :param x: Input tensor.
        :return: Model predictions.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def evaluate(self, test_loader):
        """
        Evaluates the model performance.
        :param test_loader: PyTorch DataLoader for evaluation.
        """
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = self.forward(inputs)
                loss = self.loss_function(outputs, targets)
                total_loss += loss.item()
        print(f"Test Loss: {total_loss / len(test_loader):.6f}")

    def summary(self):
        """Prints a summary of the model architecture."""
        print(self.model)


class TrainingData(Dataset):
    """
    Handles dataset creation and loading for training and evaluation.
    """

    def __init__(self, inputs, targets):
        """
        Initializes the dataset.
        :param inputs: Numpy array or tensor of input data.
        :param targets: Numpy array or tensor of target values.
        """
        if isinstance(inputs, np.ndarray):
            self.inputs = torch.tensor(inputs, dtype=torch.float32)
        else:
            self.inputs = inputs.float()

        if isinstance(targets, np.ndarray):
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = targets.float()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    @staticmethod
    def get_dataloader(inputs, targets, batch_size=32, shuffle=True):
        """
        Creates a DataLoader from input and target data.
        :param inputs: Input features.
        :param targets: Corresponding target values.
        :param batch_size: Batch size for training/testing.
        :param shuffle: Whether to shuffle the dataset.
        :return: PyTorch DataLoader object.
        """
        dataset = TrainingData(inputs, targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



