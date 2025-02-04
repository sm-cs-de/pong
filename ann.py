import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pong_game.config as cfg


class ANN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, activation=nn.ReLU, optimizer_type=optim.SGD, loss_function=nn.CrossEntropyLoss, learning_rate=0.001):
        """
        Initializes the neural network with given architecture and parameters
        :param input_size: Number of input features
        :param output_size: Number of output neurons
        :param hidden_layers: List containing the number of neurons per hidden layer
        :param activation: Activation function for hidden layers
        :param optimizer_type: Optimizer: https://www.ruder.io/optimizing-gradient-descent/
        :param loss_function: Loss function. Possible choices for our problem: NLLLoss, CrossEntropyLoss, KLDivLoss: https://neptune.ai/blog/pytorch-loss-functions
        :param learning_rate: Learning rate for the optimizer
        """
        super(ANN, self).__init__()

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

    def predict(self, input):
        """
        Makes predictions using the trained model.
        :param input: Input tensor.
        :return: Model predictions.
        """
        self.eval()
        with torch.no_grad():
            softmax = nn.Softmax(dim=1)
            return softmax(self.forward(input))[0]

    def evaluate(self, test_loader):
        """
        Evaluates the model performance.
        :param test_loader: PyTorch DataLoader for evaluation.
        """
        self.eval()

        for idx, (inputs, targets) in enumerate(test_loader):
            outputs = self.predict(inputs)
            data = torch.mul(outputs, targets)
            prob = torch.sum(data, 1)
            mean = torch.mean(prob)

            print(f"Test probabilities for batch {idx}: {prob}")
            print(f"Mean: {mean}")

    def ball_prob(self, output):
        conv = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=cfg.player_width, padding='same', padding_mode='zeros', bias=False)
        conv.weight.data = torch.full_like(conv.weight.data, 1)

        return conv(output.unsqueeze(0))[0]

    def ball_goal(self, output):
        return np.sum([i * output[i].item() for i in range(output.size(dim=0))]) / torch.sum(output).item() # we calculate the expectation value (y-index) of the output (distribution)

    def summary(self):
        print("Network: " + str(self.model))
        print("Optimizer: " + str(self.optimizer))
        print("Loss-Fct: " + str(self.loss_function))


class TrainingData(Dataset):
    """
    Handles dataset creation and loading for training and evaluation.
    """

    def __init__(self, inputs, targets):
        """
        Initializes the dataset.
        :param inputs: Numpy array of input data.
        :param targets: Numpy array of target values.
        """
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    @staticmethod
    def get_dataloader(inputs, targets, batch_size=50, shuffle=True):
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



