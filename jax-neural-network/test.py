import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 32)
        self.output = nn.Linear(32, 1)
        self.activation = nn.Sigmoid

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        return self.output(x)


nn = NeuralNetwork()
pass
