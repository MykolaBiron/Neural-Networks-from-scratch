import numpy as np
from .BaseLayer import BaseLayer

class DenseLayer(BaseLayer):
    """Represnts a fully connected layer of the network"""
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(output_size, input_size)
        self.biases = np.random.rand(output_size, 1)

    def forward(self, input):
        """Compute activations moving forward to get the output"""
        self.input = input
        return np.dot(self.weights, self.input) + self.biases

    def backward(self, output_gradient, learning_rate):
        """Backpropagate through the network using gradient descent"""
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient