import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)
        self.biases = np.random.rand(output_size, 1)

    def forward(self, input):
        """Compute activations moving forward to get the output"""
        self.input = input
        return np.dot(self.weights, self.inputs) + self.biases

    def backward(self, output_gradient, learning_rate):
        """Backpropagate through the network using gradient descent"""
        weights_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= learning_rate * weights_gradient
        self.biaases -= learning_rate * output_gradient
        return np.dot(self.weights.T, output_gradient)