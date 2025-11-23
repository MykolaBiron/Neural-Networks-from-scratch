import numpy as np
from .ActivationLayer import ActivationLayer

class Tanh(ActivationLayer):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x : 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)