from .BaseLayer import BaseLayer

class Relu(BaseLayer):
    def __init__(self):
        relu = lambda x: max(0, x)
        relu_prime = lambda x: 0 if x < 0 else 1
        super().__init__(relu, relu_prime)