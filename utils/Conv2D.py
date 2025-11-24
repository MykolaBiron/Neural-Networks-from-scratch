from .BaseLayer import BaseLayer

class Conv2D(BaseLayer):
    def __init__(self, kernel_size, n_filters):
        self.kernel_size = kernel_size
        self.n_filters = n_filters