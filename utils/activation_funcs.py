import numpy as np

def mse(actual, predcited):
    """Compute mean squared error loss"""
    return ((actual  - predcited)**2)/ len(actual)

def mse_prime(actual, predicted):
    """Compute a derivative of mean squared error loss"""
    return 2*(predicted - actual) / np.size(actual)