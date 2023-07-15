import numpy as np

def initialize(a, b):
    epsilon = 0.15
    # randomly initializes an a x (b+1) array of values exisiting in [-epsilon, +epsilon]
    c = np.random.rand(a, b + 1) * (2 * epsilon) - epsilon 
    return c