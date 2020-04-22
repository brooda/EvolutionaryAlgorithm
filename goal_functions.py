import numpy as np


def constant(x):
    return 1

def gaussian(x, mu, sig):
    return np.sum(np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))

def one_gaussian(x):
    return gaussian(x, 0, 1)

def two_gaussians(x):
    return gaussian(x, 0, 1) + gaussian(x, 3, 1)


