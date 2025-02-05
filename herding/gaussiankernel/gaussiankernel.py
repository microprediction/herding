import numpy as np


def gaussian_kernel(x, y, sigma):
    """
    Gaussian (RBF) kernel k(x,y) = exp(-||x-y||^2 / (2 sigma^2))

    :param x: One d-dimensional point
    :param y: Another d-dimensional point
    :param sigma: Kernel length scale
    :return: The value of the Gaussian kernel
    """
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))


