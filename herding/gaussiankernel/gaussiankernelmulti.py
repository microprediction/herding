import numpy as np

def gaussian_kernel_multi(x, Y, sigma):
    """
    Vectorized Gaussian (RBF) kernel k(x, y_i) for multiple landmarks.
    For each y_i in Y, we compute exp(-||x-y_i||^2 / (2 sigma^2)).

    :param x: A single d-dimensional point, shape (d,)
    :param Y: Multiple d-dimensional points (landmarks), shape (m, d)
    :param sigma: Kernel length scale (scalar)
    :return: A 1D NumPy array of shape (m,),
             where the i-th entry is the kernel value k(x, Y[i]).
    """
    x = np.asarray(x).ravel()
    Y = np.atleast_2d(Y)
    if Y.shape[1] != x.shape[0]:
        raise ValueError(f"Expected Y.shape = (m, {x.shape[0]}), but got {Y.shape}.")

    # Compute the squared norm difference for each landmark:
    # diff[i] = Y[i] - x
    # sum along axis=1 to get ||Y[i] - x||^2
    diff = Y - x  # shape (m, d)
    sq_dists = np.sum(diff**2, axis=1)  # shape (m,)

    return np.exp(-sq_dists / (2 * sigma**2))
