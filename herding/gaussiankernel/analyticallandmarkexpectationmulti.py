import numpy as np

def analytical_landmark_expectation_multi(mu, cov, Y, sigma_k):
    """
    Computes the expectation E[exp(-||X - y||^2 / (2*sigma_k^2))]
    for X ~ N(mu, cov), for multiple landmarks y in one shot.

    For each landmark y_i, the result is:
        (sigma_k^d / sqrt(det(cov + sigma_k^2 I)))
        * exp( -0.5 * (mu - y_i)^T * (cov + sigma_k^2 I)^(-1) * (mu - y_i) )

    :param mu:       Mean vector of the normal distribution, shape (d,)
    :param cov:      Covariance matrix of the normal distribution, shape (d,d)
    :param Y:        Landmark(s). If multiple, shape (m, d). If single, shape (d,)
    :param sigma_k:  Kernel length scale (scalar)
    :return:         A NumPy array of shape (m,) with each entry giving
                     the expectation for the corresponding landmark.
    """

    mu = np.asarray(mu).ravel()
    d = mu.shape[0]

    # Ensure Y is 2D: (m, d), even if a single landmark was passed
    Y = np.atleast_2d(Y)
    if Y.shape[1] != d:
        raise ValueError(
            f"Expected Y with shape (m, {d}), but got {Y.shape}."
        )

    # Precompute A = cov + sigma_k^2 * I, its determinant, and inverse
    A = cov + sigma_k**2 * np.eye(d)
    A_inv = np.linalg.inv(A)
    detA = np.linalg.det(A)

    # Leading factor is the same for all landmarks
    factor = (sigma_k**d) / np.sqrt(detA)

    # Compute the exponent term for each landmark y_i
    #   exponent_i = -0.5 * (mu - y_i)^T * A_inv * (mu - y_i)
    # We'll vectorize this via matrix multiplication.
    diff = Y - mu  # shape (m, d)
    temp = diff @ A_inv  # shape (m, d)
    # elementwise multiply and sum across columns
    # => np.sum(temp * diff, axis=1) yields the quadratic form
    exponents = -0.5 * np.sum(temp * diff, axis=1)  # shape (m,)

    return factor * np.exp(exponents)
