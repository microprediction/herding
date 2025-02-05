import numpy as np

def analytical_landmark_expectation(mu, cov, y, sigma_k):
    """
    Computes the expectation E[exp(-||X-y||^2 / (2*sigma_k^2))]
    for X ~ N(mu, cov) in closed form.

    Known result:
        E [ exp( - (1/(2*sigma^2)) ||X - y||^2 ) ]
      = (1 / sqrt(det(I + cov/sigma^2)))
        * exp( -1/2 (mu - y)^T (cov + sigma^2 I)^{-1} (mu - y) )

    :param mu:      Mean vector of the normal distribution (shape: (d,))
    :param cov:     Covariance matrix of the normal distribution (shape: (d,d))
    :param y:       The "landmark" vector (same shape as mu)
    :param sigma_k: Kernel length scale (scalar)
    :return:        The value of the expectation (float)
    """
    # Make sure mu, y are 1D arrays
    mu = np.asarray(mu).ravel()
    y = np.asarray(y).ravel()

    # Dimension
    d = mu.shape[0]

    # A = cov + sigma^2 I
    A = cov + sigma_k ** 2 * np.eye(d)

    # The leading factor = 1 / sqrt(det(I + cov/sigma_k^2))
    # Numerically, it is often convenient to compute as:
    #     factor = (sigma_k^d) / sqrt(det(A))
    factor = (sigma_k ** d) / np.sqrt(np.linalg.det(A))

    # Exponent term = -1/2 (mu - y)^T A^{-1} (mu - y)
    diff = mu - y
    exponent = -0.5 * diff @ np.linalg.inv(A) @ diff

    return factor * np.exp(exponent)
