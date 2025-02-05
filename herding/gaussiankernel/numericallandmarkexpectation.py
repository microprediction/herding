import numpy as np
from scipy.stats import qmc, norm

def numerical_landmark_expectation(mu, cov, y, sigma_k, n_samples=10_000, use_quasi=True):
    """
    Estimates E[exp(-||X-y||^2/(2*sigma_k^2))] for X ~ N(mu, cov)
    by (quasi-)Monte Carlo sampling.

    :param mu:         Mean vector of the normal distribution (shape: (d,))
    :param cov:        Covariance matrix of the normal distribution (shape: (d,d))
    :param y:          The "landmark" vector (same shape as mu)
    :param sigma_k:    Kernel length scale (scalar)
    :param n_samples:  Number of samples to use
    :param use_quasi:  If True, uses Sobol quasi-random sampling;
                       otherwise uses standard pseudo-random sampling
    :return:           The numerical approximation (float)
    """
    mu = np.asarray(mu).ravel()
    y = np.asarray(y).ravel()
    d = mu.shape[0]

    if use_quasi:
        # Use Sobol sequence in [0,1]^d
        sampler = qmc.Sobol(d, scramble=True)
        # Sobol sampler needs sample sizes of 2^m:
        m = int(np.ceil(np.log2(n_samples)))
        u = sampler.random_base2(m=m)  # shape: (2^m, d)

        # Truncate to exactly n_samples if needed
        u = u[:n_samples]

        # Transform uniform [0,1] draws -> standard normal via inverse CDF
        z = norm.ppf(u)
    else:
        # Standard pseudo-random normal draws
        rng = np.random.default_rng()
        z = rng.normal(size=(n_samples, d))

    # Correlate them properly using the Cholesky factor of cov
    L = np.linalg.cholesky(cov)
    # Each row of X is mu + (z_i @ L^T)
    X = mu + z @ L.T

    # Evaluate the kernel for each sampled X
    sq_norms = np.sum((X - y)**2, axis=1)  # ||X - y||^2 per sample
    values = np.exp(-sq_norms / (2 * sigma_k**2))

    return np.mean(values)

