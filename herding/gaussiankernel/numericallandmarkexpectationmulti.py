import numpy as np
from scipy.stats import qmc, norm

def numerical_landmark_expectation_multi(mu, cov, Y, sigma_k, n_samples=10_000, use_quasi=True):
    """
    Estimates E[exp(-||X - y_i||^2/(2*sigma_k^2))] for X ~ N(mu, cov),
    for multiple landmarks {y_i}, using the same sample of X.

    :param mu:        Mean vector of the normal distribution, shape (d,)
    :param cov:       Covariance matrix of the normal distribution, shape (d,d)
    :param Y:         One or more landmark vectors.
                      If a single landmark, shape (d,);
                      if multiple, shape (m, d).
    :param sigma_k:   Kernel length scale (scalar)
    :param n_samples: Number of Monte Carlo (or quasi-Monte Carlo) samples
    :param use_quasi: If True, uses scrambled Sobol quasi-random sampling;
                      otherwise uses standard pseudo-random sampling
    :return:          A NumPy array of shape (m,) giving the expected value
                      for each landmark. If a single landmark was provided,
                      returns shape (1,).
    """
    mu = np.asarray(mu).ravel()
    d = mu.shape[0]

    # Force Y to be 2D: (m, d)
    Y = np.atleast_2d(Y)
    if Y.shape[1] != d:
        raise ValueError(
            f"Expected Y to have shape (m, {d}), but got {Y.shape}."
        )

    m_landmarks = Y.shape[0]

    # --- Generate samples X from N(mu, cov) ---
    if use_quasi:
        # Use scrambled Sobol sequence in [0,1]^d
        sampler = qmc.Sobol(d, scramble=True)
        # Sobol sampler uses powers of 2. Find smallest power of 2 >= n_samples
        p2 = int(np.ceil(np.log2(n_samples)))
        u = sampler.random_base2(m=p2)  # shape (2^p2, d)
        # If needed, truncate to exactly n_samples
        u = u[:n_samples]
        # Transform uniform [0,1] to standard normal draws
        z = norm.ppf(u)
    else:
        # Standard pseudo-random normal
        rng = np.random.default_rng()
        z = rng.normal(size=(n_samples, d))

    # Correlate samples correctly using the Cholesky factor of cov
    L = np.linalg.cholesky(cov)
    # X has shape (n_samples, d)
    X = mu + z @ L.T

    # --- Compute the kernel values for all landmarks in a vectorized way ---
    #
    # diff[i, j, :] = X[i, :] - Y[j, :]
    # shape: (n_samples, m_landmarks, d)
    # Then sum of squares along axis=2 -> shape (n_samples, m_landmarks)
    # Then exponentiate and average across samples (axis=0).
    #
    # For memory efficiency with large n_samples * m_landmarks, you could
    # loop or do a chunked approach. For moderate sizes, the direct broadcast
    # is simpler.
    #
    diff = X[:, None, :] - Y[None, :, :]     # shape = (n_samples, m_landmarks, d)
    sq_norms = np.sum(diff**2, axis=2)       # shape = (n_samples, m_landmarks)
    values = np.exp(-sq_norms / (2 * sigma_k**2))  # shape = (n_samples, m_landmarks)

    # Take the average over samples -> shape (m_landmarks,)
    return np.mean(values, axis=0)
