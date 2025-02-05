from herding.gaussiankernel.numericallandmarkexpectation import numerical_landmark_expectation
from herding.gaussiankernel.analyticallandmarkexpectation import analytical_landmark_expectation
import numpy as np
import pytest


# Assuming these were defined (or imported) somewhere as:
# from herding.gaussiankernel.gaussiankernel import (
#     analytical_landmark_expectation,
#     numerical_landmark_expectation
# )

@pytest.mark.parametrize("d", [1, 2, 5])  # test a few dimensions
@pytest.mark.parametrize("sigma_k", [0.5, 1.0, 2.0])
def test_landmark_expectation_convergence(d, sigma_k):
    """
    Test that the numerical estimate of E[exp(-||X-y||^2/(2*sigma^2))]
    converges to the analytical result for X ~ N(mu, cov).

    We test a few dimensions, means, covariances, and sigma_k values.
    """
    rng = np.random.default_rng(seed=42)

    # Construct random mean and covariance
    mu = rng.normal(loc=0, scale=1, size=d)

    # Create a positive definite covariance matrix (e.g. random SPD)
    # A quick way is to generate a random matrix and do A @ A^T, then scale it.
    A = rng.normal(size=(d, d))
    cov = A @ A.T + 0.1 * np.eye(d)  # ensure it's strictly positive definite

    # Landmark
    y = rng.normal(loc=0, scale=1, size=d)

    # Analytical value
    val_analytical = analytical_landmark_expectation(mu, cov, y, sigma_k)

    # For numerical checks, use increasing sample sizes
    for n_samples in [10_000, 50_000]:
        val_numerical = numerical_landmark_expectation(mu, cov, y, sigma_k,
                                                       n_samples=n_samples,
                                                       use_quasi=True)
        err = abs(val_analytical - val_numerical)
        print(
            f"(d={d}, sigma={sigma_k}, n_samples={n_samples}) -> "
            f"analytical={val_analytical:.6f}, numerical={val_numerical:.6f}, diff={err:.6g}"
        )

        # For moderate sample sizes, an error within ~1e-2 or better is typically reasonable.
        assert err < 1e-2, f"Numerical estimate off by {err} for d={d}, sigma={sigma_k}"
