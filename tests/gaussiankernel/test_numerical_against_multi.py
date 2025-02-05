from herding.gaussiankernel.numericallandmarkexpectation import numerical_landmark_expectation
from herding.gaussiankernel.numericallandmarkexpectationmulti import numerical_landmark_expectation_multi
import pytest
import numpy as np

# Tests against each other



# These are your existing functions in your codebase
# from herding.gaussiankernel.numericallandmarkexpectation import numerical_landmark_expectation
# from herding.gaussiankernel.numericallandmarkexpectationmulti import numerical_landmark_expectation_multi

@pytest.mark.parametrize("d", [1, 2, 5])
def test_numerical_landmark_multi(d):
    """
    Compare the single-landmark numerical expectation vs. the multi-landmark version,
    ensuring they match for multiple landmarks in a single batch of samples.
    """
    rng = np.random.default_rng(seed=42)

    # Generate random mean
    mu = rng.normal(size=d)

    # Generate a random SPD covariance
    A = rng.normal(size=(d, d))
    cov = A @ A.T + 0.1 * np.eye(d)

    # Kernel scale
    sigma_k = 1.0

    # Number of landmarks
    m = 4
    # Generate random landmarks, shape (m, d)
    Y = rng.normal(size=(m, d))

    # Single pass sample size
    n_samples = 20_000

    # Call the multi-landmark version once
    vals_multi = numerical_landmark_expectation_multi(
        mu, cov, Y, sigma_k,
        n_samples=n_samples,
        use_quasi=True
    )

    # Check each landmark individually
    for i in range(m):
        # Extract the i-th landmark
        y_i = Y[i]
        # Call the single-landmark version
        val_single = numerical_landmark_expectation(
            mu, cov, y_i, sigma_k,
            n_samples=n_samples,
            use_quasi=True
        )
        # Compare
        diff = abs(val_single - vals_multi[i])
        print(f"[d={d}] Landmark {i}: single={val_single:.6f}, multi={vals_multi[i]:.6f}, diff={diff:.2e}")
        # For a large n_samples, we expect good agreement (e.g. < 1e-2)
        assert diff < 1e-2, f"Mismatch in single vs. multi for landmark {i} (dim={d})"
