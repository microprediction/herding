from herding.gaussiankernel.analyticallandmarkexpectation import analytical_landmark_expectation
from herding.gaussiankernel.analyticallandmarkexpectationmulti import analytical_landmark_expectation_multi
import pytest
import numpy as np

# Compare analytical landmark computations


@pytest.mark.parametrize("d", [1, 2, 5])
def test_analytical_landmark_multi(d):
    """
    Compare the single-landmark analytical computation vs. the multi-landmark
    version by verifying that the results match for each landmark.
    """
    rng = np.random.default_rng(seed=42)

    # Random mean
    mu = rng.normal(size=d)
    # Random positive definite covariance
    A = rng.normal(size=(d, d))
    cov = A @ A.T + 0.1 * np.eye(d)
    # Kernel scale
    sigma_k = 1.0

    # Generate multiple landmarks, shape (m, d)
    m = 4  # e.g., test with 4 landmarks
    Y = rng.normal(size=(m, d))

    # Compute multi-landmark result
    vals_multi = analytical_landmark_expectation_multi(mu, cov, Y, sigma_k)

    # Check each landmark individually
    for i in range(m):
        y_i = Y[i]
        val_single = analytical_landmark_expectation(mu, cov, y_i, sigma_k)
        val_multi_i = vals_multi[i]

        # Compare
        err = abs(val_single - val_multi_i)
        print(f"Landmark {i}: single={val_single:.6f}, multi={val_multi_i:.6f}, diff={err:.2e}")
        assert err < 1e-12, f"Mismatch for landmark {i} (dim={d})."
