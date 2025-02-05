#!/usr/bin/env python3

from herding.gaussiankernel.gaussiankernel import numerical_landmark_expectation, analytical_landmark_expectation
import numpy as np
import matplotlib.pyplot as plt
from randomcov import random_covariance_matrix


def main():
    rng = np.random.default_rng(seed=42)

    # Choose dimension and RBF sigma
    d = 2
    sigma_k = 1.0
    mu = rng.normal(size=d)
    cov = random_covariance_matrix(n=2)
    # Random landmark
    y = rng.normal(size=d)

    # Analytical (exact) value
    val_analytical = analytical_landmark_expectation(mu, cov, y, sigma_k)

    # Vary sample sizes
    sample_sizes = [500, 1000, 2000, 5000, 10_000, 20_000, 50_000]
    abs_errors = []

    for n_samples in sample_sizes:
        val_numerical = numerical_landmark_expectation(
            mu, cov, y, sigma_k,
            n_samples=n_samples,
            use_quasi=True  # Quasi-Monte Carlo via Sobol
        )
        err = abs(val_analytical - val_numerical)
        abs_errors.append(err)
        print(
            f"n={n_samples}, "
            f"analytical={val_analytical:.6g}, "
            f"numerical={val_numerical:.6g}, "
            f"error={err:.2e}"
        )

    # Plotting
    plt.figure(figsize=(6, 4))
    plt.plot(sample_sizes, abs_errors, marker='o', linestyle='--', color='b', label='Absolute Error')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'Convergence of Numerical vs. Analytical (d={d}, sigma={sigma_k})')
    plt.xlabel('Number of Samples (log scale)')
    plt.ylabel('Absolute Error (log scale)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
