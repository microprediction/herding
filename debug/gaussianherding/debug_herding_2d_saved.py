#!/usr/bin/env python3

"""
2D Herding Animation Demo with Small Random Noise in the Chosen Sample

Steps:
 1) Generate m random landmarks in 2D.
 2) Generate a random 2x2 covariance matrix for "true" distribution.
 3) Compute the exact (analytical) landmark expectations for X ~ N(0, true_cov).
 4) Create a generator (herding_generator) that:
    - Maintains a running sum of kernel vectors over chosen samples.
    - Periodically (e.g., every 5 steps) generates a fresh candidate set.
    - For each candidate x, computes the potential "new mean" if x were added.
    - Chooses the candidate that best reduces the objective distance to the true mean.
    - **Adds small random noise** to the chosen point best_x.
    - Yields (best_x_with_noise, running_landmark_expectation) each iteration.
 5) Visualize with your chosen animator (landmark_animation_2d or discrepancy_animation_2d).
"""

import numpy as np

# Replace if you have your own random covariance generator
from herding.animation.discrepancyanimation2d import discrepancy_animation_2d
from randomcov.randomcorrelationmatrix import random_correlation_matrix
from herding.gaussiankernel.analyticallandmarkexpectationmulti import analytical_landmark_expectation_multi
from herding.gaussiankernel.gaussiankernelmulti import gaussian_kernel_multi

EXPONENT = 1.0
NUM_LANDMARKS = 17
KERNEL_SIZE = 0.5
NOISE_SCALE = 0.25  # Scale of the random noise we add to each chosen sample

animator = discrepancy_animation_2d  # or landmark_animation_2d, your choice

def debug_herding_2d():
    rng = np.random.default_rng()

    # 1) Generate ~NUM_LANDMARKS random 2D landmarks in [-3.5, 3.5].
    m = NUM_LANDMARKS
    landmarks = rng.uniform(-5, 5, size=(m, 2))

    # 2) Generate a random 2x2 correlation (positive definite).
    true_cov = random_correlation_matrix(n=2)
    true_rho = true_cov[0, 1]
    print({'true_rho': true_rho})

    # 3) Compute "true" landmark expectations:
    #    E[exp(-||X - y||^2/(2*sigma^2))], for X ~ N(0, true_cov).
    sigma_k = KERNEL_SIZE
    mu = np.zeros(2)
    landmark_expectations = analytical_landmark_expectation_multi(
        mu, true_cov, landmarks, sigma_k
    )

    # 4) Define a herding generator that adds small random noise to each chosen sample.
    def herding_generator(n_iters=20, candidate_size=2000, refresh_every=20):
        """
        Each iteration:
         - Possibly create a fresh candidate set (every refresh_every steps).
         - For each candidate x:
             * Evaluate gaussian_kernel_multi(x, landmarks, sigma_k).
             * new_mean = (running_sum + k_x) / (running_count + 1).
             * score = sum( | new_mean - landmark_expectations |^EXPONENT ).
         - Pick candidate x that yields minimal score.
         - Add small random noise to best_x -> best_x_noisy.
         - Recompute the kernel vector for best_x_noisy (since we actually choose that).
         - Update running_sum, running_count, running_landmark_expectation.
         - Yield (best_x_noisy, running_landmark_expectation).
        """
        running_sum = np.zeros(m)
        running_count = 0

        def pick_best_candidate(candidates):
            nonlocal running_sum, running_count

            # For each candidate, compute k_x (the kernel vector)
            gk = np.array([gaussian_kernel_multi(pt, landmarks, sigma_k) for pt in candidates])
            # Score = sum( |(running_sum + gk[i])/(running_count+1) - landmark_expectations|^EXPONENT )
            new_means = (running_sum[None, :] + gk) / (running_count + 1)
            dist_sqr = np.sum(
                np.abs(new_means - landmark_expectations[None, :]) ** EXPONENT,
                axis=1
            )

            best_idx = np.argmin(dist_sqr)
            return candidates[best_idx], gk[best_idx]

        candidates = None

        for i in range(n_iters):
            # Possibly refresh candidate set
            if (i % refresh_every) == 0 or (candidates is None):
                candidates = rng.normal(loc=0.0, scale=2.0, size=(candidate_size, 2))

            best_x, best_k = pick_best_candidate(candidates)

            # --- Add small random noise to best_x ---
            import math
            noise_reduction = min(1, 1000/math.sqrt(running_count+1000))
            noise = rng.normal(scale=noise_reduction*NOISE_SCALE, size=2)
            best_x_noisy = best_x + noise

            # We need to re-compute the kernel vector for the noisy version
            k_noisy = gaussian_kernel_multi(best_x_noisy, landmarks, sigma_k)

            # Update running sums with the noisy kernel vector
            running_sum += k_noisy
            running_count += 1
            running_landmark_expectation = running_sum / running_count

            yield (best_x_noisy, running_landmark_expectation)

    gen = herding_generator(n_iters=10000)

    # 5) Animate
    animator(landmarks, landmark_expectations, sigma_k, gen, true_rho=true_rho)

    # Print summary
    print(f"Generated {m} 2D landmarks in [-3.5, 3.5].")
    print(f"Random correlation:\n{true_cov}")
    print("True Landmark Expectations (first 5 shown):", landmark_expectations[:5])


if __name__ == "__main__":
    debug_herding_2d()
