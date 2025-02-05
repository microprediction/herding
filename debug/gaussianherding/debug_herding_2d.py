#!/usr/bin/env python3

"""
2D Herding Animation Demo

Steps:
 1) Generate m random landmarks in 2D.
 2) Generate a random 2x2 covariance matrix for "true" distribution.
 3) Compute the exact (analytical) landmark expectations for X ~ N(0, true_cov).
 4) Create a generator (herding_generator) that:
    - Maintains a running sum of kernel vectors over chosen samples.
    - Periodically (e.g., every 5 steps) generates a fresh candidate set.
    - For each candidate x, computes the potential "new mean" if x were added.
    - Chooses the candidate that best reduces the Euclidean distance to the true mean.
    - Yields (best_x, running_landmark_expectation) each iteration.
 5) Visualize with landmark_animation_2d.
"""

import numpy as np
from randomcov import random_covariance_matrix

# Assuming these functions are available from your codebase:
from herding.animation.landmarkanimation2d import landmark_animation_2d
from herding.animation.discrepancyanimation2d import discrepancy_animation_2d
from randomcov.randomcorrelationmatrix import random_correlation_matrix
from herding.gaussiankernel.analyticallandmarkexpectationmulti import analytical_landmark_expectation_multi
from herding.gaussiankernel.gaussiankernelmulti import gaussian_kernel_multi


animator = discrepancy_animation_2d  # <--- CHOOSE

def debug_herding_2d():
    rng = np.random.default_rng()

    # 1) Generate ~25 random 2D landmarks in [-5, 5].
    m = 50
    landmarks = rng.uniform(-3.5, 3.5, size=(m, 2))

    # 2) Generate a random 2x2 covariance (positive definite).
    true_cov = random_correlation_matrix(n=2)
    true_rho = true_cov[0][1]
    print({'true_rho':true_rho})

    # 3) Compute "true" landmark expectations:
    #    E[exp(-||X - y||^2/(2*sigma^2))], where X ~ N(0, true_cov).
    sigma_k = 0.25
    # We'll assume the mean is 0 in R^2:
    mu = np.zeros(2)
    landmark_expectations = analytical_landmark_expectation_multi(
        mu, true_cov, landmarks, sigma_k
    )

    # 4) Define a herding generator.
    def herding_generator(n_iters=20, candidate_size=2000, refresh_every=20):
        """
        At each iteration:
         - Possibly create a fresh candidate set (every 'refresh_every' steps)
         - For each candidate x:
             * Evaluate gaussian_kernel_multi(x, landmarks, sigma_k)
             * Estimate new_mean = (running_sum + k_x) / (running_count + 1)
             * Score = ||new_mean - landmark_expectations||^2
         - Pick candidate x that yields minimal score
         - Update running_sum, running_count, running_landmark_expectation
         - Yield (best_x, running_landmark_expectation)
        """
        running_sum = np.zeros(m)
        running_count = 0

        # For convenience, define a function to get the best next sample
        def pick_best_sample(candidates):
            nonlocal running_sum, running_count
            # Compute kernel vectors for all candidates, shape = (candidate_size, m)
            # k[i, :] = kernel of candidates[i] with all landmarks
            gk = np.array([gaussian_kernel_multi(pt, landmarks, sigma_k) for pt in candidates])

            # We'll compute the "score" of choosing each candidate
            # new_mean = (running_sum + k[i]) / (running_count+1)
            # we measure L2 distance from the true mean
            # => dist^2 = ||(running_sum + k[i])/(running_count+1) - landmark_expectations||^2
            # We'll pick the candidate with minimal dist^2

            dist_sqr = np.sum(
                np.abs(
                        (running_sum[None, :] + gk) / (running_count + 1)
                        - landmark_expectations[None, :]
                ) ** 1.3,
                axis=1
            )
            best_idx = np.argmin(dist_sqr)
            best_pt = candidates[best_idx]
            best_k = gk[best_idx]

            # Update running sums
            running_sum += best_k
            running_count += 1

            # The new running mean
            running_landmark_expectation = running_sum / running_count
            return best_pt, running_landmark_expectation

        candidates = None

        for i in range(n_iters):
            # Possibly refresh candidate set
            if (i % refresh_every) == 0 or (candidates is None):
                # Generate candidate_size random 2D points
                # For demonstration, let's do standard normal N(0,I):
                candidates = rng.normal(loc=0.0, scale=2.0, size=(candidate_size, 2))

            # Pick best candidate
            best_x, running_exp = pick_best_sample(candidates)

            # We could remove that candidate from the list if we wanted to avoid re-picking the same point,
            # but herding doesn't necessarily need that. We'll keep it simple.

            yield (best_x, running_exp)

    gen = herding_generator(n_iters=2000)

    # 5) Animate it
    animator(landmarks, landmark_expectations, sigma_k, gen, true_rho=true_rho)

    # Print a small summary
    print(f"Generated {m} 2D landmarks in [-5, 5].")
    print(f"Random true_cov:\n{true_cov}")
    print("True Landmark Expectations (first 5 shown):", landmark_expectations[:5])


if __name__ == "__main__":
    debug_herding_2d()
