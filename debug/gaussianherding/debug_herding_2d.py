#!/usr/bin/env python3

"""
2D Herding Animation Demo with Repulsion + Noise

Steps:
 1) Generate m random landmarks in 2D.
 2) Generate a random 2x2 correlation matrix (positive-definite).
 3) Compute the exact (analytical) landmark expectations for X ~ N(0, true_cov).
 4) Create a generator (herding_generator) that:
    - Maintains a running sum of kernel vectors over chosen samples.
    - Periodically (e.g., every 20 steps) generates a fresh candidate set.
    - For each candidate x, computes:
        * The 'distance-based' herding objective vs. the true mean
        * A 'repulsion' penalty that grows if x is close to older samples
        * total_score = dist_score + repulsion_weight * repulsion
      Then picks the candidate with minimal total_score.
    - **Adds small random noise** to that best candidate x before finalizing it.
    - Recomputes the kernel vector for the noisy version and updates the herding sums.
    - Yields (x_noisy, running_landmark_expectation).
 5) Visualize with your chosen animator (e.g., discrepancy_animation2d).
"""

import numpy as np
import math

# If you have your own random covariance or correlation matrix generator:
from randomcov.randomcorrelationmatrix import random_correlation_matrix
from statsmodels.distributions.copula.archimedean import FrankCopula

from herding.animation.landmarkanimation2d import landmark_animation_2d
from herding.animation.discrepancyanimation2d import discrepancy_animation_2d
from herding.gaussiankernel.analyticallandmarkexpectationmulti import (
    analytical_landmark_expectation_multi,
)
from herding.gaussiankernel.gaussiankernelmulti import gaussian_kernel_multi

# ------------------------------- Parameters -------------------------------

EXPONENT = 2.0         # Exponent for the distance-based measure
NUM_LANDMARKS = 20       # Number of 2D landmarks
KERNEL_SIZE = 0.5       # sigma_k for the RBF kernel
REPULSION_WEIGHT = 0e-6  # scaling factor for repulsion penalty
REPULSION_EXPONENT = 1.0
EPS = 1e-8             # small offset to avoid dividing by zero
NOISE_SCALE = 0.0      # scale of random noise added to each chosen sample
FRACTION = 1.0
TRUE_RHO = -0.75
NUM_CANDIDATES = 5000

animator = discrepancy_animation_2d  # or landmark_animation_2d, your choice

def debug_herding_2d():
    rng = np.random.default_rng(seed=42)

    # 1) Generate random 2D landmarks
    m = NUM_LANDMARKS
    landmarks = rng.uniform(-5, 5, size=(m, 2))

    # 2) Generate a random 2x2 correlation matrix
    true_cov = random_correlation_matrix(n=2)
    true_cov[0][1] = TRUE_RHO
    true_cov[1][0] = TRUE_RHO
    true_rho = true_cov[0, 1]
    print(f"true_rho = {true_rho:.3f}")

    # 3) Compute "true" landmark expectations:
    #    E[exp(-||X - y||^2/(2*sigma^2))], X ~ N(0, true_cov).
    sigma_k = KERNEL_SIZE
    mu = np.zeros(2)
    landmark_expectations = analytical_landmark_expectation_multi(
        mu, true_cov, landmarks, sigma_k
    )

    # ------------------ 4) Define the Herding Generator ------------------
    def herding_generator(n_iters=2000, candidate_size=NUM_CANDIDATES, refresh_every=20):
        """
        Each iteration:
          - Possibly refresh candidate set
          - For each candidate x:
              * Evaluate kernel vector gk(x) w.r.t. landmarks
              * distance-based score: dist_score = sum( abs( new_mean - landmark_expectations )^EXPONENT )
              * repulsion penalty: sum(1/(||x - old_sample||^2 + EPS)) / #samples
              * total_score = dist_score + REPULSION_WEIGHT * repulsion
          - Pick candidate with minimal total_score
          - Add small random noise to best_x -> best_x_noisy
          - Recompute kernel for best_x_noisy, update sums
          - Yield (best_x_noisy, running_landmark_expectation)
        """
        running_sum = np.zeros(m)
        running_count = 0
        sample_history = []  # store previously chosen points

        # -- helper to compute repulsion for each candidate --
        def compute_repulsion(candidates):
            """
            For each candidate, sum(1/(||x-c||^2 + EPS)) over old samples c in sample_history,
            then average by #samples to keep it scale-limited.
            Output shape: (len(candidates),)
            """
            if not sample_history:
                return np.zeros(len(candidates))
            old_arr = np.array(sample_history)  # shape (n, 2)
            c_expanded = candidates[:, None, :]  # shape (candidate_size, 1, 2)
            o_expanded = old_arr[None, :, :]     # shape (1, n, 2)

            sq_dists = np.sum(np.abs((c_expanded - o_expanded))**REPULSION_EXPONENT, axis=2)  # shape (candidate_size, n)
            repulsions = np.sum(1.0 / (sq_dists + EPS), axis=1)
            repulsions /= np.median(repulsions)
            return repulsions

        def pick_best_candidate(candidates):
            nonlocal running_sum, running_count


            # Evaluate kernel vectors for all candidates
            gk = np.array([gaussian_kernel_multi(pt, landmarks, sigma_k) for pt in candidates])

            # new_mean[i] = (running_sum + gk[i]) / (running_count + 1)
            new_means = (running_sum[None, :] + gk*FRACTION) / (running_count + FRACTION)

            # distance-based measure
            dist_arr = np.abs(new_means - landmark_expectations[None, :])**EXPONENT
            dist_score = np.sum(dist_arr, axis=1)  # shape (candidate_size,)

            # repulsion-based measure
            repulsions = compute_repulsion(candidates)  # shape (candidate_size,)

            import math
            repulsion_discount = math.exp(-0.001*running_count)

            # total score
            total_score = dist_score + REPULSION_WEIGHT * repulsions * repulsion_discount
            best_idx = np.argmin(total_score)

            # Remove the candidate from the list
            if candidates.shape[0] > 5:
                candidates = np.delete(candidates, best_idx, axis=0)

            return candidates[best_idx], gk[best_idx]

        candidates = None

        for i in range(n_iters):
            if (i % refresh_every) == 0 or (candidates is None):
                # Re-generate candidate points
                candidates = rng.normal(loc=0.0, scale=2.0, size=(candidate_size, 2))

            best_x, best_gk = pick_best_candidate(candidates)

            # Remove the candidate from the list


            # ----------- Add small random noise to best_x -----------
            # Example: reduce noise a bit over iterations, or keep it constant
            noise_reduction = math.exp(-0.001*running_count)
            noise = rng.normal(scale=noise_reduction * NOISE_SCALE, size=2)
            best_x_noisy = best_x + noise

            # Re-compute kernel for the noisy version
            k_noisy = gaussian_kernel_multi(best_x_noisy, landmarks, sigma_k)

            # Update sums
            running_sum += k_noisy
            running_count += 1
            sample_history.append(best_x_noisy)

            running_landmark_expectation = running_sum / running_count

            yield (best_x_noisy, running_landmark_expectation)

    # ------------------ 5) Animate the results ------------------
    gen = herding_generator(n_iters=2000)
    animator(landmarks, landmark_expectations, sigma_k, gen, true_rho=true_rho)

    # Print summary
    print(f"Generated {m} 2D landmarks in [-5, 5].")
    print("Random correlation matrix:\n", true_cov)
    print("True Landmark Expectations (first 5 shown):", landmark_expectations[:5])


if __name__ == "__main__":
    debug_herding_2d()
