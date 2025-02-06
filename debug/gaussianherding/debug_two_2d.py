#!/usr/bin/env python3

"""
2D Herding Animation Demo (Selecting Two Points Per Iteration)

Modifications:
 1) Instead of choosing **one** point per iteration, this version selects **two**.
 2) The best pair is chosen by evaluating candidate pairs against the herding objective.
 3) The update rule considers **both** points when updating expectations.
 4) Small random noise is added to **both** chosen points before finalizing.
 5) Repulsion term ensures samples are not too close to each other.

"""

import numpy as np
import math
from randomcov.randomcorrelationmatrix import random_correlation_matrix

from debug.gaussianherding.debug_herding_2d import NUM_CANDIDATES
from herding.animation.landmarkanimation2d import landmark_animation_2d
from herding.animation.discrepancyanimation2d import discrepancy_animation_2d
from herding.gaussiankernel.analyticallandmarkexpectationmulti import (
    analytical_landmark_expectation_multi,
)
from herding.gaussiankernel.gaussiankernelmulti import gaussian_kernel_multi

# ------------------------------- Parameters -------------------------------
EXPONENT = 1.3         # Exponent for the distance-based measure
NUM_LANDMARKS = 30     # Number of 2D landmarks
NUM_CANDIDATES = 40
KERNEL_SIZE = 1.0      # sigma_k for the RBF kernel
REPULSION_WEIGHT = 0.00001  # Repulsion scaling factor
EPS = 1e-8             # Small offset to avoid division by zero
NOISE_SCALE = 0.05     # Scale of random noise added to each chosen sample
TRUE_RHO = -0.8        # True correlation value
animator = discrepancy_animation_2d  # or landmark_animation_2d

def debug_herding_2d():
    rng = np.random.default_rng()

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
          - Evaluate **pairs** of candidates instead of single points
          - Compute distance-based measure for each pair
          - Compute repulsion penalty for each pair
          - Choose the best **pair** (x1, x2)
          - Add small random noise to both x1 and x2
          - Recompute kernel for both x1 and x2
          - Update running sum and running count using both points
          - Yield (x1_noisy, x2_noisy, running_landmark_expectation)
        """
        running_sum = np.zeros(m)
        running_count = 0
        sample_history = []  # Store previously chosen points


        def pick_best_pair(candidates):
            nonlocal running_sum, running_count

            # Compute kernel vectors for all candidates
            gk = np.array([gaussian_kernel_multi(pt, landmarks, sigma_k) for pt in candidates])

            # Generate all possible pairs of candidates
            pair_indices = np.triu_indices(len(candidates), k=1)
            pairs = np.stack((candidates[pair_indices[0]], candidates[pair_indices[1]]), axis=1)
            gk_pairs = np.stack((gk[pair_indices[0]], gk[pair_indices[1]]), axis=1)

            # Compute new means for each pair
            new_means = (running_sum[None, :] + gk_pairs.sum(axis=1)) / (running_count + 2)

            # Compute distance-based measure
            dist_arr = np.abs(new_means - landmark_expectations[None, :]) ** EXPONENT
            dist_score = np.sum(dist_arr, axis=1)

            # Compute total score
            total_score = dist_score
            best_idx = np.argmin(total_score)

            return pairs[best_idx], gk_pairs[best_idx]

        candidates = None

        for i in range(n_iters):
            if (i % refresh_every) == 0 or (candidates is None):
                # Re-generate candidate points
                candidates = rng.normal(loc=0.0, scale=2.0, size=(candidate_size, 2))

            (best_x1, best_x2), (best_gk1, best_gk2) = pick_best_pair(candidates)

            # ----------- Add small random noise to both chosen points -----------
            noise_discount = math.exp(-0.001*running_count)
            noise_scale = NOISE_SCALE*noise_discount
            noise1 = rng.normal(scale=noise_scale, size=2)
            noise2 = rng.normal(scale=noise_scale, size=2)
            best_x1_noisy = best_x1 + noise1
            best_x2_noisy = best_x2 + noise2

            # Recompute kernel for noisy versions
            k_noisy1 = gaussian_kernel_multi(best_x1_noisy, landmarks, sigma_k)
            k_noisy2 = gaussian_kernel_multi(best_x2_noisy, landmarks, sigma_k)

            # Update running sums and counts using both points
            running_sum += k_noisy1 + k_noisy2
            running_count += 2
            sample_history.append(best_x1_noisy)
            sample_history.append(best_x2_noisy)

            running_landmark_expectation = running_sum / running_count

            yield (best_x1_noisy, best_x2_noisy, running_landmark_expectation)

    # ------------------ 5) Animate the results ------------------
    gen = herding_generator(n_iters=2000)
    animator(landmarks, landmark_expectations, sigma_k, gen, true_rho=true_rho)

    # Print summary
    print(f"Generated {m} 2D landmarks.")
    print("Random correlation matrix:\n", true_cov)
    print("True Landmark Expectations (first 5 shown):", landmark_expectations[:5])


if __name__ == "__main__":
    debug_herding_2d()
