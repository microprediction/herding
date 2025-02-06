#!/usr/bin/env python3

"""
2D Herding Animation Demo with a Tiny Repulsion Term

Steps:
 1) Generate m random landmarks in 2D.
 2) Generate a random 2x2 covariance matrix for "true" distribution.
 3) Compute the exact (analytical) landmark expectations for X ~ N(0, true_cov).
 4) Create a generator (herding_generator) that:
    - Maintains a running sum of kernel vectors over chosen samples.
    - Periodically (e.g., every 5 steps) generates a fresh candidate set.
    - For each candidate x, computes the potential "new mean" if x were added.
    - **Adds a small repulsion penalty** to avoid picking points too close
      to previously chosen samples.
    - Chooses the candidate x that yields the minimal combined score.
    - Yields (best_x, running_landmark_expectation) each iteration.
 5) Visualize with the provided animation functions (e.g. discrepancy_animation2d).
"""

import numpy as np
from debug.gaussianherding.debug_herding_2d import TRUE_RHO
from herding.animation.discrepancyanimation2d import discrepancy_animation_2d
from randomcov.randomcorrelationmatrix import random_correlation_matrix
from herding.gaussiankernel.analyticallandmarkexpectationmulti import analytical_landmark_expectation_multi
from herding.gaussiankernel.gaussiankernelmulti import gaussian_kernel_multi

EXPONENT = 1.0
NUM_LANDMARKS = 50
KERNEL_SIZE = 1.0
REPULSION_WEIGHT = 0.0000  # small constant factor for repulsion
EPS = 1e-8  # small offset in denominator
FRACTION_HIGH = 1.0
FRACTION_LOW = 0.5

TRUE_RHO = -0.8

animator = discrepancy_animation_2d  # <--- choose which animator you want

def debug_herding_2d():
    rng = np.random.default_rng()

    # 1) Generate ~NUM_LANDMARKS random 2D landmarks in [-3.5, 3.5].
    m = NUM_LANDMARKS
    landmarks = rng.uniform(-3.5, 3.5, size=(m, 2))

    # 2) Generate a random 2x2 correlation matrix (positive definite).
    true_cov = random_correlation_matrix(n=2)
    true_cov[0][1]= TRUE_RHO
    true_cov[1][0] = TRUE_RHO
    true_rho = true_cov[0][1]
    print({'true_rho': true_rho})

    # 3) Compute "true" landmark expectations:
    #    E[exp(-||X - y||^2/(2*sigma^2))], where X ~ N(0, true_cov).
    sigma_k = KERNEL_SIZE
    mu = np.zeros(2)
    landmark_expectations = analytical_landmark_expectation_multi(
        mu, true_cov, landmarks, sigma_k
    )

    # 4) Define a herding generator with a small repulsion term.
    def herding_generator(n_iters=20, candidate_size=2000, refresh_every=20):
        """
        Each iteration:
         - Possibly create a fresh candidate set (every 'refresh_every' steps)
         - For each candidate x:
             * Evaluate gaussian_kernel_multi(x, landmarks, sigma_k)
             * new_mean = (running_sum + k_x) / (running_count + 1)
             * dist_score = sum( abs( new_mean - landmark_expectations )^EXPONENT )
             * repulsion = sum( 1 / (||x - x_old||^2 + EPS) ) for all old samples
             * total_score = dist_score + REPULSION_WEIGHT * repulsion
         - Pick the candidate with minimal total_score
         - Update running_sum, running_count, and sample_history
         - Yield (best_x, running_landmark_expectation)
        """
        running_sum = np.zeros(m)
        running_count = 0
        sample_history = []  # store previously chosen samples (2D points)

        # Define a function to compute the repulsion for each candidate
        def compute_repulsion(candidates):
            """
            For each candidate x, compute sum of 1/(||x - x_old||^2 + EPS)
            over all old samples x_old in sample_history.
            Output shape: (len(candidates),)
            """
            if len(sample_history) == 0:
                return np.zeros(len(candidates))
            # sample_history -> shape (n, 2)
            old_arr = np.array(sample_history)
            # Expand dims so we can do (len(candidates), 1, 2) - (1, n, 2)
            c_expanded = candidates[:, None, :]  # shape (candidate_size, 1, 2)
            o_expanded = old_arr[None, :, :]     # shape (1, n, 2)

            sq_dists = np.sum((c_expanded - o_expanded)**2, axis=2)  # shape (candidate_size, n)

            repulsion = np.sum(1.0/(sq_dists + EPS), axis=1)/(len(sample_history))         # shape (candidate_size,)
            return repulsion

        def pick_best_sample(candidates):
            nonlocal running_sum, running_count
            # 1) Compute kernel vectors for all candidates
            gk = np.array([gaussian_kernel_multi(pt, landmarks, sigma_k) for pt in candidates])

            # 2) distance-based score for each candidate
            #    new_mean = (running_sum + gk[i]) / (running_count+1)
            #    dist_score[i] = sum( |new_mean - landmark_expectations|^EXPONENT )
            fraction = (FRACTION_HIGH-FRACTION_LOW) * np.random.rand()+ FRACTION_LOW
            new_means = (running_sum[None, :] + gk*fraction) / (running_count + fraction)
            dist_arr = np.abs(new_means - landmark_expectations[None, :]) ** EXPONENT
            dist_score = np.sum(dist_arr, axis=1)  # shape (candidate_size,)

            # 3) repulsion-based penalty
            repulsions = compute_repulsion(candidates)  # shape (candidate_size,)

            # 4) total score = dist_score + REPULSION_WEIGHT * repulsions
            total_score = dist_score + REPULSION_WEIGHT * repulsions
            best_idx = np.argmin(total_score)

            best_pt = candidates[best_idx]
            best_k = gk[best_idx]

            # update sums
            running_sum += best_k
            running_count += 1
            sample_history.append(best_pt)

            # new running mean
            running_landmark_expectation = running_sum / running_count
            return best_pt, running_landmark_expectation

        candidates = None

        for i in range(n_iters):
            # Possibly refresh candidate set
            if (i % refresh_every) == 0 or (candidates is None):
                candidates = rng.normal(loc=0.0, scale=2.0, size=(candidate_size, 2))

            best_x, running_exp = pick_best_sample(candidates)
            yield (best_x, running_exp)

    # Create the generator
    gen = herding_generator(n_iters=10000)

    # 5) Animate it
    animator(landmarks, landmark_expectations, sigma_k, gen, true_rho=true_rho)

    # Print a small summary
    print(f"Generated {m} 2D landmarks in [-3.5, 3.5].")
    print(f"Random correlation matrix:\n{true_cov}")
    print(f"true_rho = {true_rho}")
    print("True Landmark Expectations (first 5 shown):", landmark_expectations[:5])


if __name__ == "__main__":
    debug_herding_2d()
