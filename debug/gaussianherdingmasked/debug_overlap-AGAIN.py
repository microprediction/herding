#!/usr/bin/env python3

"""
herding_4d_masked.py

Herding code in 4D with masked RBF kernels.
We assume there is a separate file (e.g. `herding/animation/maskedanimation4d.py`)
that provides the function `masked_animation_4d`.
This script imports `masked_animation_4d` and uses it to animate the herding results,
but will update the plot less often by skipping frames.
"""

import numpy as np
import math

from herding.animation.maskedanimation4d import masked_animation_4d  # <-- Import from another file

# Global constants for demonstration
NUM_ITER = 50000
NUM_CANDIDATES = 300
NUM_LANDMARKS = 1000
SIGMA_K = 0.5
EXPONENT = 1.3
NOISE_SCALE = 0.001
SUBSETS = [(0, 1, 2, 3)]  # example single subset using all coords

def masked_gaussian_kernel(x, landmarks, subset, sigma):
    x_sub = x[list(subset)]
    L_sub = landmarks[:, list(subset)]
    diff = L_sub - x_sub[None, :]
    dist_sq = np.sum(diff**2, axis=1)
    return np.exp(-0.5 * dist_sq / (sigma**2))

def analytical_landmark_expectation_masked(mu, cov, landmarks, subset, sigma):
    sub = list(subset)
    mu_sub = mu[sub]
    cov_sub = cov[np.ix_(sub, sub)]
    d = len(sub)

    Sigma_s = cov_sub + sigma**2 * np.eye(d)
    inv_Sigma_s = np.linalg.inv(Sigma_s)
    det_Sigma_s = np.linalg.det(Sigma_s)

    L_sub = landmarks[:, sub]
    results = []
    for j in range(len(landmarks)):
        M_j = L_sub[j] - mu_sub
        exponent = -0.5 * (M_j @ inv_Sigma_s @ M_j)
        prefactor = ((2 * math.pi)**(d / 2) * (det_Sigma_s**0.5))**(-1)
        val = prefactor * np.exp(exponent)
        results.append(val)

    return np.array(results)

def masked_herding_generator(
    mu_4d,
    cov_4d,
    subsets,
    landmarks,
    true_masked_exps,
    sigma_k=1.0,
    n_iters=NUM_ITER,
    candidate_size=NUM_CANDIDATES,
    refresh_every=20,
    noise_scale=0.01,
    exponent=EXPONENT
):
    rng = np.random.default_rng()

    running_sum = {s: np.zeros(len(landmarks), dtype=np.float64) for s in subsets}
    running_count = 0
    sample_history = []

    def pick_best_pair(candidates):
        kernels_for_candidate = []
        for pt in candidates:
            subset_kvals = {}
            for s in subsets:
                subset_kvals[s] = masked_gaussian_kernel(pt, landmarks, s, sigma_k)
            kernels_for_candidate.append(subset_kvals)

        idx_i, idx_j = np.triu_indices(len(candidates), k=1)
        best_score = None
        best_pair = None
        best_pair_kvals = None

        for i_c, j_c in zip(idx_i, idx_j):
            total_discrepancy = 0.0
            for s in subsets:
                old_sum_s = running_sum[s]
                new_sum_s = old_sum_s + kernels_for_candidate[i_c][s] + kernels_for_candidate[j_c][s]
                new_mean_s = new_sum_s / (running_count + 2)
                diff_s = new_mean_s - true_masked_exps[s]
                dist_arr = np.abs(diff_s)**exponent
                total_discrepancy += np.sum(dist_arr)

            if (best_score is None) or (total_discrepancy < best_score):
                best_score = total_discrepancy
                best_pair = (i_c, j_c)
                best_pair_kvals = (kernels_for_candidate[i_c], kernels_for_candidate[j_c])

        return best_pair, best_score, best_pair_kvals

    candidates = None

    for iteration in range(n_iters):
        if (iteration % refresh_every == 0) or (candidates is None):
            candidates = rng.normal(loc=0.0, scale=2.0, size=(candidate_size, 4))

        (i_best, j_best), _, (kvals_i, kvals_j) = pick_best_pair(candidates)
        x1 = candidates[i_best]
        x2 = candidates[j_best]

        # Noise that decays a bit
        noise_discount = math.exp(-0.05 * running_count)
        current_noise_scale = noise_scale * noise_discount
        x1_noisy = x1 + rng.normal(scale=current_noise_scale, size=4)
        x2_noisy = x2 + rng.normal(scale=current_noise_scale, size=4)

        kvals_x1_noisy = {}
        kvals_x2_noisy = {}
        for s in subsets:
            kvals_x1_noisy[s] = masked_gaussian_kernel(x1_noisy, landmarks, s, sigma_k)
            kvals_x2_noisy[s] = masked_gaussian_kernel(x2_noisy, landmarks, s, sigma_k)

        for s in subsets:
            running_sum[s] += kvals_x1_noisy[s] + kvals_x2_noisy[s]

        running_count += 2
        sample_history.append(x1_noisy)
        sample_history.append(x2_noisy)

        current_running_exps = {}
        for s in subsets:
            current_running_exps[s] = running_sum[s] / running_count

        yield (x1_noisy, x2_noisy, current_running_exps, sample_history)

#
# HELPER TO SKIP FRAMES
#
def skip_frames(generator, skip=10):
    """
    Only yield every N-th item from the generator.
    :param generator: an iterator (e.g. from masked_herding_generator)
    :param skip:      how many frames to skip between yields
    """
    for i, frame in enumerate(generator):
        # yield only if (i mod skip == 0)
        if i % skip == 0:
            yield frame

def debug_herding_4d_masked():
    """
    Main function demonstrating masked herding in 4D
    and calling the animation from another file (masked_animation_4d).
    We'll skip frames so the animation updates less often.
    """
    rng = np.random.default_rng()

    mu_4d = np.zeros(4)
    cov_4d = np.array([
        [1.0,  0.8,  0.3,  0.3],
        [0.8,  1.0,  0.2,  0.1],
        [0.3,  0.2,  1.0, -0.4],
        [0.3,  0.1, -0.4,  1.0]
    ])

    m = NUM_LANDMARKS
    landmarks_4d = rng.uniform(-3.5, 3.5, size=(m, 4))

    subsets = SUBSETS
    sigma_k = SIGMA_K

    # Compute "true" expectations for each subset
    true_masked_exps = {}
    for s in subsets:
        true_masked_exps[s] = analytical_landmark_expectation_masked(
            mu_4d, cov_4d, landmarks_4d, s, sigma_k
        )

    # Create a "full" herding generator
    full_gen = masked_herding_generator(
        mu_4d=mu_4d,
        cov_4d=cov_4d,
        subsets=subsets,
        landmarks=landmarks_4d,
        true_masked_exps=true_masked_exps,
        sigma_k=sigma_k,
        n_iters=NUM_ITER,
        candidate_size=NUM_CANDIDATES,
        refresh_every=25,
        noise_scale=NOISE_SCALE,
        exponent=1.2
    )

    # Wrap the generator to skip frames (update the animation less often)
    # e.g. skip=10 means we only update once every 10 herding steps
    thinned_gen = skip_frames(full_gen, skip=10)

    from herding.animation.maskedanimation4d import masked_animation_4d
    anim = masked_animation_4d(
        subsets=subsets,
        landmarks=landmarks_4d,
        true_masked_exps=true_masked_exps,
        sigma_k=sigma_k,
        gen=thinned_gen,  # pass the SKIPPED-FRAMES generator here
        cov_4d=cov_4d,
        scatter_pairs=[(0,1), (2,3), (1,3)],
        max_frames=300
    )

if __name__ == "__main__":
    debug_herding_4d_masked()
