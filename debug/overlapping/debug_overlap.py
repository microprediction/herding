#!/usr/bin/env python3

import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

NUM_ITER = 5000
NUM_CANDIDATES = 40

# ============================================================================
# 1) Masked (subset) RBF kernel
# ============================================================================
def masked_gaussian_kernel(x, landmarks, subset, sigma):
    """
    Compute RBF kernel between a single point x (in 4D) and an array of landmarks (shape (m,4)),
    but only using the coordinates in 'subset'.

    :param x:         shape (4,)
    :param landmarks: shape (m, 4)
    :param subset:    e.g. (0,1,2) or (0,2,3), etc.
    :param sigma:     scalar RBF scale
    :return:          shape (m,)
    """
    # Extract relevant coordinates
    x_sub = x[list(subset)]            # shape (len(subset),)
    L_sub = landmarks[:, list(subset)] # shape (m, len(subset))

    diff = L_sub - x_sub[None, :]  # shape (m, len(subset))
    dist_sq = np.sum(diff**2, axis=1)
    return np.exp(-0.5 * dist_sq / (sigma**2))


# ============================================================================
# 2) Analytical expectation for masked kernel
#    We'll assume X ~ Normal(μ, Σ) in 4D, then for each subset, compute
#    E[k(X, L_j)] = integral exp( -|| x_sub - L_sub_j ||^2 / (2σ^2) ) dP(x)
#    but only on subset coords. This is analogous to your existing code
#    "analytical_landmark_expectation_multi", except we only consider the subset.
# ============================================================================
def analytical_landmark_expectation_masked(mu, cov, landmarks, subset, sigma):
    """
    For a 4D Gaussian (mu, cov),
    compute E[ masked_gaussian_kernel(X, L_j, subset, sigma ) ] for j=1..m.

    This is basically a 3D integral if subset has size 3.

    We'll do it by noting that if X ~ Normal(μ, Σ) in R^4,
    then X_sub ~ Normal(μ_sub, Σ_sub). We just ignore the other coordinates.

    Then the expectation is basically the standard formula for the
    "Gaussian integral of a Gaussian kernel" in dimension len(subset).
    """
    # Sub-cov, sub-mu
    sub = list(subset)
    mu_sub = mu[sub]            # shape (len(sub),)
    cov_sub = cov[np.ix_(sub, sub)]  # shape (len(sub), len(sub))
    inv_Sigma_sub = np.linalg.inv(cov_sub)

    L_sub = landmarks[:, sub]   # shape (m, len(sub))
    d = len(sub)                # e.g. 3

    # We'll use the known formula for the integral of
    # exp( -||x - y||^2 / (2σ^2) ) w.r.t. x ~ Normal(μ_sub, cov_sub).
    # The result is:
    #
    #   ( (2π)^{d/2} |Σ_sub|^{1/2} / ( (2π)^{d/2} |Σ_sub|^{1/2} ) ) * ...
    #
    # More compact approach:
    # E[e^{-||X_sub - L_j||^2/(2σ^2)}]
    #  = det( A )^{-1/2 } * exp( -1/2 * (L_j - μ_sub)^T B (L_j - μ_sub) )
    #
    # with appropriate definitions of A, B. See standard "sum of Gaussians" integrals.
    #
    # If we let Σ_s = cov_sub + σ^2 I,
    # then E[e^{-(X_sub - L_sub_j)^2/(2σ^2)}] = |Σ_s|^{-1/2} * exp( -1/2 * M_j^T * Σ_s^{-1} * M_j ),
    # where M_j = (L_sub_j - μ_sub).
    #
    # We'll implement that directly:

    # Precompute Σ_s = cov_sub + sigma^2 I
    Sigma_s = cov_sub + (sigma**2) * np.eye(d)
    inv_Sigma_s = np.linalg.inv(Sigma_s)
    det_Sigma_s = np.linalg.det(Sigma_s)

    results = []
    for j in range(len(landmarks)):
        M_j = L_sub[j] - mu_sub  # shape (d,)

        # exponent
        exponent = -0.5 * M_j @ inv_Sigma_s @ M_j
        # prefactor
        prefactor = ( (2*math.pi)**(d/2) * math.sqrt(det_Sigma_s) )**(-1)

        val = prefactor * np.exp(exponent)
        results.append(val)

    return np.array(results)


# ============================================================================
# 3) A generator to do herding in 4D, picking 2 points each iteration
#    We'll keep a separate running sum for each subset, since each subset
#    has its own masked kernel. We want to minimize total discrepancy across all subsets.
# ============================================================================
def masked_herding_generator(
    mu_4d, cov_4d,
    subsets,           # list of tuples, e.g. [(0,1,2), (0,1,3), ...]
    landmarks,         # shape (m,4)
    true_masked_exps,  # dict: subset -> shape (m,) true expectations
    sigma_k=1.0,
    n_iters=2000,
    candidate_size=50,
    refresh_every=20,
    noise_scale=0.01,
    exponent=1.0
):
    """
    :param mu_4d:            shape (4,)
    :param cov_4d:           shape (4,4)
    :param subsets:          e.g. [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
    :param landmarks:        shape (m,4)
    :param true_masked_exps: dict with keys = subsets (as tuples), values = shape (m,)
    :param sigma_k:          RBF scale
    :param exponent:         exponent used in the discrepancy measure
    :yield:  a tuple:
             (
               (x1_4d, x2_4d),          # the two chosen points in 4D
               current_running_exps     # dict: subset -> shape (m,)
             )
    """
    rng = np.random.default_rng()

    # We'll keep a running sum for each subset.
    # Example: running_sum[ (0,1,2) ] = array of shape (m,)
    running_sum = {}
    for s in subsets:
        running_sum[s] = np.zeros(len(landmarks), dtype=np.float64)
    running_count = 0

    sample_history = []  # store previously chosen points in 4D

    def pick_best_pair(candidates):
        """
        Among all candidate pairs, choose the pair that MINIMIZES total discrepancy
        across all subsets.

        We'll define "discrepancy" for each subset as:
           sum( |(running_mean - true)|^exponent )  over the m landmarks
        but we re-compute the running_mean if we add 2 new points to the sample set.
        Then we sum across all subsets.
        """
        nonlocal running_sum, running_count

        # For each candidate in 4D, precompute all subset-kernel vectors
        # kernels_for_candidate[c_idx][s] = shape (m,)
        kernels_for_candidate = []
        for pt in candidates:
            subset_kvals = {}
            for s in subsets:
                kval = masked_gaussian_kernel(pt, landmarks, s, sigma_k)
                subset_kvals[s] = kval
            kernels_for_candidate.append(subset_kvals)

        # We need to try all pairs (i<j) in candidates
        idx_i, idx_j = np.triu_indices(len(candidates), k=1)

        best_score = None
        best_pair  = None
        best_pair_kvals = None  # so we can reuse the kernel vectors

        for i_c, j_c in zip(idx_i, idx_j):
            # This pair is (candidates[i_c], candidates[j_c])
            # If we add them, the new running_count = old_count + 2
            # The new sums = running_sum[s] + k_i[s] + k_j[s] for each subset s

            # Compute new means for each subset, then the discrepancy
            total_discrepancy = 0.0
            for s in subsets:
                old_sum_s = running_sum[s]
                new_sum_s = old_sum_s + kernels_for_candidate[i_c][s] + kernels_for_candidate[j_c][s]
                new_mean_s = new_sum_s / (running_count + 2)
                diff_s = new_mean_s - true_masked_exps[s]
                dist_arr = np.abs(diff_s) ** exponent
                total_discrepancy += np.sum(dist_arr)

            if (best_score is None) or (total_discrepancy < best_score):
                best_score = total_discrepancy
                best_pair  = (i_c, j_c)
                # We'll store the actual kernel vectors for later use
                best_pair_kvals = (kernels_for_candidate[i_c], kernels_for_candidate[j_c])

        return best_pair, best_score, best_pair_kvals

    # We do the usual loop
    candidates = None

    for iteration in range(n_iters):
        if (iteration % refresh_every == 0) or (candidates is None):
            # sample new candidate points in 4D
            candidates = rng.normal(loc=0.0, scale=2.0, size=(candidate_size, 4))

        (i_best, j_best), _, (kvals_i, kvals_j) = pick_best_pair(candidates)

        x1 = candidates[i_best]
        x2 = candidates[j_best]

        # optional noise
        noise_discount = math.exp(-0.0005 * running_count)
        current_noise_scale = noise_scale * noise_discount
        x1_noisy = x1 + rng.normal(scale=current_noise_scale, size=4)
        x2_noisy = x2 + rng.normal(scale=current_noise_scale, size=4)

        # Recompute the kernel vectors for the noisy versions
        # (We can do that from scratch or just reuse the old ones if noise is small.
        #  But let's do it precisely.)
        kvals_x1_noisy = {}
        kvals_x2_noisy = {}
        for s in subsets:
            kvals_x1_noisy[s] = masked_gaussian_kernel(x1_noisy, landmarks, s, sigma_k)
            kvals_x2_noisy[s] = masked_gaussian_kernel(x2_noisy, landmarks, s, sigma_k)

        # Update running sums
        for s in subsets:
            running_sum[s] += kvals_x1_noisy[s] + kvals_x2_noisy[s]

        running_count += 2

        sample_history.append(x1_noisy)
        sample_history.append(x2_noisy)

        # Current running_exps for each subset
        current_running_exps = {}
        for s in subsets:
            current_running_exps[s] = running_sum[s] / running_count

        yield (x1_noisy, x2_noisy, current_running_exps, sample_history)


# ============================================================================
# 4) An example animation function for 4D samples
#    We'll show multiple 2D scatter plots of different coordinate pairs,
#    plus a multi-line correlation plot, plus a "subset discrepancy" plot.
# ============================================================================
def masked_animation_4d(
    subsets,           # e.g. [(0,1,2), ...]
    landmarks,         # shape (m,4)
    true_masked_exps,  # dict
    sigma_k,
    gen,               # generator from above
    cov_4d,            # to show "true" correlation lines
    scatter_pairs=[(0,1), (2,3), (1,3)],
    max_frames=300
):
    """
    :param scatter_pairs: Which coordinate pairs to plot in 2D scatter.
                          You can choose e.g. [(0,1), (2,3), (1,3)] or more.
    :param max_frames:    Maximum frames to animate (since generator can be large).
    """

    # We'll also track correlations among all 6 pairs (since d=4 => 6 pairs).
    all_pairs = list(itertools.combinations(range(4), 2))  # (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)

    # Make a figure with a grid of subplots
    # Example: 2 rows x 3 cols
    #   row=1, col=1 => scatter of (0,1)
    #   row=1, col=2 => scatter of (2,3)
    #   row=1, col=3 => correlation lines
    #   row=2, col=1 => scatter of (1,3)  (or something)
    #   row=2, col=2 => discrepancy lines
    #   row=2, col=3 => we could do something else if desired
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("4D Herding w/ Masked Kernels (sigma_k=%.3f)" % sigma_k)

    ax_scat_1 = axes[0,0]
    ax_scat_2 = axes[0,1]
    ax_corr   = axes[0,2]
    ax_scat_3 = axes[1,0]
    ax_disc   = axes[1,1]
    ax_dummy  = axes[1,2]  # you can add another plot or leave it blank

    # Prepare the scatter axes
    ax_scat_1.set_title(f"Scatter coords {scatter_pairs[0]}")
    ax_scat_2.set_title(f"Scatter coords {scatter_pairs[1]}")
    ax_scat_3.set_title(f"Scatter coords {scatter_pairs[2]}")

    for ax in [ax_scat_1, ax_scat_2, ax_scat_3]:
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal', 'box')

    # Prepare correlation axis
    ax_corr.set_title("Running Correlations")
    ax_corr.set_xlabel("Iteration")
    ax_corr.set_ylabel("Corr")

    # We'll keep a line for each of the 6 possible pairs in 4D.
    lines_corr = {}
    for pair in all_pairs:
        (ln,) = ax_corr.plot([], [], label=f"{pair}")
        lines_corr[pair] = ln
    ax_corr.legend(loc='upper right')

    # We'll store correlation data over time
    corr_data = {pair: [] for pair in all_pairs}
    iters_list = []

    # The "true" correlations from cov_4d
    # We can also plot horizontal lines for each pair's "true" corr if desired
    true_corrs = {}
    diag = np.sqrt(np.diag(cov_4d))
    for pair in all_pairs:
        i, j = pair
        # correlation = cov[i,j] / sqrt(cov[i,i]*cov[j,j])
        rho_ij = cov_4d[i, j] / (diag[i]*diag[j])
        true_corrs[pair] = rho_ij
        ax_corr.axhline(rho_ij, color='red', linestyle='--', alpha=0.2)

    # Prepare discrepancy axis
    ax_disc.set_title("Subset Discrepancy (Sum of |running - true| per subset)")
    ax_disc.set_xlabel("Iteration")
    ax_disc.set_ylabel("Discrepancy")
    # We'll keep a line per subset
    lines_disc = {}
    for s in subsets:
        (ln,) = ax_disc.plot([], [], label=f"s={s}")
        lines_disc[s] = ln
    ax_disc.legend(loc='upper right')

    disc_data = {s: [] for s in subsets}

    # "Dummy" axis can be used to display some text or remain empty
    ax_dummy.set_title("You can place additional info here")
    ax_dummy.axis("off")

    # For convenience, define a function to compute correlation across pairs
    def get_correlations(samples_4d):
        # samples_4d: list or array of shape (n,4)
        arr = np.array(samples_4d)
        if arr.shape[0] < 2:
            # not enough points to have a correlation
            return {pair: 0.0 for pair in all_pairs}
        c = np.corrcoef(arr.T)  # shape (4,4)
        return {pair: c[pair[0], pair[1]] for pair in all_pairs}

    # We also need to update the scatter plots. Let's define
    # separate scatter handle for each of the 3 subplots
    scat_handles = []
    for _ in range(3):
        scat = None
        scat_handles.append(scat)

    # init function for FuncAnimation
    def init():
        return []

    # update function
    def update(frame_id):
        # frame is (x1_4d, x2_4d, current_running_exps, sample_history)
        try:
            x1_4d, x2_4d, current_running_exps, sample_history = frame_id
        except ValueError:
            return []

        i = len(iters_list)
        iters_list.append(i)

        # 1) Update sample scatter plots
        arr = np.array(sample_history)  # shape (n,4)

        # pick 3 pairs from scatter_pairs
        for ax, scat, pair in zip([ax_scat_1, ax_scat_2, ax_scat_3],
                                  scat_handles,
                                  scatter_pairs):
            ax.clear()
            ax.set_title(f"Scatter coords {pair}")
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_aspect('equal', 'box')
            ax.scatter(arr[:, pair[0]], arr[:, pair[1]], c='k', s=10)
            ax.scatter([x1_4d[pair[0]]], [x1_4d[pair[1]]], c='r', s=40)
            ax.scatter([x2_4d[pair[0]]], [x2_4d[pair[1]]], c='r', s=40)

        # 2) Correlation lines
        corrs = get_correlations(sample_history)  # dict: pair -> float
        for pair in all_pairs:
            corr_data[pair].append(corrs[pair])
            # update line data
            xvals = np.arange(len(corr_data[pair]))
            yvals = corr_data[pair]
            lines_corr[pair].set_data(xvals, yvals)

        ax_corr.set_xlim(0, max(10, len(iters_list)))
        all_corr_values = [v for pair in all_pairs for v in corr_data[pair]]
        if all_corr_values:
            cmin, cmax = min(all_corr_values), max(all_corr_values)
            if cmin == cmax:
                cmin, cmax = cmin - 1e-3, cmax + 1e-3
            ax_corr.set_ylim(cmin - 0.1*abs(cmin), cmax + 0.1*abs(cmax))

        # 3) Subset discrepancy lines
        # For each subset s, compute sum(|running - true|).
        for s in subsets:
            run_s = current_running_exps[s]
            true_s = true_masked_exps[s]
            disc = np.sum(np.abs(run_s - true_s))
            disc_data[s].append(disc)
            xvals = np.arange(len(disc_data[s]))
            yvals = disc_data[s]
            lines_disc[s].set_data(xvals, yvals)

        ax_disc.set_xlim(0, max(10, len(iters_list)))
        all_disc_values = [v for s in subsets for v in disc_data[s]]
        if all_disc_values:
            dmin, dmax = min(all_disc_values), max(all_disc_values)
            if dmin == dmax:
                dmin, dmax = dmin - 1e-3, dmax + 1e-3
            ax_disc.set_ylim(dmin - 0.1*abs(dmin), dmax + 0.1*abs(dmax))

        return list(lines_corr.values()) + list(lines_disc.values())

    anim = FuncAnimation(
        fig, update, frames=iter(gen),
        init_func=init, interval=50, blit=False, repeat=False
    )

    # If you only want to animate up to max_frames, you might do:
    # anim = FuncAnimation(
    #     fig, update, frames=itertools.islice(gen, max_frames),
    #     init_func=init, interval=50, blit=False, repeat=False
    # )

    plt.tight_layout()
    plt.show()
    return anim


# ============================================================================
# 5) Main debug function
# ============================================================================
def debug_herding_4d_masked():
    rng = np.random.default_rng()

    # A) Setup correlation structure in 4D
    mu_4d = np.zeros(4)
    cov_4d = np.array([
        [1.0, -0.8, 0.3,  0.0],
        [-0.8, 1.0, 0.2,  0.1],
        [0.3,  0.2, 1.0,  -0.4],
        [0.0,  0.1, -0.4, 1.0]
    ])
    # Make sure it's positive semidefinite. In practice you might use
    # a random correlation matrix generator or ensure SPD. For the sake
    # of this example, let's assume it's SPD enough. If not, you'd
    # do a small shift or use e.g. nearest SPD matrix methods.

    # B) Generate random 4D landmarks
    m = 40
    landmarks_4d = rng.uniform(-3, 3, size=(m, 4))

    # C) Define subsets of size 3 out of 4
    subsets = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]

    # D) Compute "true" masked kernel expectations for each subset
    sigma_k = 1.0
    true_masked_exps = {}
    for s in subsets:
        true_masked_exps[s] = analytical_landmark_expectation_masked(
            mu_4d, cov_4d, landmarks_4d, s, sigma_k
        )

    # E) Create the herding generator
    gen = masked_herding_generator(
        mu_4d, cov_4d,
        subsets, landmarks_4d, true_masked_exps,
        sigma_k=sigma_k,
        n_iters=NUM_ITER,
        candidate_size=NUM_CANDIDATES,
        refresh_every=25,
        noise_scale=0.05,
        exponent=1.2
    )

    # F) Animate
    anim = masked_animation_4d(
        subsets, landmarks_4d, true_masked_exps,
        sigma_k, gen, cov_4d,
        scatter_pairs=[(0,1), (2,3), (1,3)],
        max_frames=300
    )


if __name__ == "__main__":
    debug_herding_4d_masked()
