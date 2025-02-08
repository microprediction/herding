# animation_4d.py

import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm

def masked_animation_4d(
    subsets,
    landmarks,
    true_masked_exps,
    sigma_k,
    gen,               # a herding generator
    cov_4d,
    scatter_pairs=[(0,1), (2,3), (1,3)],
    max_frames=300
):
    """
    Show multiple 2D scatter plots (of selected coordinate pairs),
    plus a multi-line correlation plot, plus a "subset discrepancy" plot.
    """

    # 1) All coordinate pairs for correlation
    all_pairs = list(itertools.combinations(range(4), 2))  # e.g. (0,1), (0,2), (0,3), etc.

    # 2) Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f"4D Herding w/ Masked Kernels (sigma_k={sigma_k:.3f})")

    ax_scat_1 = axes[0,0]
    ax_scat_2 = axes[0,1]
    ax_corr   = axes[0,2]
    ax_scat_3 = axes[1,0]
    ax_disc   = axes[1,1]
    ax_dummy  = axes[1,2]
    ax_dummy.axis("off")  # optional blank spot

    # Titles for scatter
    ax_scat_1.set_title(f"Scatter coords {scatter_pairs[0]}")
    ax_scat_2.set_title(f"Scatter coords {scatter_pairs[1]}")
    ax_scat_3.set_title(f"Scatter coords {scatter_pairs[2]}")

    for ax in [ax_scat_1, ax_scat_2, ax_scat_3]:
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal', 'box')

    # 3) Correlation axis
    ax_corr.set_title("Running Correlations")
    ax_corr.set_xlabel("Iteration")
    ax_corr.set_ylabel("Corr")

    # We'll keep a line for each of the 6 possible pairs in 4D
    lines_corr = {}
    corr_data = {pair: [] for pair in all_pairs}
    iters_list = []

    # Color map: ensures each pair has the same color for “running” line + “true” line
    color_cycle = cm.get_cmap('tab10', len(all_pairs))

    # Compute "true" correlations
    diag = np.sqrt(np.diag(cov_4d))
    true_corrs = {}
    for i, pair in enumerate(all_pairs):
        color = color_cycle(i)
        (ln,) = ax_corr.plot([], [], label=f"{pair}", color=color)
        lines_corr[pair] = ln

        # dashed horizontal line for true correlation
        p, q = pair
        rho_ij = cov_4d[p, q] / (diag[p]*diag[q])
        true_corrs[pair] = rho_ij
        ax_corr.axhline(rho_ij, color=color, linestyle='--', alpha=0.4)

    ax_corr.legend(loc='upper right')

    # 4) Discrepancy axis
    ax_disc.set_title("Subset Discrepancy (Sum of |running - true| per subset)")
    ax_disc.set_xlabel("Iteration")
    ax_disc.set_ylabel("Discrepancy")

    lines_disc = {}
    disc_data = {s: [] for s in subsets}
    for s in subsets:
        (ln,) = ax_disc.plot([], [], label=f"s={s}")
        lines_disc[s] = ln
    ax_disc.legend(loc='upper right')

    # Helper to get correlations
    def get_correlations(samples_4d):
        arr = np.array(samples_4d)
        if arr.shape[0] < 2:
            return {pair: 0.0 for pair in all_pairs}
        c = np.corrcoef(arr.T)  # shape (4,4)
        return {pair: c[pair[0], pair[1]] for pair in all_pairs}

    # init func for FuncAnimation
    def init():
        return []

    # update func
    def update(frame):
        try:
            x1_4d, x2_4d, current_running_exps, sample_history = frame
        except ValueError:
            return []

        i = len(iters_list)
        iters_list.append(i)

        # Scatter plots
        arr = np.array(sample_history)
        for ax, pair in zip([ax_scat_1, ax_scat_2, ax_scat_3], scatter_pairs):
            ax.clear()
            ax.set_title(f"Scatter coords {pair}")
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_aspect('equal', 'box')

            # Plot the entire sample history in black
            ax.scatter(arr[:, pair[0]], arr[:, pair[1]], c='k', s=10)

            # Plot the landmarks in green
            ax.scatter(
                landmarks[:, pair[0]],
                landmarks[:, pair[1]],
                c='g',
                s=10,
                marker='o',
                label='landmarks'
            )

            # Plot the two newly chosen points in red
            ax.scatter([x1_4d[pair[0]]], [x1_4d[pair[1]]], c='r', s=40)
            ax.scatter([x2_4d[pair[0]]], [x2_4d[pair[1]]], c='r', s=40)

        # Correlations
        corrs = get_correlations(sample_history)
        for pair in all_pairs:
            corr_data[pair].append(corrs[pair])
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

        # Discrepancy
        for s in subsets:
            run_s = current_running_exps[s]
            true_s = true_masked_exps[s]
            disc_val = np.sum(np.abs(run_s - true_s))
            disc_data[s].append(disc_val)
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

        # Return line handles so animation updates them
        return list(lines_corr.values()) + list(lines_disc.values())

    anim = FuncAnimation(
        fig, update,
        frames=iter(gen),  # or e.g. itertools.islice(gen, max_frames)
        init_func=init,
        interval=5,
        blit=False,
        repeat=False
    )

    plt.tight_layout()
    plt.show()
    return anim
