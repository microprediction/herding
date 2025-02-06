import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import Delaunay

INTERVAL = 30  # Animation speed


def discrepancy_animation_2d(landmarks, landmark_expectations, sigma_k, gen, true_rho):
    """
    Extended 2D Herding Animation with a 3x2 grid of subplots:

      Top row (3D surfaces):
        (1,1) True surface of the landmarks
        (1,2) Running (herding) surface
        (1,3) Difference surface (true - running)

      Bottom row (2D plots):
        (2,1) Per-landmark convergence curves: (true - running) vs. iteration
        (2,2) Running correlation of sampled points
               plus a horizontal line at 'true_rho'
        (2,3) A scatter plot of all sampled points in 2D, updated each iteration.

    We omit the legend on the per-landmark lines to avoid clutter, but do include
    a small legend in the correlation plot for the running corr line + the constant line.

    :param landmarks:  shape (m, 2). Landmark positions in 2D.
    :param landmark_expectations: shape (m,). The "true" expectation for each landmark.
    :param sigma_k:    Kernel scale (not used directly in the plot, but shown in title).
    :param gen:        A generator yielding (x_current, running_landmark_expectation)
                       where x_current is (2,) and running_landmark_expectation is (m,).
    :param true_rho:   A scalar "true" correlation value to display in correlation plot.
    :return:           The FuncAnimation object (so Python keeps a reference).
    """

    # --- Validate shapes ---
    landmarks = np.asarray(landmarks)
    landmark_expectations = np.asarray(landmark_expectations)
    m = len(landmark_expectations)
    if landmarks.shape != (m, 2):
        raise ValueError(f"landmarks must be shape (m,2). Got {landmarks.shape} but m={m}.")

    # --- Figure and Subplots (3x2) ---
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f"2D Herding Animation (sigma_k={sigma_k:.3f})", fontsize=14)

    # Subplots indexing: ax[row, col], using "add_subplot(nrows, ncols, index)"
    ax_true = fig.add_subplot(2, 3, 1, projection='3d')
    ax_run = fig.add_subplot(2, 3, 2, projection='3d')
    ax_diff = fig.add_subplot(2, 3, 3, projection='3d')

    ax_conv = fig.add_subplot(2, 3, 4)  # per-landmark differences
    ax_rho  = fig.add_subplot(2, 3, 5)  # correlation
    ax_samples = fig.add_subplot(2, 3, 6)  # 2D scatter of samples

    # --- Titles & labels ---
    ax_true.set_title("True Surface")
    ax_true.set_xlabel("X")
    ax_true.set_ylabel("Y")

    ax_run.set_title("Running Surface")
    ax_run.set_xlabel("X")
    ax_run.set_ylabel("Y")

    ax_diff.set_title("Difference (True - Running)")
    ax_diff.set_xlabel("X")
    ax_diff.set_ylabel("Y")

    ax_conv.set_title("Per-landmark Convergence")
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("True - Running")

    ax_rho.set_title("Running Correlation")
    ax_rho.set_xlabel("Iteration")
    ax_rho.set_ylabel("Corr(x[:,0], x[:,1])")

    ax_samples.set_title("Sampled Points in 2D")
    ax_samples.set_xlabel("x0")
    ax_samples.set_ylabel("x1")

    # Prepare Delaunay triangulation for the 3D surfaces
    tri = Delaunay(landmarks)

    # --- 1) Plot the static "true" surface on ax_true ---
    ax_true.plot_trisurf(
        landmarks[:, 0], landmarks[:, 1],
        landmark_expectations,
        triangles=tri.simplices, cmap='viridis', alpha=0.8
    )
    ax_true.view_init(elev=35, azim=-60)

    # --- 2) Running surface: start with zeros ---
    run_surf = ax_run.plot_trisurf(
        landmarks[:, 0], landmarks[:, 1],
        np.zeros_like(landmark_expectations),
        triangles=tri.simplices, cmap='viridis', alpha=0.8
    )
    ax_run.view_init(elev=35, azim=-60)

    # --- 3) Difference surface: start with (true - 0) = true ---
    diff_surf = ax_diff.plot_trisurf(
        landmarks[:, 0], landmarks[:, 1],
        landmark_expectations,
        triangles=tri.simplices, cmap='coolwarm', alpha=0.8
    )
    ax_diff.view_init(elev=35, azim=-60)

    # --- 4) Per-landmark difference lines: no legend shown here
    lines = []
    for j in range(m):
        line_j, = ax_conv.plot([], [])
        lines.append(line_j)

    # We'll store iteration indices and differences
    iteration_list = []
    differences_list = [[] for _ in range(m)]

    # --- 5) Running correlation line, plus a horizontal line at true_rho ---
    (line_corr,) = ax_rho.plot([], [], 'b-', label="Running corr")
    # Add a dashed line at y = true_rho
    ax_rho.axhline(y=true_rho, color='red', linestyle='--', label=f"true_rho={true_rho:.3f}")
    ax_rho.legend(loc='upper right')

    # We'll store correlation history
    corr_history = []

    # --- 6) Sampled Points in 2D (scatter) ---
    # We'll keep a history of points; each iteration we re-plot
    sample_history = []

    def init():
        return []

    def update(frame):
        """
        Each frame = (x_current, running_landmark_expectation).
        """
        i = len(iteration_list)
        iteration_list.append(i)

        if len(frame) == 2:
            # Single-point case
            x_current, running_arr = frame
            x2_current = None  # No second point
        elif len(frame) == 3:
            # Two-point case
            x_current, x2_current, running_arr = frame
        else:
            raise ValueError("Unexpected number of elements in frame!")


        sample_history.append(x_current)
        if x2_current is not None:
            sample_history.append(x2_current)

        # --- (a) Running surface re-plot ---
        ax_run.clear()
        ax_run.set_title("Running Surface")
        ax_run.set_xlabel("X")
        ax_run.set_ylabel("Y")
        ax_run.view_init(elev=35, azim=-60)
        ax_run.plot_trisurf(
            landmarks[:, 0], landmarks[:, 1],
            running_arr,
            triangles=tri.simplices, cmap='viridis', alpha=0.8
        )
        # Dot at (x_current, z=0)
        ax_run.scatter([x_current[0]], [x_current[1]], [0],
                       c='k', marker='o', s=20)

        # --- (b) Difference surface ---
        ax_diff.clear()
        ax_diff.set_title("Difference (True - Running)")
        ax_diff.set_xlabel("X")
        ax_diff.set_ylabel("Y")
        ax_diff.view_init(elev=35, azim=-60)
        diff_arr = landmark_expectations - running_arr
        ax_diff.plot_trisurf(
            landmarks[:, 0], landmarks[:, 1],
            diff_arr,
            triangles=tri.simplices, cmap='coolwarm', alpha=0.8
        )

        # --- (c) Per-landmark differences ---
        for j in range(m):
            differences_list[j].append(diff_arr[j])
            lines[j].set_data(iteration_list, differences_list[j])

        # Adjust x-limits
        ax_conv.set_xlim(0, max(1, i))
        # Adjust y-limits
        all_diffs = [val for diffs_j in differences_list for val in diffs_j]
        if all_diffs:
            ymin, ymax = min(all_diffs), max(all_diffs)
            if ymin == ymax:
                ymin, ymax = ymin - 1, ymax + 1
            ax_conv.set_ylim(ymin - 0.1*abs(ymin), ymax + 0.1*abs(ymax))

        # --- (d) Running correlation ---
        if len(sample_history) > 1:
            arr = np.array(sample_history)
            corr_mat = np.corrcoef(arr.T)
            running_corr = corr_mat[0, 1]
        else:
            running_corr = 0.0
        corr_history.append(running_corr)
        line_corr.set_data(iteration_list, corr_history)

        ax_rho.set_xlim(0, max(1, i))
        if corr_history:
            cmin, cmax = min(corr_history + [true_rho]), max(corr_history + [true_rho])
            if cmin == cmax:
                cmin, cmax = cmin - 0.1, cmax + 0.1
            ax_rho.set_ylim(cmin - 0.1*abs(cmin), cmax + 0.1*abs(cmax))

        # --- (e) 2D scatter of all sampled points ---
        ax_samples.clear()
        ax_samples.set_title("Sampled Points in 2D")
        ax_samples.set_xlabel("x0")
        ax_samples.set_ylabel("x1")

        # Plot entire history in black
        hist_arr = np.array(sample_history)
        ax_samples.scatter(hist_arr[:, 0], hist_arr[:, 1], c='k', s=10)

        ax_samples.scatter(hist_arr[-2:, 0], hist_arr[-2:, 1], c='r', s=20)

        if x2_current is None:
            # Plot a single red dot when only one point is chosen
            ax_run.scatter([x_current[0]], [x_current[1]], [0], c='r', marker='o', s=40)
        else:
            # Draw a red line between the two selected points
            ax_run.plot(
                [x_current[0], x2_current[0]],
                [x_current[1], x2_current[1]],
                [0, 0], 'r-', linewidth=2.5
            )

            # Also mark the selected points explicitly as red dots to make them visible
            ax_run.scatter(
                [x_current[0], x2_current[0]],
                [x_current[1], x2_current[1]],
                [0, 0], c='r', marker='o', s=40
            )

        # Ensure landmarks remain green
        ax_samples.scatter(
            landmarks[:, 0], landmarks[:, 1],
            c='g', s=40, marker='o', label="Landmarks"
        )

        # Auto-scale or set manual limits if desired
        xall, yall = hist_arr[:,0], hist_arr[:,1]
        xmin, xmax = min(xall), max(xall)
        ymin, ymax = min(yall), max(yall)
        margin = 1.0
        ax_samples.set_xlim(xmin - margin, xmax + margin)
        ax_samples.set_ylim(ymin - margin, ymax + margin)

        return lines + [line_corr]

    anim = FuncAnimation(
        fig, update, frames=gen,
        init_func=init, interval=INTERVAL,
        blit=False, repeat=False
    )

    plt.tight_layout()
    plt.show()
    return anim
