import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import Delaunay

# 3 x 1 plot with  landmark surface, sample landmark surface, and all discrepancies

def landmark_animation_2d(landmarks, landmark_expectations, sigma_k, gen):
    """
    Visualization of a 2D set of landmarks and their associated "true" vs. "running"
    expectations, plus convergence curves.

    We create three subplots:
      - Left:  3D surface ("tent") using the 2D (x,y) locations of landmarks
               and their "true" expectation as the z-value.
      - Middle: 3D surface using the same landmark (x,y) but updated "running"
                expectations at each iteration of the generator 'gen'.
      - Right: 2D plot showing separate convergence curves for each landmark
               (the difference between true and running expectation) over iterations.

    :param landmarks:           shape (m, 2). The 2D coords of each landmark
    :param landmark_expectations: shape (m,). The "true" expectation for each landmark
    :param sigma_k:             kernel scale (unused directly in plotting, but shown if you want)
    :param gen:                 a generator that yields tuples (x, running_landmark_expectation)
                                where
                                    x is some point visited (shape (2,)),
                                    running_landmark_expectation is an array (m,)
                                    with the current approximate expectation per landmark.
    """
    landmarks = np.asarray(landmarks)
    landmark_expectations = np.asarray(landmark_expectations)
    m = len(landmark_expectations)
    if landmarks.shape != (m, 2):
        raise ValueError(f"landmarks must be shape (m,2). Got {landmarks.shape} but m={m}.")

    # --- Set up the figure and subplots ---
    fig = plt.figure(figsize=(12, 4))

    ax_left = fig.add_subplot(1, 3, 1, projection='3d')
    ax_middle = fig.add_subplot(1, 3, 2, projection='3d')
    ax_right = fig.add_subplot(1, 3, 3)

    fig.suptitle(f"Landmark Animation 2D (sigma_k = {sigma_k})")

    # --- Prepare a triangulation for the "tent" surfaces ---
    tri = Delaunay(landmarks)  # For plotting a surface from scattered 2D points

    # --- Left Plot: "True" Landmark Expectation ---
    surf_left = ax_left.plot_trisurf(
        landmarks[:, 0], landmarks[:, 1], landmark_expectations,
        triangles=tri.simplices, cmap='viridis', alpha=0.8
    )
    ax_left.set_title("True Landmark Expectations")
    ax_left.set_xlabel("X")
    ax_left.set_ylabel("Y")
    ax_left.set_zlabel("Z")

    # We'll set a nice view angle
    ax_left.view_init(elev=35, azim=-60)

    # --- Middle Plot: "Running" Landmark Expectation ---
    # Initialize with zero (or the first iteration's data) for demonstration
    running_surf = ax_middle.plot_trisurf(
        landmarks[:, 0], landmarks[:, 1],
        np.zeros_like(landmark_expectations),  # placeholder
        triangles=tri.simplices, cmap='viridis', alpha=0.8
    )
    ax_middle.set_title("Running Landmark Expectations")
    ax_middle.set_xlabel("X")
    ax_middle.set_ylabel("Y")
    ax_middle.set_zlabel("Z")
    ax_middle.view_init(elev=35, azim=-60)

    # --- Right Plot: Convergence curves (one line per landmark) ---
    ax_right.set_title("Convergence (Difference: True - Running)")
    ax_right.set_xlabel("Iteration")
    ax_right.set_ylabel("Difference")

    # We create one line per landmark
    lines = []
    for j in range(m):
        line_j, = ax_right.plot([], [], label=f"Landmark {j}")
        lines.append(line_j)
    #ax_right.legend()

    # We’ll store data over frames
    iteration_list = []       # x-axis in the right plot
    differences_list = [[] for _ in range(m)]  # one array per landmark

    def init():
        # No special initialization needed (the left 3D surface is already plotted)
        return []

    def update(frame):
        """
        Called each time we get a new (x, running_landmark_expectation) from gen.
        frame is expected to be (x, running_array).
        """
        i = len(iteration_list)  # current iteration index
        iteration_list.append(i)

        x_current, running_arr = frame  # x_current ~ shape (2,); running_arr ~ shape (m,)

        # 1) Update the middle 3D surface to reflect the new "running" values
        ax_middle.clear()
        ax_middle.plot_trisurf(
            landmarks[:, 0], landmarks[:, 1],
            running_arr,  # shape (m,)
            triangles=tri.simplices, cmap='viridis', alpha=0.4
        )
        ax_middle.set_title("Running Landmark Expectations")
        ax_middle.set_xlabel("X")
        ax_middle.set_ylabel("Y")
        ax_middle.set_zlabel("Z")
        ax_middle.view_init(elev=35, azim=-60)

        # 2) Update the difference lines on the right
        diff = landmark_expectations - running_arr
        for j in range(m):
            differences_list[j].append(diff[j])
            lines[j].set_data(iteration_list, differences_list[j])

        # Adjust the x-limits (iteration)
        ax_right.set_xlim(0, max(1, len(iteration_list)-1))

        # Adjust y-limits to fit the min/max difference so far
        all_diffs = [val for diffs_j in differences_list for val in diffs_j]
        ax_right.set_ylim(min(all_diffs)*1.05, max(all_diffs)*1.05 if all_diffs else 1)

        return lines

    # Create the animation
    anim = FuncAnimation(
        fig, update, frames=gen, init_func=init,
        interval=1000, blit=False, repeat=False
    )

    plt.tight_layout()
    plt.show()
    return anim
