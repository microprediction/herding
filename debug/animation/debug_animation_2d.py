#!/usr/bin/env python3

"""
Example usage of landmark_animation_2d with ~25 random landmarks in 2D.

Call this script to see how we might invoke the function and
step through some mock data in 2D for demonstration purposes.
"""

import numpy as np

from herding.animation.landmarkanimation2d import landmark_animation_2d

def debug_animation_2d():
    # Reproducible random state (optional)
    rng = np.random.default_rng(seed=42)

    # 1) Generate ~25 random 2D landmarks
    m = 25
    landmarks = rng.uniform(-5, 5, size=(m, 2))

    # 2) "True" landmark expectations (one per landmark), random for demo
    landmark_expectations = rng.random(size=m)

    # 3) Kernel scale
    sigma_k = 1.0

    # 4) Define a mock generator that yields (x, running_landmark_expectation).
    #    In practice, you'll replace this with your real data generator.
    def mock_generator(num_iter=5):
        for i in range(num_iter):
            # For demonstration, let x be a random 2D point
            x = rng.normal(scale=0.5, size=2)

            # "running_landmark_expectation" might be your current approximate
            # estimate of the expectation (just random for now).
            running_exp = rng.random(size=m)
            yield (x, running_exp)

    gen = mock_generator(num_iter=10)

    # 5) Call the function (uncomment actual import and usage in real code)
    landmark_animation_2d(landmarks, landmark_expectations, sigma_k, gen)

    # For demonstration here, we'll just print a preview
    print("Landmarks (first 5 shown):\n", landmarks[:5])
    print("Landmark Expectations (first 5 shown):\n", landmark_expectations[:5])
    print("sigma_k:", sigma_k)
    print("\nMock generator output:")
    for idx, (x, run_exp) in enumerate(gen):
        if idx < 3:  # just print first few
            print(f"Iteration {idx}, x={x}, run_exp[:5]={run_exp[:5]}")
        else:
            break


if __name__ == "__main__":
    debug_animation_2d()
