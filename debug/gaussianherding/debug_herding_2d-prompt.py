#!/usr/bin/env python3

"""
    2d herding animation

    1. Generate m random landmarks in 2d
    2. Generate true_cov randomly
    3. Compute landmark_expectations
    4. Create a generator that does the following:
        - Samples a random point 2d point x with randn()
        - Adds to the list of herding_samples
        - Computes the gaussian kernel   ker(x,y) for all landmarks y
        - Increments a running empirical mean of the numerical landmark expectations called running_landmark_expectation
        - Yields  (x, running_landmark_expectation)
        - Proceeds to the next point but ...
        -      Rather than sample randomly,
        -         Simulate 5,000 possible candidate points x from the plane if we don't have a 'fresh' candidate list
        -         Compute  [ ker(x,y) for all landmarks y] for all candidates x if we don't have that saved already
        -         Use this to figure out which x to use next, given that we want running_landmark_expectation to converge towards landmark_expectation eventually
        -      Choose the 'best' x
        -      Increments a running empirical mean of the numerical landmark expectations called running_landmark_expectation
        -      Every now and again, create a 'fresh' candidate list 


"""

import numpy as np

from herding.animation.landmarkanimation2d import landmark_animation_2d
from randomcov.randomcovariancematrix import random_covariance_matrix
from herding.gaussiankernel.analyticallandmarkexpectationmulti import analytical_landmark_expectation_multi
from herding.gaussiankernel.gaussiankernelmulti import gaussian_kernel_multi

def debug_herding_2d():
    # Reproducible random state (optional)
    rng = np.random.default_rng(seed=42)

    true_cov = random_covariance_matrix(n=2)

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
