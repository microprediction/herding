from herding.gaussiankernel.gaussiankernelmulti import gaussian_kernel_multi
from herding.gaussiankernel.gaussiankernel import gaussian_kernel

# tests/gaussiankernel/test_gaussian_kernel_against_multi.py

# Compare

import numpy as np
import pytest

# Assuming these imports refer to your own modules/functions:
# from herding.gaussiankernel.gaussiankernelmulti import gaussian_kernel_multi
# from herding.gaussiankernel.gaussiankernel import gaussian_kernel


@pytest.mark.parametrize("d", [1, 2, 5])
@pytest.mark.parametrize("m", [1, 3, 5])
@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_gaussian_kernel_multi(d, m, sigma):
    """
    Compare the single-landmark gaussian_kernel(x, y, sigma)
    against the multi-landmark version gaussian_kernel_multi(x, Y, sigma).
    We generate random x, random Y (m landmarks), and ensure that
    single calls match the vectorized result.
    """
    rng = np.random.default_rng(seed=42)

    # Random x of dimension d
    x = rng.normal(size=d)

    # Generate m landmarks Y (shape: (m, d))
    Y = rng.normal(size=(m, d))

    # Compute multi-landmark results: shape (m,)
    vals_multi = gaussian_kernel_multi(x, Y, sigma)

    # Compare to single-landmark calls:
    for i in range(m):
        y_i = Y[i]
        val_single = gaussian_kernel(x, y_i, sigma)
        diff = abs(vals_multi[i] - val_single)
        print(
            f"d={d}, m={m}, sigma={sigma}, landmark={i} -> "
            f"multi={vals_multi[i]:.6f}, single={val_single:.6f}, diff={diff:.2e}"
        )
        assert diff < 1e-12, (
            f"Mismatch between single vs. multi for d={d}, m={m}, sigma={sigma}, landmark={i}"
        )
