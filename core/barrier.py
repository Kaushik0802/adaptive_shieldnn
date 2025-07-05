# core/barrier.py

import numpy as np

# ---- Parameters (from Yasser's setup) ----
r_bar = 2.0       # Minimum distance from obstacle
sigma = 0.4       # Barrier shaping constant
lr = 2.0          # Rear-axle length
v_max = 5.0       # Max velocity for alpha scaling

def h(x: np.ndarray) -> float:
    """
    Barrier function h(x), should be >= 0 to remain safe.
    Args:
        x: state [x, y, r, xi, v, beta]
    Returns:
        h(x): scalar
    """
    r = x[2]
    xi = x[3]
    return (sigma * np.cos(xi / 2) + 1 - sigma) / r_bar - 1 / r

def dh_dx(x: np.ndarray) -> np.ndarray:
    """
    Gradient of the barrier function ∇h(x)
    Args:
        x: state [x, y, r, xi, v, beta]
    Returns:
        ∇h(x): np.array of shape (6,)
    """
    r = x[2]
    xi = x[3]

    # Partial derivatives
    dh_dr = 1 / (r ** 2)
    dh_dxi = -sigma * np.sin(xi / 2) / (2 * r_bar)

    grad = np.zeros_like(x)
    grad[2] = dh_dr     # ∂h/∂r
    grad[3] = dh_dxi    # ∂h/∂xi
    return grad

def alpha(h_val: float) -> float:
    """
    Class K function for CBF relaxation. Typically α(h) = k*h or tanh-based.
    Args:
        h_val: scalar h(x)
    Returns:
        α(h): scalar
    """
    # Conservative linear relaxation (Yasser used 3*vmax*sigma / (2*rbar*lr))
    gain = 3 * v_max * sigma / (2 * r_bar * lr)
    return gain * h_val
