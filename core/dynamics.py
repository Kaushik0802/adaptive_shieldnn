# core/dynamics.py

import numpy as np

# Vehicle parameters (match barrier.py)
lr = 2.0
delta_f_max = np.pi / 4  # 45 deg steering
beta_max = np.arctan(0.5 * np.tan(delta_f_max))

def f_x(x: np.ndarray) -> np.ndarray:
    """
    Drift dynamics f(x) for KBM in polar coordinates.
    Args:
        x: state [x, y, r, xi, v, beta]
    Returns:
        np.array: shape (6,)
    """
    _, _, r, xi, v, beta = x

    dxdt = np.zeros_like(x)
    dxdt[2] = -v * np.cos(xi - beta)    # dr/dt
    dxdt[3] = (v / r) * np.sin(xi - beta) - (v / lr) * np.sin(beta)  # dxi/dt
    dxdt[4] = 0  # dv/dt (assumed constant in this model)
    dxdt[5] = 0  # dbeta/dt (will be overridden externally)

    return dxdt

def g_x(x: np.ndarray) -> np.ndarray:
    """
    Control matrix g(x) → how u affects ẋ
    Only affects betȧ (6th state)
    Returns:
        np.array: shape (6, 1)
    """
    g = np.zeros((6, 1))
    g[5] = 1  # dbeta/dt = u
    return g
