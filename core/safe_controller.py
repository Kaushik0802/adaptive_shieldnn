# core/safe_controller.py

import numpy as np
from core.shield_model import ShieldModel
from core.dynamics import f_x, g_x
from core.barrier import h, dh_dx, alpha

class SafeController:
    """
    Applies closed-form safety override using Control Barrier Function (CBF) logic:
        u_safe = u_nom + λ · g(x)^T ∇h(x)
    """

    def __init__(self, shield_model: ShieldModel):
        self.shield_model = shield_model

    def compute_lambda(self, x: np.ndarray, u_nom: float, delta: float) -> float:
        """
        Computes the scalar λ that minimally corrects u_nom to satisfy the CBF constraint.

        Args:
            x (np.ndarray): State vector
            u_nom (float): Nominal control input
            delta (float): Adaptive margin δ(x)

        Returns:
            float: Scalar lambda correction
        """
        grad_h = dh_dx(x)                          # shape: (state_dim,)
        fx = f_x(x)                                # shape: (state_dim,)
        gx = g_x(x)                                # shape: (state_dim, 1)
        gT_dh = grad_h @ gx                        # shape: scalar
        G_sq = (gT_dh ** 2).item() + 1e-6          # avoid divide-by-zero

        h_dot = grad_h @ (fx + gx.flatten() * u_nom)   # shape: scalar
        alpha_h = alpha(h(x))

        numerator = delta - h_dot - alpha_h
        lambda_val = numerator / G_sq
        return lambda_val

    def override(self, x: np.ndarray, u_nom: float) -> float:
        """
        Returns a corrected control that satisfies the CBF constraint.

        Args:
            x (np.ndarray): System state
            u_nom (float): Nominal control from policy

        Returns:
            float: Safe control u_safe
        """
        delta = float(self.shield_model(x[[2, 3]]))  # xi and beta only
        lambda_val = self.compute_lambda(x, u_nom, delta)

        grad_h = dh_dx(x)
        gx = g_x(x)
        correction = lambda_val * (gx.T @ grad_h.reshape(-1, 1)).item()

        u_safe = u_nom + correction
        return u_safe
