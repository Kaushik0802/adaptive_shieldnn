# core/certifier.py

import os
import pickle
import numpy as np
from scipy.interpolate import RegularGridInterpolator

class DeltaCertifier:
    """
    Loads certified δ_min(x) values and provides fast δ clamping via interpolation.
    """

    def __init__(self, cert_paths=None):
        """
        Args:
            cert_paths (list[str]): List of pickle files, e.g.,
                ['certs/deriv1_certs.p', 'certs/deriv2_certs.p', 'certs/deriv3_certs.p']
        """
        if cert_paths is None:
            cert_paths = [
                "certs/deriv1_certs.p",
                "certs/deriv2_certs.p",
                "certs/deriv3_certs.p",
            ]

        self.interpolators = []
        for path in cert_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Certification file not found: {path}")
            with open(path, "rb") as f:
                data = pickle.load(f)

                # Expecting keys: "xi", "beta", "delta"
                xi = data["xi"]
                beta = data["beta"]
                delta_min = data["delta"]

                interp = RegularGridInterpolator(
                    (xi, beta), delta_min, bounds_error=False, fill_value=None
                )
                self.interpolators.append(interp)

    def get_certified_min(self, xi: float, beta: float) -> float:
        """
        Returns the maximum certified δ_min(x) across all levels.

        Args:
            xi (float)
            beta (float)

        Returns:
            float: certified lower bound δ_min(x)
        """
        point = np.array([[xi, beta]])
        values = [interp(point)[0] for interp in self.interpolators]
        return max(values)

    def clamp(self, delta: float, xi: float, beta: float) -> float:
        """
        Enforces δ ≥ δ_min(x)

        Args:
            delta (float): proposed δ(x)
            xi (float)
            beta (float)

        Returns:
            float: clamped δ(x)
        """
        delta_min = self.get_certified_min(xi, beta)
        return max(delta, delta_min)
