# utils/visualization.py

import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_delta_surface(policy, xi_range=(-np.pi, np.pi), beta_range=(-1.0, 1.0), resolution=100):
    """
    Plots the learned δ(x) over (xi, beta) grid.
    """
    xi_vals = np.linspace(*xi_range, resolution)
    beta_vals = np.linspace(*beta_range, resolution)
    Z = np.zeros((resolution, resolution))

    for i, xi in enumerate(xi_vals):
        for j, beta in enumerate(beta_vals):
            x_input = np.array([[0, 0, 1, xi, 2.0, beta]])  # dummy x, y, v = 2.0
            x_tensor = torch.tensor(x_input, dtype=torch.float32)
            with torch.no_grad():
                Z[j, i] = policy(x_tensor).item()

    plt.figure(figsize=(6, 5))
    plt.contourf(xi_vals, beta_vals, Z, levels=50, cmap="viridis")
    plt.colorbar(label="δ(x)")
    plt.xlabel("xi (radians)")
    plt.ylabel("beta")
    plt.title("Learned Adaptive Margin δ(x)")
    plt.tight_layout()
    plt.show()

def plot_lambda_over_time(lambda_vals):
    """
    Plots λ(x) correction over a rollout.
    """
    plt.figure()
    plt.plot(lambda_vals, label="λ correction")
    plt.xlabel("Time step")
    plt.ylabel("λ")
    plt.title("CBF Override Magnitude Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_trajectory(xy_coords):
    """
    Plots x-y trajectory of the agent.
    """
    xy = np.array(xy_coords)
    plt.figure()
    plt.plot(xy[:, 0], xy[:, 1], marker="o", markersize=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Agent Trajectory in 2D Space")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
