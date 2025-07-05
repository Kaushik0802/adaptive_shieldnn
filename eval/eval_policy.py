# eval/eval_policy.py

import torch
from env.kbm_env import KBMEnv
from networks.margin_net import MarginNet
from core.certifier import DeltaCertifier
from utils.visualization import (
    plot_delta_surface,
    plot_lambda_over_time,
    plot_trajectory,
)

import numpy as np
import os

def evaluate_policy(model_path="checkpoints/delta_final.pt", render=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model ---
    model = MarginNet(input_dim=6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Environment and Certifier ---
    env = KBMEnv()
    certifier = DeltaCertifier()

    obs = env.reset()
    done = False
    max_steps = 200

    lambda_vals = []
    delta_vals = []
    trajectory = [obs[:2]]
    delta_prev = 0.0

    for _ in range(max_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            delta = model(obs_tensor).item()

        delta_clamped = certifier.clamp(delta, xi=obs[3], beta=obs[5])
        u_safe = env.controller.override(obs, 0.0)
        lambda_vals.append(u_safe)
        delta_vals.append(delta_clamped)

        obs, _, done, _ = env.step(delta_clamped)
        trajectory.append(obs[:2])

        if done:
            break

    # --- Visualization ---
    plot_delta_surface(model)
    plot_lambda_over_time(lambda_vals)
    plot_trajectory(trajectory)

    if render:
        print("🧭 Final state:", obs)
        print("✅ Goal reached:", obs[2] <= env.goal_radius)
