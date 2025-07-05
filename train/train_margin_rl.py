# train/train_margin_rl.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from env.kbm_env import KBMEnv
from networks.margin_net import MarginNet
from core.certifier import DeltaCertifier

from collections import deque

# --- Hyperparameters ---
epochs = 500
rollout_steps = 100
gamma = 0.99
lr = 3e-4
clip_eps = 0.2
train_iters = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Initialize Environment ---
env = KBMEnv()
state_dim = env.observation_space.shape[0]

# --- Networks and Certifier ---
policy = MarginNet(input_dim=state_dim).to(device)
optimizer = optim.Adam(policy.parameters(), lr=lr)
certifier = DeltaCertifier()

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32).to(device)

for epoch in range(epochs):
    states, actions, rewards, log_probs = [], [], [], []
    obs = env.reset()
    for step in range(rollout_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        delta = policy(obs_tensor)
        delta_clamped = certifier.clamp(delta.item(), xi=obs[3], beta=obs[5])

        # Step through env using δ(x) to influence override
        next_obs, reward, done, info = env.step(delta_clamped)

        # Store rollout
        states.append(obs_tensor)
        actions.append(torch.tensor([delta_clamped]))
        rewards.append(reward)

        obs = next_obs
        if done:
            break

    # Compute returns and PPO-style update
    returns = compute_returns(rewards, gamma)

    for _ in range(train_iters):
        for s, a, R in zip(states, actions, returns):
            pred = policy(s)
            loss = (pred - R) ** 2  # simple policy gradient baseline
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"[Epoch {epoch}] Total reward: {sum(rewards):.2f}")

