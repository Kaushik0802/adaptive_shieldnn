# scripts/pretrain_certified_delta.py

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from networks.margin_net import MarginNet
import os

def load_certified_data(cert_paths):
    data = []
    for path in cert_paths:
        with open(path, "rb") as f:
            cert = pickle.load(f)
            xi = cert["xi"]
            beta = cert["beta"]
            delta = cert["delta"]
            XI, BETA = np.meshgrid(xi, beta, indexing='ij')
            data.append((XI.flatten(), BETA.flatten(), delta.flatten()))
    # Combine and take pointwise max δ across cert levels
    all_xi = data[0][0]
    all_beta = data[0][1]
    all_delta = np.maximum.reduce([d[2] for d in data])
    return all_xi, all_beta, all_delta

def build_dataset(xi_vals, beta_vals, delta_vals):
    inputs = []
    targets = []
    for xi, beta, delta in zip(xi_vals, beta_vals, delta_vals):
        x_input = [0, 0, 1, xi, 2.0, beta]  # fixed x, y, r, v
        inputs.append(x_input)
        targets.append(delta)
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

def main():
    cert_paths = [
        "certs/deriv1_certs.p",
        "certs/deriv2_certs.p",
        "certs/deriv3_certs.p",
    ]

    xi_vals, beta_vals, delta_vals = load_certified_data(cert_paths)
    X, y = build_dataset(xi_vals, beta_vals, delta_vals)

    model = MarginNet(input_dim=6)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(300):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0:
            print(f"[Epoch {epoch}] Loss: {loss.item():.6f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/delta_pretrained.pt")
    print("✅ Saved pretrained δ(x) model to checkpoints/delta_pretrained.pt")

if __name__ == "__main__":
    main()
