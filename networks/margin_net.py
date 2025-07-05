# networks/margin_net.py

import torch
import torch.nn as nn

class MarginNet(nn.Module):
    """
    Neural network that learns adaptive safety margin δ(x) from full KBM state.
    Uses deeper MLP with LayerNorm and Softplus output.
    """

    def __init__(self, input_dim=6, hidden_dim=128):
        super(MarginNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # ensures δ(x) > 0 and smooth
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, 6)
        Returns:
            Tensor: δ(x), shape (batch_size, 1)
        """
        return self.net(x)
