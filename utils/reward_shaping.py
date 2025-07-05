# utils/reward_shaping.py

import numpy as np

def compute_reward(x, delta, delta_prev, cbf_residual, goal_radius=1.5):
    """
    Constructs a shaped reward for adaptive margin training.

    Args:
        x (np.ndarray): current state
        delta (float): current δ(x)
        delta_prev (float): previous δ(x)
        cbf_residual (float): ḣ(x, u_nom) + α(h(x)) - δ(x)
        goal_radius (float): threshold for reaching goal

    Returns:
        float: reward signal
    """
    r = x[2]

    # --- Reward components ---
    reached_goal = r <= goal_radius
    cbf_violation_penalty = 10.0 * max(0, -cbf_residual)  # only if constraint is violated
    delta_penalty = 0.1 * delta                           # discourage overly conservative δ(x)
    delta_jump_penalty = 2.0 * abs(delta - delta_prev)    # smoothness penalty
    goal_reward = 100.0 if reached_goal else 0.0

    # --- Final reward ---
    reward = goal_reward - cbf_violation_penalty - delta_penalty - delta_jump_penalty

    return reward
