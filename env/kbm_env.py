# env/kbm_env.py

import numpy as np
import gym
from gym import spaces

from core.safe_controller import SafeController
from core.shield_model import ShieldModel
from core.certifier import DeltaCertifier
from core.dynamics import f_x, g_x

class KBMEnv(gym.Env):
    """
    Gym-compatible environment for the Kinematic Bicycle Model (KBM)
    with CBF-based safe control filtering.
    """

    def __init__(self):
        super().__init__()

        # State: [x, y, r, xi, v, beta]
        self.state_dim = 6
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        # Action: proposed δ(x) (scalar)
        self.action_space = spaces.Box(low=0.0, high=2.0, shape=(1,), dtype=np.float32)

        # Components
        self.shield = ShieldModel("3.h5")
        self.controller = SafeController(self.shield)
        self.certifier = DeltaCertifier()

        self.dt = 0.05
        self.max_steps = 200
        self.step_count = 0
        self.goal_radius = 1.5

        self.reset()

    def reset(self):
        # Reset initial state
        x = np.random.uniform(2.5, 4.0)
        y = np.random.uniform(-1.5, 1.5)
        r = np.sqrt(x ** 2 + y ** 2)
        xi = np.arctan2(y, x)
        v = 2.0  # constant
        beta = 0.0

        self.state = np.array([x, y, r, xi, v, beta])
        self.step_count = 0
        return self.state.copy()

    def step(self, delta_proposed):
        self.step_count += 1

        # Clamp δ with certified min
        xi = self.state[3]
        beta = self.state[5]
        delta = self.certifier.clamp(delta_proposed, xi, beta)

        # Use safe controller to get u_safe (steering rate)
        u_nom = 0.0  # default behavior: no beta change
        u_safe = self.controller.override(self.state, u_nom)

        # Apply dynamics
        dx = f_x(self.state) + g_x(self.state).flatten() * u_safe
        self.state += self.dt * dx

        # Update r and xi
        x, y = self.state[0], self.state[1]
        self.state[2] = np.sqrt(x ** 2 + y ** 2)  # update r
        self.state[3] = np.arctan2(y, x) - self.state[5]  # update xi

        # Compute reward (stub for now)
        reward = -1.0  # shaped later

        # Check terminal conditions
        done = False
        if self.state[2] <= self.goal_radius:
            done = True  # reached goal
            reward += 100.0
        if self.step_count >= self.max_steps:
            done = True

        return self.state.copy(), reward, done, {}
