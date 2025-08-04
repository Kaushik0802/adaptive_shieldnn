import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import math

class AdaptiveMarginAgent(nn.Module):
    """
    RL Agent for adaptive safety margins that balances aggressive performance with safety.
    Uses efficient reward design to push vehicle to high performance while staying safe.
    GPU-optimized for faster training.
    """

    def __init__(self, state_dim=5, hidden_dim=64, learning_rate=3e-4, device='cpu'):
        super().__init__()

        self.device = device

        # Neural network architecture
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()  # Output in [0,1], scaled to margin range
        ).to(self.device)

        # Training parameters
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # Margin scaling parameters
        self.margin_min = 0.1   # Minimum conservative margin
        self.margin_max = 0.8   # Maximum aggressive margin
        self.margin_range = self.margin_max - self.margin_min

        # Performance tracking
        self.episode_rewards = []
        self.safety_violations = 0
        self.performance_metrics = {
            'avg_speed': 0,
            'track_completion': 0,
            'safety_margin': 0
        }

    def get_margin(self, state, training=True):
        """
        Get adaptive margin based on current state.

        Args:
            state: [xi, r, v, steer, throttle] - current vehicle state
            training: Whether to add exploration noise

        Returns:
            margin: Adaptive safety margin in [margin_min, margin_max]
        """
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            margin_raw = self.network(state_tensor).item()

            # Scale to margin range
            margin = self.margin_min + margin_raw * self.margin_range

            # Add exploration noise during training
            if training and random.random() < self.epsilon:
                noise = np.random.normal(0, 0.1)
                margin = np.clip(margin + noise, self.margin_min, self.margin_max)

            return margin

    def compute_reward(self, state, margin, next_state, safety_violation,
                      performance_metrics, step_reward):
        """
        Compute reward for margin agent that balances safety and performance.

        Args:
            state: Current state [xi, r, v, steer, throttle]
            margin: Current margin used
            next_state: Next state after action
            safety_violation: Whether safety was violated
            performance_metrics: Dict with speed, completion, etc.
            step_reward: Original environment reward

        Returns:
            reward: Computed reward for margin agent
        """
        reward = 0.0

        # 1. SAFETY REWARD (High penalty for violations)
        if safety_violation:
            reward -= 100.0  # Large penalty for safety violations
        else:
            reward += 10.0   # Reward for maintaining safety

        # 2. PERFORMANCE REWARD (Encourage aggressive but safe behavior)
        xi, r, v, steer, throttle = state

        # Reward for high speed when safe
        if not safety_violation and v > 15.0:  # m/s
            speed_reward = min((v - 15.0) / 10.0, 1.0) * 5.0
            reward += speed_reward

        # Reward for aggressive steering when safe
        if not safety_violation and abs(steer) > 0.3:
            steering_reward = min(abs(steer) / 0.5, 1.0) * 3.0
            reward += steering_reward

        # 3. MARGIN EFFICIENCY REWARD (Penalize overly conservative margins)
        optimal_margin = self._compute_optimal_margin(state)
        margin_efficiency = 1.0 - abs(margin - optimal_margin) / self.margin_range

        if margin_efficiency > 0.8:  # Good margin selection
            reward += 5.0
        elif margin_efficiency < 0.3:  # Poor margin selection
            reward -= 3.0

        # 4. PROGRESS REWARD (Encourage forward movement)
        if performance_metrics.get('track_progress', 0) > 0:
            progress_reward = performance_metrics['track_progress'] * 2.0
            reward += progress_reward

        # 5. CONSISTENCY REWARD (Penalize erratic margin changes)
        if hasattr(self, 'last_margin'):
            margin_change = abs(margin - self.last_margin)
            if margin_change < 0.1:  # Consistent margins
                reward += 1.0
            elif margin_change > 0.3:  # Erratic changes
                reward -= 2.0

        self.last_margin = margin

        # 6. ADAPTIVE REWARD (Based on environmental conditions)
        # More aggressive margins in open areas, conservative in tight spaces
        if r > 20.0 and abs(xi) < 0.5:  # Open road, straight ahead
            if margin < 0.4:  # Aggressive margin
                reward += 3.0
        elif r < 10.0 or abs(xi) > 1.0:  # Tight space or sharp turn
            if margin > 0.5:  # Conservative margin
                reward += 2.0

        return reward

    def _compute_optimal_margin(self, state):
        """
        Compute optimal margin based on state characteristics.
        This is a heuristic for reward computation.
        """
        xi, r, v, steer, throttle = state

        # Base margin on distance to obstacle
        distance_factor = max(0.1, min(0.8, 1.0 - (r / 50.0)))

        # Adjust based on relative angle
        angle_factor = 1.0 - abs(xi) / math.pi

        # Adjust based on speed
        speed_factor = max(0.3, min(0.8, v / 30.0))

        # Combine factors
        optimal = (distance_factor * 0.4 + angle_factor * 0.3 + speed_factor * 0.3)

        return self.margin_min + optimal * self.margin_range

    def store_experience(self, state, margin, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, margin, reward, next_state, done))

    def train_step(self):
        """Train the agent on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, margins, rewards, next_states, dones = zip(*batch)

        # Convert to tensors and move to device
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        margins = torch.tensor(margins, dtype=torch.float32).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Compute current Q-values
        current_q = self.network(states)

        # Compute target Q-values
        with torch.no_grad():
            next_q = self.network(next_states)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss and update
        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def update_performance_metrics(self, metrics):
        """Update performance tracking metrics."""
        self.performance_metrics.update(metrics)

    def save_model(self, path):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'performance_metrics': self.performance_metrics,
            'device': self.device
        }, path)

    def load_model(self, path):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.performance_metrics = checkpoint['performance_metrics']
        if 'device' in checkpoint:
            self.device = checkpoint['device']


class AdaptiveMarginTrainer:
    """
    Trainer class for the adaptive margin agent with efficient reward design.
    """

    def __init__(self, margin_agent, env, training_config):
        self.margin_agent = margin_agent
        self.env = env
        self.config = training_config

        # Training statistics
        self.episode_rewards = []
        self.safety_violations = []
        self.performance_history = []

    def train_episode(self):
        """Train for one episode."""
        state = self.env.reset()
        episode_reward = 0
        safety_violations = 0
        performance_data = []

        done = False
        step_count = 0

        while not done and step_count < self.config['max_steps']:
            # Get adaptive margin
            margin = self.margin_agent.get_margin(state, training=True)

            # Store performance data
            performance_data.append({
                'margin': margin,
                'speed': state[2],  # velocity
                'distance': state[1],  # distance to obstacle
                'angle': state[0]  # relative angle
            })

            # Take action in environment (this would be done by the main PPO agent)
            # For now, we simulate the environment step
            next_state, reward, done, info = self.env.step([0, 0])  # Placeholder action

            # Check for safety violations
            safety_violation = info.get('collision', False) or info.get('safety_violation', False)
            if safety_violation:
                safety_violations += 1

            # Compute margin agent reward
            margin_reward = self.margin_agent.compute_reward(
                state, margin, next_state, safety_violation,
                info.get('performance_metrics', {}), reward
            )

            # Store experience
            self.margin_agent.store_experience(state, margin, margin_reward, next_state, done)

            # Train margin agent
            if len(self.margin_agent.memory) >= self.margin_agent.batch_size:
                loss = self.margin_agent.train_step()

            episode_reward += margin_reward
            state = next_state
            step_count += 1

        # Update statistics
        self.episode_rewards.append(episode_reward)
        self.safety_violations.append(safety_violations)

        # Compute episode performance metrics
        avg_margin = np.mean([d['margin'] for d in performance_data])
        avg_speed = np.mean([d['speed'] for d in performance_data])

        self.performance_history.append({
            'avg_margin': avg_margin,
            'avg_speed': avg_speed,
            'safety_violations': safety_violations,
            'episode_reward': episode_reward
        })

        return episode_reward, safety_violations

    def train(self, num_episodes):
        """Train the margin agent for specified number of episodes."""
        print(f"Starting training for {num_episodes} episodes...")

        for episode in range(num_episodes):
            episode_reward, safety_violations = self.train_episode()

            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_violations = np.mean(self.safety_violations[-100:])
                avg_margin = np.mean([p['avg_margin'] for p in self.performance_history[-100:]])

                print(f"Episode {episode}: Avg Reward={avg_reward:.2f}, "
                      f"Avg Violations={avg_violations:.2f}, "
                      f"Avg Margin={avg_margin:.3f}, "
                      f"Epsilon={self.margin_agent.epsilon:.3f}")

        print("Training completed!")
        return self.performance_history


# Configuration for training
TRAINING_CONFIG = {
    'max_steps': 1000,
    'learning_rate': 3e-4,
    'batch_size': 64,
    'gamma': 0.99,
    'epsilon': 0.1,
    'epsilon_decay': 0.995,
    'epsilon_min': 0.01
}


if __name__ == "__main__":
    # Example usage
    margin_agent = AdaptiveMarginAgent()
    print("Adaptive Margin Agent created successfully!")
    print(f"Margin range: [{margin_agent.margin_min}, {margin_agent.margin_max}]")