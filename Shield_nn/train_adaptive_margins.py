#!/usr/bin/env python3
"""
Training script for ShieldNN with Adaptive Margins.
Integrates PPO agent with adaptive margin RL agent for enhanced safety and performance.
GPU-optimized for faster training.
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import json
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from adaptive_margin_agent import AdaptiveMarginAgent, AdaptiveMarginTrainer
from CarlaEnv.enhanced_wrappers import EnhancedSafetyFilter, AdaptiveMarginEnvironment
from ppo import PPO
from vae_common import create_encode_state_fn, load_vae
from reward_functions import reward_functions
from utils import VideoRecorder, compute_gae

# Import CARLA environment
from CarlaEnv.carla_lap_env import CarlaLapEnv as CarlaEnv


class DualAgentTrainer:
    """
    Trainer for dual RL agents: PPO controller + Adaptive Margin agent.
    Balances aggressive performance with safety through adaptive margins.
    GPU-optimized for faster training.
    """

    def __init__(self, config):
        self.config = config

        # Check for GPU availability
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = torch.device('cpu')
            print("⚠️  No GPU detected - Using CPU training")

        # Initialize agents
        self.margin_agent = AdaptiveMarginAgent().to(self.device)
        self.ppo_agent = None  # Will be initialized with environment

        # Training statistics
        self.training_stats = {
            'ppo_rewards': [],
            'margin_rewards': [],
            'safety_violations': [],
            'performance_metrics': [],
            'margin_history': []
        }

        # Create environment
        self.env = self._create_environment()

        # Initialize enhanced safety filter
        self.enhanced_filter = EnhancedSafetyFilter(
            onnx_models_path=config['onnx_models_path'],
            margin_agent_path=config.get('margin_agent_path')
        )

        # Wrap environment with adaptive margins
        self.adaptive_env = AdaptiveMarginEnvironment(self.env, self.enhanced_filter)

        # Initialize PPO agent
        self._initialize_ppo_agent()

    def _create_environment(self):
        """Create CARLA environment with safety filter."""
        return CarlaEnv(
            host=self.config.get('host', '127.0.0.1'),
            port=self.config.get('port', 2000),
            viewer_res=(1280, 720),  # Full resolution for GPU
            obs_res=(160, 80),  # Optimized resolution
            reward_fn=reward_functions[self.config['reward_fn']],
            encode_state_fn=create_encode_state_fn(self.config['vae_model']),
            synchronous=self.config.get('synchronous', True),
            fps=self.config.get('fps', 30),
            action_smoothing=self.config.get('action_smoothing', 0.9),
            start_carla=self.config.get('start_carla', True),
            apply_filter=True,  # Enable safety filtering
            obstacle=True,  # Enable obstacles for safety testing
            penalize_steer_diff=False,
            penalize_dist_obstacle=False,
            gaussian=False,
            track=self.config.get('track', 1),
            mode='train'
        )

    def _initialize_ppo_agent(self):
        """Initialize PPO agent with environment."""
        obs = self.adaptive_env.reset()
        input_shape = obs.shape

        self.ppo_agent = PPO(
            input_shape=input_shape,
            action_space=self.adaptive_env.action_space,
            learning_rate=self.config['learning_rate'],
            lr_decay=self.config['lr_decay'],
            discount_factor=self.config['discount_factor'],
            gae_lambda=self.config['gae_lambda'],
            ppo_epsilon=self.config['ppo_epsilon'],
            initial_std=self.config['initial_std'],
            value_scale=self.config['value_scale'],
            entropy_scale=self.config['entropy_scale'],
            horizon=self.config['horizon'],
            num_epochs=self.config['num_epochs'],
            batch_size=self.config['batch_size']
        )

    def train_episode(self, episode_num):
        """
        Train for one episode with both PPO and margin agents.

        Args:
            episode_num: Current episode number

        Returns:
            episode_stats: Dictionary with episode statistics
        """
        obs = self.adaptive_env.reset()
        episode_reward = 0
        margin_reward = 0
        safety_violations = 0
        episode_data = []
        step_count = 0

        # Move observation to device
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

        done = False
        while not done and step_count < self.config['max_steps']:
            # Get PPO action
            action, log_prob, value = self.ppo_agent.get_action(obs)

            # Convert action to numpy for environment
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()

            # Get current state for margin agent
            current_state = self.adaptive_env._get_current_state()

            # Get adaptive margin
            margin = self.enhanced_filter.get_adaptive_margin(current_state)

            # Store episode data
            episode_data.append({
                'state': current_state,
                'margin': margin,
                'action': action,
                'step': step_count
            })

            # Take step in environment
            next_obs, reward, done, info = self.adaptive_env.step(action)

            # Move next observation to device
            if isinstance(next_obs, np.ndarray):
                next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)

            # Check for safety violations
            collision = info.get('collision', False)
            if collision:
                safety_violations += 1

            # Store PPO experience
            self.ppo_agent.store_experience(obs, action, reward, next_obs, done, log_prob, value)

            # Compute margin agent reward
            margin_reward_step = self._compute_margin_reward(info)
            margin_reward += margin_reward_step

            episode_reward += reward
            obs = next_obs
            step_count += 1

        # Train PPO agent
        if len(self.ppo_agent.memory) >= self.ppo_agent.batch_size:
            ppo_loss = self.ppo_agent.train_step()

        # Train margin agent
        self._train_margin_agent(episode_data)

        # Compute episode statistics
        avg_margin = np.mean([d['margin'] for d in episode_data]) if episode_data else 0.0

        episode_stats = {
            'ppo_reward': episode_reward,
            'margin_reward': margin_reward,
            'safety_violations': safety_violations,
            'avg_margin': avg_margin,
            'steps': step_count,
            'collision_rate': safety_violations / max(step_count, 1)
        }

        return episode_stats

    def _compute_margin_reward(self, info):
        """Compute reward for margin agent based on episode info."""
        reward = 0.0

        # Safety reward
        if info.get('collision', False):
            reward -= 50.0
        else:
            reward += 5.0

        # Performance reward
        performance_summary = info.get('performance_summary', {})
        safety_rate = performance_summary.get('safety_rate', 1.0)
        filter_rate = performance_summary.get('filter_rate', 0.0)

        # Reward for high safety rate
        reward += safety_rate * 10.0

        # Penalty for excessive filtering (overly conservative)
        if filter_rate > 0.8:
            reward -= 5.0

        # Reward for optimal margin usage
        margin_used = info.get('margin_used', 0.5)
        if 0.2 <= margin_used <= 0.6:  # Good margin range
            reward += 3.0

        return reward

    def _train_margin_agent(self, episode_data):
        """Train margin agent using episode data."""
        if len(episode_data) < 10:  # Need minimum data
            return

        # Convert episode data to training format
        for i in range(len(episode_data) - 1):
            current_data = episode_data[i]
            next_data = episode_data[i + 1]

            state = current_data['state']
            margin = current_data['margin']
            reward = self._compute_margin_reward({
                'collision': current_data['collision'],
                'margin_used': margin,
                'performance_summary': {'safety_rate': 1.0 if not current_data['collision'] else 0.0}
            })
            next_state = next_data['state']
            done = i == len(episode_data) - 2

            # Store experience for margin agent
            self.margin_agent.store_experience(state, margin, reward, next_state, done)

        # Train margin agent
        if len(self.margin_agent.memory) >= self.margin_agent.batch_size:
            loss = self.margin_agent.train_step()

    def train(self, num_episodes):
        """
        Train both agents for specified number of episodes.

        Args:
            num_episodes: Number of episodes to train

        Returns:
            training_stats: Complete training statistics
        """
        print(f"Starting dual agent training for {num_episodes} episodes...")
        print(f"Device: {self.device}")
        print(f"PPO Learning Rate: {self.config['learning_rate']}")
        print(f"Margin Agent Epsilon: {self.margin_agent.epsilon}")

        start_time = time.time()

        for episode in range(num_episodes):
            episode_stats = self.train_episode(episode)

            # Store statistics
            self.training_stats['ppo_rewards'].append(episode_stats['ppo_reward'])
            self.training_stats['margin_rewards'].append(episode_stats['margin_reward'])
            self.training_stats['safety_violations'].append(episode_stats['safety_violations'])
            self.training_stats['margin_history'].append(episode_stats['avg_margin'])

            # Log progress
            if episode % 50 == 0:
                avg_ppo_reward = np.mean(self.training_stats['ppo_rewards'][-50:])
                avg_margin_reward = np.mean(self.training_stats['margin_rewards'][-50:])
                avg_violations = np.mean(self.training_stats['safety_violations'][-50:])
                avg_margin = np.mean(self.training_stats['margin_history'][-50:])

                elapsed_time = time.time() - start_time
                episodes_per_hour = (episode + 1) / (elapsed_time / 3600)

                print(f"Episode {episode}/{num_episodes}: "
                      f"PPO_Reward={avg_ppo_reward:.2f}, "
                      f"Margin_Reward={avg_margin_reward:.2f}, "
                      f"Violations={avg_violations:.2f}, "
                      f"Avg_Margin={avg_margin:.3f}, "
                      f"Epsilon={self.margin_agent.epsilon:.3f}, "
                      f"Episodes/Hour={episodes_per_hour:.1f}")

            # Save models periodically
            if episode % 500 == 0 and episode > 0:
                self.save_models(episode)

        print("Training completed!")
        self.save_models(num_episodes)
        return self.training_stats

    def save_models(self, episode_num):
        """Save trained models."""
        # Save PPO agent
        ppo_path = f"{self.config['model_save_path']}/ppo_agent_episode_{episode_num}.pth"
        self.ppo_agent.save_model(ppo_path)

        # Save margin agent
        margin_path = f"{self.config['model_save_path']}/margin_agent_episode_{episode_num}.pth"
        self.margin_agent.save_model(margin_path)

        # Save training statistics
        stats_path = f"{self.config['model_save_path']}/training_stats_episode_{episode_num}.json"
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)

        print(f"Models saved at episode {episode_num}")

    def plot_training_results(self):
        """Plot training results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # PPO Rewards
        axes[0, 0].plot(self.training_stats['ppo_rewards'])
        axes[0, 0].set_title('PPO Agent Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')

        # Margin Agent Rewards
        axes[0, 1].plot(self.training_stats['margin_rewards'])
        axes[0, 1].set_title('Margin Agent Rewards')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')

        # Safety Violations
        axes[1, 0].plot(self.training_stats['safety_violations'])
        axes[1, 0].set_title('Safety Violations')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Violations')

        # Average Margins
        axes[1, 1].plot(self.training_stats['margin_history'])
        axes[1, 1].set_title('Average Margins')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Margin')

        plt.tight_layout()
        plt.savefig(f"{self.config['model_save_path']}/training_results.png", dpi=300)
        plt.show()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ShieldNN with Adaptive Margins')

    # Environment parameters
    parser.add_argument('--host', type=str, default='127.0.0.1', help='CARLA host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA port')
    parser.add_argument('--start_carla', action='store_true', help='Start CARLA automatically')
    parser.add_argument('--synchronous', action='store_true', default=True, help='Synchronous mode')
    parser.add_argument('--fps', type=int, default=30, help='FPS')
    parser.add_argument('--track', type=int, default=1, help='Track number')

    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=2000, help='Number of episodes')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=1.0, help='Learning rate decay')
    parser.add_argument('--discount_factor', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--ppo_epsilon', type=float, default=0.2, help='PPO epsilon')
    parser.add_argument('--initial_std', type=float, default=1.0, help='Initial std')
    parser.add_argument('--value_scale', type=float, default=0.5, help='Value scale')
    parser.add_argument('--entropy_scale', type=float, default=0.01, help='Entropy scale')
    parser.add_argument('--horizon', type=int, default=512, help='Horizon')
    parser.add_argument('--num_epochs', type=int, default=4, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max steps per episode')

    # Model parameters
    parser.add_argument('--reward_fn', type=str, default='reward_fn', help='Reward function')
    parser.add_argument('--vae_model', type=str, default='vae', help='VAE model')
    parser.add_argument('--onnx_models_path', type=str, default='./', help='Path to ONNX models')
    parser.add_argument('--margin_agent_path', type=str, default=None, help='Path to trained margin agent')
    parser.add_argument('--model_save_path', type=str, default='./models', help='Path to save models')

    args = parser.parse_args()

    # Create configuration
    config = vars(args)

    # Create save directory
    os.makedirs(config['model_save_path'], exist_ok=True)

    # Initialize trainer
    trainer = DualAgentTrainer(config)

    # Train agents
    training_stats = trainer.train(config['num_episodes'])

    # Plot results
    trainer.plot_training_results()

    print("Training completed successfully!")


if __name__ == "__main__":
    main()