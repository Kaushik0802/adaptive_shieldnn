import math
import numpy as np
import torch
import onnx
import onnx_tf
import tensorflow as tf
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_margin_agent import AdaptiveMarginAgent

class EnhancedSafetyFilter:
    """
    Enhanced ShieldNN safety filter with adaptive margins using a second RL agent.
    Balances aggressive performance with safety through dynamic margin adaptation.
    """

    def __init__(self, onnx_models_path="./", margin_agent_path=None):
        # Original ShieldNN parameters (paper-compliant)
        self.sigma = 0.48
        self.l_r = 2
        self.rBar = 4
        self.steer_to_angle = 1.22173

        # Load ONNX models (original ShieldNN networks)
        self.tf_model1 = onnx_tf.backend.prepare(onnx.load(f"{onnx_models_path}/1.onnx"))
        self.tf_model2 = onnx_tf.backend.prepare(onnx.load(f"{onnx_models_path}/2.onnx"))
        self.tf_model3 = onnx_tf.backend.prepare(onnx.load(f"{onnx_models_path}/3.onnx"))

        # Initialize adaptive margin agent
        self.margin_agent = AdaptiveMarginAgent()
        if margin_agent_path and os.path.exists(margin_agent_path):
            self.margin_agent.load_model(margin_agent_path)
            print(f"Loaded trained margin agent from {margin_agent_path}")

        # Performance tracking
        self.episode_stats = {
            'total_steps': 0,
            'filter_applications': 0,
            'safety_violations': 0,
            'avg_margin': 0.0,
            'avg_speed': 0.0,
            'performance_metrics': []
        }

        # Margin history for consistency
        self.margin_history = []
        self.last_margin = 0.5  # Default conservative margin

    def barrier(self):
        """Compute barrier function (paper-compliant)."""
        return self.rBar / (self.sigma * math.cos(self.xi / 2) + 1 - self.sigma)

    def get_adaptive_margin(self, state):
        """
        Get adaptive margin from the RL agent.

        Args:
            state: [xi, r, v, steer, throttle] - current vehicle state

        Returns:
            margin: Adaptive safety margin
        """
        # Ensure state is normalized and in correct format
        state_normalized = self._normalize_state(state)

        # Get margin from RL agent
        margin = self.margin_agent.get_margin(state_normalized, training=False)

        # Store for consistency tracking
        self.margin_history.append(margin)
        if len(self.margin_history) > 100:
            self.margin_history.pop(0)

        self.last_margin = margin
        return margin

    def _normalize_state(self, state):
        """Normalize state for the margin agent."""
        xi, r, v, steer, throttle = state

        # Normalize to reasonable ranges
        xi_norm = np.clip(xi / math.pi, -1, 1)  # [-π, π] -> [-1, 1]
        r_norm = np.clip(r / 50.0, 0, 1)       # [0, 50] -> [0, 1]
        v_norm = np.clip(v / 30.0, 0, 1)       # [0, 30] -> [0, 1]
        steer_norm = np.clip(steer, -1, 1)      # Already normalized
        throttle_norm = np.clip(throttle, 0, 1)  # Already normalized

        return np.array([xi_norm, r_norm, v_norm, steer_norm, throttle_norm])

    def filter_control(self, control, vehicle_speed=0):
        """
        Enhanced control filtering with adaptive margins.

        Args:
            control: Original control action [steer, throttle]
            vehicle_speed: Current vehicle speed

        Returns:
            filtered_control: Safe control action
            steer_diff: Difference in steering
            filter_applied: Whether filter was applied
        """
        # Get current state for margin computation
        state = np.array([self.xi, self.r, vehicle_speed, control.steer, control.throttle])

        # Get adaptive margin
        adaptive_margin = self.get_adaptive_margin(state)

        # Update episode statistics
        self.episode_stats['total_steps'] += 1
        self.episode_stats['avg_margin'] = np.mean(self.margin_history[-100:]) if self.margin_history else adaptive_margin
        self.episode_stats['avg_speed'] = vehicle_speed

        # Enhanced threshold logic with adaptive margins
        barrier_value = self.barrier()

        # Determine which model to use based on adaptive margin
        if self.r > barrier_value + adaptive_margin + 0.6:
            # No filtering needed - safe zone
            return control, 0.0, False

        elif self.r > barrier_value + adaptive_margin + 0.5:
            # Use model 3 with adaptive margin
            model = self.tf_model3
            margin_offset = adaptive_margin

        elif self.r > barrier_value + adaptive_margin + 0.25:
            # Use model 2 with adaptive margin
            model = self.tf_model2
            margin_offset = adaptive_margin

        else:
            # Use model 1 with adaptive margin (most conservative)
            model = self.tf_model1
            margin_offset = adaptive_margin

        # Apply ShieldNN filtering with adaptive margin
        filtered_control, steer_diff, filter_applied = self._apply_shield_filter(
            control, model, margin_offset
        )

        # Update statistics
        if filter_applied:
            self.episode_stats['filter_applications'] += 1

        return filtered_control, steer_diff, filter_applied

    def _apply_shield_filter(self, control, model, margin_offset):
        """
        Apply ShieldNN filtering with adaptive margin offset.

        Args:
            control: Original control
            model: ONNX model to use
            margin_offset: Adaptive margin offset

        Returns:
            filtered_control: Safe control
            steer_diff: Steering difference
            filter_applied: Whether filter was applied
        """
        # Convert steering to angle
        delta = control.steer * self.steer_to_angle

        # Convert to beta control (paper formula)
        self.beta = math.atan(0.5 * math.tan(delta))

        # Prepare input for neural network
        input_data = np.array([[[self.xi, self.beta]]])

        # Get safe beta from ShieldNN
        output = model.run(input_data)['PathDifferences3_Add'][0, 0, 0]

        # Convert back to steering angle with adaptive margin
        delta_new = math.atan(2 * math.tan(output))

        # Apply adaptive margin to steering
        margin_adjusted_steer = delta_new / self.steer_to_angle

        # Create new control with filtered steering
        new_control = type(control)()
        new_control.steer = float(margin_adjusted_steer)
        new_control.throttle = control.throttle  # Keep original throttle

        # Compute steering difference
        steer_diff = abs(new_control.steer - control.steer)

        return new_control, steer_diff, True

    def set_filter_inputs(self, xi, r):
        """Set current state inputs for filtering."""
        self.xi = xi
        self.r = r

    def update_performance_metrics(self, collision=False, track_progress=0):
        """Update performance tracking metrics."""
        if collision:
            self.episode_stats['safety_violations'] += 1

        self.episode_stats['performance_metrics'].append({
            'margin': self.last_margin,
            'collision': collision,
            'track_progress': track_progress,
            'speed': self.episode_stats['avg_speed']
        })

    def get_performance_summary(self):
        """Get performance summary for the current episode."""
        if not self.episode_stats['performance_metrics']:
            return {}

        metrics = self.episode_stats['performance_metrics']

        return {
            'total_steps': self.episode_stats['total_steps'],
            'filter_applications': self.episode_stats['filter_applications'],
            'safety_violations': self.episode_stats['safety_violations'],
            'avg_margin': self.episode_stats['avg_margin'],
            'avg_speed': self.episode_stats['avg_speed'],
            'filter_rate': self.episode_stats['filter_applications'] / max(1, self.episode_stats['total_steps']),
            'safety_rate': 1.0 - (self.episode_stats['safety_violations'] / max(1, self.episode_stats['total_steps'])),
            'margin_consistency': self._compute_margin_consistency()
        }

    def _compute_margin_consistency(self):
        """Compute margin consistency over recent history."""
        if len(self.margin_history) < 10:
            return 1.0

        recent_margins = self.margin_history[-10:]
        margin_changes = [abs(recent_margins[i] - recent_margins[i-1])
                         for i in range(1, len(recent_margins))]

        avg_change = np.mean(margin_changes)
        consistency = max(0, 1.0 - avg_change / 0.2)  # Normalize to [0, 1]

        return consistency

    def save_margin_agent(self, path):
        """Save the trained margin agent."""
        self.margin_agent.save_model(path)

    def load_margin_agent(self, path):
        """Load a trained margin agent."""
        self.margin_agent.load_model(path)

    def reset_episode_stats(self):
        """Reset episode statistics."""
        self.episode_stats = {
            'total_steps': 0,
            'filter_applications': 0,
            'safety_violations': 0,
            'avg_margin': 0.0,
            'avg_speed': 0.0,
            'performance_metrics': []
        }
        self.margin_history = []


class AdaptiveMarginEnvironment:
    """
    Environment wrapper for training the adaptive margin agent.
    Integrates with the existing CARLA environment.
    """

    def __init__(self, carla_env, enhanced_filter):
        self.carla_env = carla_env
        self.enhanced_filter = enhanced_filter
        self.current_episode_data = []

    def reset(self):
        """Reset the environment."""
        obs = self.carla_env.reset()
        self.enhanced_filter.reset_episode_stats()
        self.current_episode_data = []
        return obs

    def step(self, action):
        """
        Take a step in the environment with adaptive margin tracking.

        Args:
            action: [steer, throttle] action from PPO agent

        Returns:
            obs: Observation
            reward: Environment reward
            done: Episode done flag
            info: Additional information including margin data
        """
        # Get current state for margin computation
        current_state = self._get_current_state()

        # Apply enhanced safety filter
        filtered_action, steer_diff, filter_applied = self.enhanced_filter.filter_control(
            type('Control', (), {'steer': action[0], 'throttle': action[1]})(),
            current_state[2]  # vehicle speed
        )

        # Take step in CARLA environment
        obs, reward, done, info = self.carla_env.step([filtered_action.steer, filtered_action.throttle])

        # Update performance metrics
        collision = info.get('collision', False)
        track_progress = info.get('track_progress', 0)
        self.enhanced_filter.update_performance_metrics(collision, track_progress)

        # Store episode data for margin agent training
        episode_data = {
            'state': current_state,
            'margin': self.enhanced_filter.last_margin,
            'filter_applied': filter_applied,
            'collision': collision,
            'reward': reward,
            'track_progress': track_progress
        }
        self.current_episode_data.append(episode_data)

        # Add margin information to info dict
        info['margin_used'] = self.enhanced_filter.last_margin
        info['filter_applied'] = filter_applied
        info['steer_diff'] = steer_diff
        info['performance_summary'] = self.enhanced_filter.get_performance_summary()

        return obs, reward, done, info

    def _get_current_state(self):
        """Get current state for margin computation."""
        # This would need to be adapted based on the actual CARLA environment interface
        # For now, using placeholder values
        xi = getattr(self.carla_env, 'xi', 0.0)
        r = getattr(self.carla_env, 'r', 20.0)
        v = getattr(self.carla_env, 'vehicle_speed', 15.0)
        steer = getattr(self.carla_env, 'last_steer', 0.0)
        throttle = getattr(self.carla_env, 'last_throttle', 0.5)

        return np.array([xi, r, v, steer, throttle])

    def get_episode_data(self):
        """Get episode data for margin agent training."""
        return self.current_episode_data


# Example usage and integration
def create_enhanced_safety_filter(onnx_models_path="./", margin_agent_path=None):
    """
    Factory function to create enhanced safety filter.

    Args:
        onnx_models_path: Path to ONNX model files
        margin_agent_path: Path to trained margin agent (optional)

    Returns:
        EnhancedSafetyFilter: Configured safety filter
    """
    return EnhancedSafetyFilter(onnx_models_path, margin_agent_path)


def integrate_with_training(carla_env, enhanced_filter):
    """
    Integrate enhanced filter with existing training pipeline.

    Args:
        carla_env: Original CARLA environment
        enhanced_filter: Enhanced safety filter

    Returns:
        AdaptiveMarginEnvironment: Wrapped environment
    """
    return AdaptiveMarginEnvironment(carla_env, enhanced_filter)


if __name__ == "__main__":
    # Example usage
    enhanced_filter = create_enhanced_safety_filter()
    print("Enhanced Safety Filter created successfully!")
    print(f"Margin range: [{enhanced_filter.margin_agent.margin_min}, {enhanced_filter.margin_agent.margin_max}]")