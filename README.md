# Adaptive ShieldNN 🚗🛡️

This project enhances the ShieldNN framework with **adaptive, learnable safety margins** δ(x) using Reinforcement Learning. It leverages certified Control Barrier Functions (CBFs), closed-form override λ(x), and state-aware safety constraints in a simulated Kinematic Bicycle Model (KBM) environment.

---

adaptive_shieldnn/
│
├── core/
│   ├── shield_model.py             # Loads pretrained safety networks (δ filters)
│   ├── safe_controller.py          # Applies λ override using δ(x) and CBF logic
│   ├── certifier.py                # Loads certified δ_min(x) values and clamps δ(x)
│   ├── barrier.py                  # h(x), ∇h(x), α(h), and Lie derivative logic
│   └── dynamics.py                 # Kinematic Bicycle Model (KBM) dynamics
│
├── env/
│   └── kbm_env.py                  # Gym-compatible environment (state, reward, step)
│
├── train/
│   └── train_margin_rl.py          # RL training loop (PPO/SAC), adaptive δ(x)
│
├── networks/
│   └── margin_net.py               # Neural network architecture for δ(x)
│
├── scripts/
│   └── pretrain_certified_delta.py # Optional: supervised warmup for δ(x)
│
├── utils/
│   ├── reward_shaping.py           # Constructs r_t with all penalties
│   └── visualization.py            # Plots: δ(x), λ(x), constraint violations
│
├── main.py                         # Entry point: runs training, testing, or eval
└── README.md                       # Project overview and instructions


## ⚙️ Installation

```bash
git clone https://github.com/yourname/adaptive-shieldnn.git
cd adaptive-shieldnn
pip install -r requirements.txt

## 1 - Pretrain with Certified δ(x)

python scripts/pretrain_certified_delta.py

## 2 - Train Adaptive δ(x) with PPO

python main.py --mode train --epochs 500

## 3 - Evaluate and Visualize

python main.py --mode eval --model checkpoints/delta_final.pt
