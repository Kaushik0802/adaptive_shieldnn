
# Adaptive ShieldNN: Reinforcement Learning-Guided Safety Margins for Control Barrier Functions

Adaptive ShieldNN extends the ShieldNN framework by enabling state-dependent, learnable safety margins (δ(x)) in Control Barrier Function (CBF)-based control. This allows for adaptive, less conservative safety constraints while maintaining certified safety guarantees.

---

## 🚩 Project Goals

- **Replicate** certified safety margin computation using Yasser's original ShieldNN framework.
- **Introduce adaptive δ(x)** networks trained via reinforcement learning (PPO/SAC) to adjust safety margins dynamically.
- **Maintain safety** using closed-form λ-corrections derived from δ(x) and the CBF constraint.
- **Visualize** and evaluate adaptive vs. certified margin performance in simulation environments.

---

## 📁 Folder Structure

```
adaptive_shieldnn/
│
├── core/
│   ├── shield_model.py             # Loads δ(x) networks
│   ├── safe_controller.py          # λ override logic
│   ├── certifier.py                # Certified δ_min(x) loader & clamping
│   ├── barrier.py                  # h(x), ∇h(x), α(h), Lie derivatives
│   └── dynamics.py                 # Kinematic Bicycle Model (KBM)
│
├── env/
│   └── kbm_env.py                  # Gym-compatible KBM environment
│
├── train/
│   └── train_margin_rl.py          # PPO/SAC training for δ(x)
│
├── networks/
│   └── margin_net.py               # δ(x) neural network definition
│
├── scripts/
│   └── pretrain_certified_delta.py # Supervised warm-up using δ_min(x)
│
├── utils/
│   ├── reward_shaping.py           # Shaped reward with smoothness/safety penalties
│   └── visualization.py            # Plotting δ(x), λ(x), violations
│
├── certs/
│   ├── gen_d2_vals.py              # Generates certified δ(x) mesh (D2CertVals.p)
│   ├── deriv_definitions.py        # Derivative bounds used in certification
│   ├── deriv1_certs.p              # Pickled bound data (1st order)
│   ├── deriv2_certs.p              # Pickled bound data (2nd order)
│   ├── deriv3_certs.p              # Pickled bound data (3rd order)
│   └── D2CertVals.p                # Generated certified δ(x) mesh
│
├── main.py                         # Entry point: train/test/evaluate
└── README.md                       # This file
```

---

## ⚙️ Setup Instructions

```bash
# Clone the repository and navigate into it
git clone https://github.com/yourusername/adaptive_shieldnn.git
cd adaptive_shieldnn

# Create virtual environment
python -m venv shieldnn_env
shieldnn_env\Scripts\activate    # On Windows
# source shieldnn_env/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## ✅ Running the Project

### Step 1: Generate Certified δ(x) Mesh

This generates a file `D2CertVals.p` based on Yasser's method.

```bash
cd certs
python gen_d2_vals.py
```

Make sure you have these files in `certs/`:  
- `deriv1_certs.p`
- `deriv2_certs.p`
- `deriv3_certs.p`

---

### Step 2: Pretrain Adaptive δ(x)

Supervised learning on the certified δ(x) mesh.

```bash
cd scripts
python pretrain_certified_delta.py
```

---

### Step 3: Train Adaptive Margin via RL

```bash
cd train
python train_margin_rl.py
```

This uses PPO/SAC to learn δ(x) online in a simulated KBM environment.

---

### Step 4: Evaluate and Visualize

Use `main.py` or scripts in `utils/visualization.py` to analyze and plot δ(x), λ(x), violations, and rewards.

```bash
python main.py --mode eval
```

---

## 📌 Notes

- All margin values are clamped below δ_min(x) using `certifier.py`.
- RL safety is ensured by closed-form λ override.
- Smoothness and conservatism are penalized in reward shaping.

---

## 📧 Contact

Project by Kaushik. Based on ShieldNN by Yasser Shoukry's ShieldNN

