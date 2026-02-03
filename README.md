# 1. Clone repository
git clone https://github.com/andreapan1999-cloud/PPO-based-dispatch-with-GNN-ETA.git
cd PPO-based-dispatch-with-GNN-ETA

# 2. Create Python environment (Python 3.9 recommended)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -U pip
pip install -r requirements.txt

# 4. Quick smoke test (2k steps)
python -m train.train_dispatch_ppo --seed 0 --timesteps 2048

# 5. Evaluate pretrained model (deterministic)
python -m train.eval_dispatch_ppo \
  --model outputs/runs/seed0_T50000_kl0.05_20260201_145329/model \
  --vecnorm outputs/runs/seed0_T50000_kl0.05_20260201_145329/vecnormalize.pkl \
  --episodes 50 --seed_base 2000 --deterministic