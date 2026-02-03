# train/train_ppo_dispatch.py
from __future__ import annotations
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from sim.env_dispatch_ppo import DispatchPPOEnv
from stable_baselines3.common.logger import configure

def main():
    # vectorized env for SB3
    env = make_vec_env(
        lambda: DispatchPPOEnv(
            data_dir="outputs",
            duration=180,
            use_gnn_eta=True,
            calibrate_gnn=True,
            calib_samples=2000,
            R=4,
            O=4,
            invalid_action_penalty=-0.5,
            wait_penalty=-0.02,
        ),
        n_envs=4,
    )
    new_logger = new_logger = configure("outputs/tb", format_strings=["tensorboard"])
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        n_steps=256,
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
    )
    model.set_logger(new_logger)

    model.learn(total_timesteps=50_000)

    out = Path("outputs/ppo_dispatch.zip")
    model.save(out)
    print("Saved:", out)


if __name__ == "__main__":
    main()