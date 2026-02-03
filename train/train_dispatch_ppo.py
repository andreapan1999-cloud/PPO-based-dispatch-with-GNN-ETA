from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.env_dispatch_ppo import DispatchPPOEnv


def make_env():
    return DispatchPPOEnv(
        data_dir="outputs",
        duration=180,
        use_gnn_eta=True,
        calibrate_gnn=True,
        calib_samples=2000,
        R=4,
        O=4,
        invalid_action_penalty=-2.0,
        wait_penalty=-0.01,
    )


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def reset_vecenv(venv, seed: int):
    try:
        return venv.reset(seed=seed)
    except TypeError:
        try:
            venv.seed(seed)
        except Exception:
            pass
        return venv.reset()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--tb", action="store_true")
    parser.add_argument("--target_kl", type=float, default=0.05)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_global_seeds(args.seed)

    venv = DummyVecEnv([make_env])
    venv = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    reset_vecenv(venv, args.seed)

    log_dir = out_dir / "tb" / f"seed{args.seed}"
    log_dir.mkdir(parents=True, exist_ok=True)

    outputs = ["stdout"]
    if args.tb:
        outputs.append("tensorboard")
    logger = configure(str(log_dir), outputs)

    model = PPO(
        policy="MlpPolicy",
        env=venv,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        n_epochs=10,
        clip_range=0.2,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=float(args.target_kl),
        verbose=1,
    )
    model.set_logger(logger)

    model.learn(total_timesteps=int(args.timesteps))

    vec_path = out_dir / f"vecnormalize_seed{args.seed}.pkl"
    model_path = out_dir / f"ppo_seed{args.seed}.zip"

    venv.save(str(vec_path))
    model.save(str(model_path))

    try:
        venv.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()