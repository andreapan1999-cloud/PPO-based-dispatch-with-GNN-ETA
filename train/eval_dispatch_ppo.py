# train/eval_dispatch_ppo.py

import argparse
import random
from typing import List, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.env_dispatch_ppo import DispatchPPOEnv
import json
import csv
from pathlib import Path

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
        wait_penalty=-0.1,
    )


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
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


def run_one(model, venv, seed: int, deterministic: bool) -> float:
    obs = reset_vecenv(venv, seed)

    done = [False]
    ep_reward = 0.0

    while not done[0]:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = venv.step(action)
        ep_reward += float(reward[0])

    return ep_reward


def evaluate(
    model_path: str,
    vecnorm_path: str,
    episodes: int,
    seed_base: int,
    deterministic: bool,
) -> Tuple[np.ndarray, List[int]]:
    model = PPO.load(model_path)

    venv = DummyVecEnv([make_env])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False

    model.set_env(venv)

    rewards: List[float] = []
    seeds: List[int] = []

    for i in range(episodes):
        seed = seed_base + i
        set_global_seeds(seed)

        venv = DummyVecEnv([make_env])
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False

        model.set_env(venv)

        r = run_one(model, venv, seed=seed, deterministic=deterministic)
        rewards.append(r)
        seeds.append(seed)

        venv.close()

    return np.asarray(rewards, dtype=np.float64), seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/ppo.zip")
    parser.add_argument("--vecnorm", type=str, default="outputs/vecnormalize.pkl")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed_base", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    rewards, seeds = evaluate(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        episodes=args.episodes,
        seed_base=args.seed_base,
        deterministic=args.deterministic,
    )

    # ===== EXPORT RESULTS (for paper / sharing) =====
    out_dir = Path("outputs/eval_exports")
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = "deterministic" if args.deterministic else "stochastic"
    tag = f"_{args.tag}" if args.tag else ""

    # CSV: per-episode rewards
    csv_path = out_dir / f"eval_{mode}{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "seed", "reward"])
        for i, (s, r) in enumerate(zip(seeds, rewards)):
            writer.writerow([i, s, float(r)])

    # JSON: summary stats
    summary = {
        "model": args.model,
        "vecnormalize": args.vecnorm,
        "episodes": int(len(rewards)),
        "seed_base": int(args.seed_base),
        "mode": mode,
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
    }

    json_path = out_dir / f"summary_{mode}{tag}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[EXPORT] CSV saved to: {csv_path}")
    print(f"[EXPORT] JSON saved to: {json_path}")
    # ===============================================

    print(f"episodes: {len(rewards)}")
    if seeds:
        print(f"seeds: {seeds[0]}..{seeds[-1]}")
    print(
        "reward mean/std/min/max: "
        f"{rewards.mean():.3f} / {rewards.std():.3f} / {rewards.min():.3f} / {rewards.max():.3f}"
    )


if __name__ == "__main__":
    main()