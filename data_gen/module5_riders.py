from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import csv
import random

try:
    # when run as a package: python -m data_gen.module5_riders
    from .module1_config import load_config
    from .module4_orders import load_nodes, choose_center_node
except ImportError:
    # when run as a script: python data_gen/module5_riders.py
    from module1_config import load_config
    from module4_orders import load_nodes, choose_center_node



def generate_riders(cfg: Dict[str, Any], nodes: List[int]) -> List[Dict[str, Any]]:
    sim = cfg.get("simulation", {})
    duration = int(sim.get("duration", 180))
    seed = int(sim.get("seed", 42))
    rng = random.Random(seed + 1000)  # offset to avoid being identical to orders

    rider_cfg = cfg.get("riders", {})
    n_riders = int(rider_cfg.get("n_riders", 30))

    # Speed factor options (multiplier on travel time or speed)
    speed_factor_options = rider_cfg.get("speed_factor_options", [0.8, 1.0, 1.2])
    speed_factor_prob = rider_cfg.get("speed_factor_prob", [0.2, 0.6, 0.2])
    if len(speed_factor_options) != len(speed_factor_prob):
        raise ValueError("riders.speed_factor_options and riders.speed_factor_prob must have same length")

    # Shift times
    earliest_start = int(rider_cfg.get("earliest_start_min", 0))
    latest_start = int(rider_cfg.get("latest_start_min", max(0, duration - 60)))
    min_shift = int(rider_cfg.get("min_shift_min", 60))
    max_shift = int(rider_cfg.get("max_shift_min", min(180, duration)))

    # Clustered initial positions (optional)
    p_center = float(rider_cfg.get("p_center", 0.25))
    center = choose_center_node(nodes, cfg)

    riders: List[Dict[str, Any]] = []
    for rid in range(n_riders):
        start = rng.randint(earliest_start, latest_start) if latest_start >= earliest_start else earliest_start
        shift_len = rng.randint(min_shift, max_shift) if max_shift >= min_shift else min_shift
        end = min(duration, start + shift_len)

        # initial node
        if rng.random() < p_center:
            init_node = center
        else:
            init_node = rng.choice(nodes)

        speed_factor = rng.choices(speed_factor_options, weights=speed_factor_prob, k=1)[0]

        riders.append(
            {
                "rider_id": rid,
                "start_min": start,
                "end_min": end,
                "init_node": init_node,
                "speed_factor": float(speed_factor),
            }
        )

    riders.sort(key=lambda x: x["rider_id"])
    return riders


def save_riders_csv(riders: List[Dict[str, Any]], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["rider_id", "start_min", "end_min", "init_node", "speed_factor"],
        )
        w.writeheader()
        for row in riders:
            w.writerow(row)

    return out_path


def main() -> None:
    cfg = load_config("config.yaml")
    out_dir = Path(cfg.get("output", {}).get("dir", "outputs"))

    nodes = load_nodes(out_dir / "nodes.csv")
    riders = generate_riders(cfg, nodes)

    out_name = cfg.get("riders", {}).get("filename", "riders.csv")
    out_path = out_dir / out_name
    save_riders_csv(riders, out_path)

    print("=== Riders generated ===")
    print("Nodes:", len(nodes))
    print("Riders:", len(riders))
    print("Saved:", out_path)
    print("First 5:", riders[:5])


if __name__ == "__main__":
    main()
