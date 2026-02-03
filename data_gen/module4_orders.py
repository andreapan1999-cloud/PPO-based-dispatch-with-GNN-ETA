from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple

import csv
import random

from .module1_config import load_config


def load_nodes(nodes_path: str | Path) -> List[int]:
    nodes_path = Path(nodes_path)
    node_ids: List[int] = []
    with nodes_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            node_ids.append(int(row["node_id"]))
    if not node_ids:
        raise ValueError(f"No nodes loaded from {nodes_path}")
    return node_ids


def sample_time_within_hour(h: int, duration_min: int, rng: random.Random) -> int:
    """
    Sample a minute timestamp t within hour h, respecting the simulation duration.
    This avoids the 'last minute spike' bug.
    """
    start = h * 60
    end = min(start + 60, duration_min)
    if start >= duration_min:
        return duration_min - 1
    return rng.randint(start, end - 1)


def choose_center_node(nodes: List[int], cfg: Dict[str, Any]) -> int:
    """
    Choose a center node for clustered sampling.
    For synthetic grid, we pick the median id as a simple deterministic proxy.
    (You can later replace this with geometric centroid.)
    """
    order_cfg = cfg.get("orders", {})
    center = order_cfg.get("center_node", None)
    if center is not None:
        return int(center)
    nodes_sorted = sorted(nodes)
    return nodes_sorted[len(nodes_sorted) // 2]


def pick_node_clustered(
    nodes: List[int], center: int, cfg: Dict[str, Any], rng: random.Random
) -> int:
    """
    Simple clustered pick: with probability p_center choose center,
    else uniform random from nodes.
    """
    order_cfg = cfg.get("orders", {})
    p_center = float(order_cfg.get("p_center", 0.25))
    if rng.random() < p_center:
        return center
    return rng.choice(nodes)


def generate_orders(cfg: Dict[str, Any], nodes: List[int]) -> List[Dict[str, Any]]:
    sim = cfg.get("simulation", {})
    duration = int(sim.get("duration", 180))
    seed = int(sim.get("seed", 42))
    rng = random.Random(seed)

    order_cfg = cfg.get("orders", {})
    n_orders = int(order_cfg.get("n_orders", 200))

    # Hour weights (0-23). If not provided, uniform.
    hour_weights = order_cfg.get("hour_weights", None)
    if hour_weights is None:
        hour_weights = [1.0] * 24
    if len(hour_weights) != 24:
        raise ValueError("orders.hour_weights must have length 24")

    center = choose_center_node(nodes, cfg)

    orders: List[Dict[str, Any]] = []
    for oid in range(n_orders):
        h = rng.choices(range(24), weights=hour_weights, k=1)[0]
        t = sample_time_within_hour(h, duration, rng)

        origin = pick_node_clustered(nodes, center, cfg, rng)
        dest = pick_node_clustered(nodes, center, cfg, rng)
        while dest == origin:
            dest = pick_node_clustered(nodes, center, cfg, rng)

        orders.append(
            {
                "order_id": oid,
                "t_min": t,
                "origin": origin,
                "dest": dest,
            }
        )

    # Sort by time then id (useful for simulator)
    orders.sort(key=lambda x: (x["t_min"], x["order_id"]))
    return orders


def save_orders_csv(orders: List[Dict[str, Any]], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["order_id", "t_min", "origin", "dest"])
        w.writeheader()
        for row in orders:
            w.writerow(row)

    return out_path


def main() -> None:
    cfg = load_config("config.yaml")
    out_dir = Path(cfg.get("output", {}).get("dir", "outputs"))

    nodes = load_nodes(out_dir / "nodes.csv")

    orders = generate_orders(cfg, nodes)

    out_name = cfg.get("orders", {}).get("filename", "orders.csv")
    out_path = out_dir / out_name
    save_orders_csv(orders, out_path)

    print("=== Orders generated ===")
    print("Nodes:", len(nodes))
    print("Orders:", len(orders))
    print("Saved:", out_path)
    print("First 5:", orders[:5])


if __name__ == "__main__":
    main()
