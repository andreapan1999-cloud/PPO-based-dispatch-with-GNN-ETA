# train/train_dispatch_gnn.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple

import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam

from torch_geometric.data import HeteroData

from sim.env_dispatch import DispatchEnv
from models.dispatch_bipartite_gnn import DispatchBiGNN

_SP_CACHE = {}

def _shortest_eta_cached(env: DispatchEnv, a: int, b: int, t: int) -> float:
    # cache by time bucket (same as your calibration bucket)
    bucket_size = 45
    tb = int((t // bucket_size) * bucket_size)
    key = (a, b, tb)
    if key in _SP_CACHE:
        return _SP_CACHE[key]
    val = float(env._shortest_eta(a, b, tb))
    _SP_CACHE[key] = val
    return val

def _rider_feat(env: DispatchEnv, rider: Dict[str, Any], t: int) -> List[float]:
    # simple features: [t_norm, node_x, node_y, speed_factor]
    rid = int(rider["rider_id"])
    node = int(env.rider_pos.get(rid, int(rider["init_node"])))
    x, y = env.node_xy[node]
    return [t / max(1.0, float(env.duration)), float(x), float(y), float(rider["speed_factor"])]


def _order_feat(env: DispatchEnv, order: Dict[str, Any], t: int) -> List[float]:
    # [age_norm, origin_x, origin_y, dest_x, dest_y]
    age = max(0, t - int(order["t_min"]))
    ox, oy = env.node_xy[int(order["origin"])]
    dx, dy = env.node_xy[int(order["dest"])]
    return [age / 180.0, float(ox), float(oy), float(dx), float(dy)]


def _teacher_cost(env: DispatchEnv, rider: Dict[str, Any], order: Dict[str, Any], t: int) -> float:
    # cost = eta(rider_node->origin) + eta(origin->dest), time-dependent (csv or calibrated gnn)
    rid = int(rider["rider_id"])
    rider_node = int(env.rider_pos.get(rid, int(rider["init_node"])))
    o = int(order["origin"])
    d = int(order["dest"])

    eta1 = _shortest_eta_cached(env, rider_node, o, t) / float(rider["speed_factor"])
    eta2 = _shortest_eta_cached(env, o, d, t)
    return eta1 + eta2


def build_bipartite_graph(env: DispatchEnv, riders: List[Dict[str, Any]], orders: List[Dict[str, Any]], t: int) -> Tuple[HeteroData, torch.Tensor]:
    data = HeteroData()

    rider_x = torch.tensor([_rider_feat(env, r, t) for r in riders], dtype=torch.float32)
    order_x = torch.tensor([_order_feat(env, o, t) for o in orders], dtype=torch.float32)

    data["rider"].x = rider_x
    data["order"].x = order_x

    # fully connect riders->orders as candidate edges
    src, dst, y = [], [], []
    for i, r in enumerate(riders):
        for j, o in enumerate(orders):
            src.append(i)
            dst.append(j)
            y.append(_teacher_cost(env, r, o, t))

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    data[("rider", "to", "order")].edge_index = edge_index
    data[("order", "rev_to", "rider")].edge_index = edge_index.flip(0)

    y = torch.tensor(y, dtype=torch.float32)  # target cost per edge
    return data, y


def collect_samples(env: DispatchEnv, steps: int = 160, max_r: int = 4, max_o: int = 4) -> List[Tuple[HeteroData, torch.Tensor]]:
    buf = []
    obs, _ = env.reset()
    for _ in range(steps):
        # advance time without dispatch to populate queues a bit
        env.step(0)

        if len(env.idle_riders) == 0 or len(env.pending_orders) == 0:
            continue

        riders = env.idle_riders[:max_r]
        orders = env.pending_orders[:max_o]

        data, y = build_bipartite_graph(env, riders, orders, env.t)
        buf.append((data, y))
        if len(buf) % 5 == 0:
            print("collected", len(buf), "samples at t=", env.t)
    return buf


def main() -> None:
    random.seed(0)
    torch.manual_seed(0)

    # Use calibrated GNN ETA inside env (already stable for you)
    env = DispatchEnv("outputs", 180, use_gnn_eta=True, calibrate_gnn=True, calib_samples=2000)

    samples = collect_samples(env, steps=160, max_r=4, max_o=4)
    print("samples:", len(samples))
    if not samples:
        raise RuntimeError("No samples collected. Try increasing steps.")

    model = DispatchBiGNN(rider_dim=4, order_dim=5, hidden=64)
    opt = Adam(model.parameters(), lr=1e-3)

    # simple train/val split
    n = len(samples)
    split = int(0.8 * n)
    train_s = samples[:split]
    val_s = samples[split:]

    for epoch in range(1, 41):
        model.train()
        loss_sum = 0.0
        for data, y in train_s:
            pred = model(data)
            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())

        model.eval()
        with torch.no_grad():
            vloss = 0.0
            for data, y in val_s:
                pred = model(data)
                vloss += float(F.mse_loss(pred, y).item())

        if epoch % 20 == 0:
            print(f"epoch {epoch:03d} train_mse={loss_sum/max(1,len(train_s)):.4f} val_mse={vloss/max(1,len(val_s)):.4f}")

    out = Path("outputs/dispatch_bignn.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print("Saved:", out)


if __name__ == "__main__":
    main()