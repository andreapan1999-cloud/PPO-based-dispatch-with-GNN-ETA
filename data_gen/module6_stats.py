from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, List

import csv
from collections import Counter

from .module1_config import load_config


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def main() -> None:
    cfg = load_config("config.yaml")
    out_dir = Path(cfg.get("output", {}).get("dir", "outputs"))

    nodes_path = out_dir / "nodes.csv"
    edges_path = out_dir / "edges.csv"
    tt_path = out_dir / cfg.get("travel_time", {}).get("filename", "travel_times.csv")
    orders_path = out_dir / cfg.get("orders", {}).get("filename", "orders.csv")
    riders_path = out_dir / cfg.get("riders", {}).get("filename", "riders.csv")

    duration = int(cfg.get("simulation", {}).get("duration", 0))

    # Load
    nodes = read_csv_rows(nodes_path)
    edges = read_csv_rows(edges_path)
    tts = read_csv_rows(tt_path)
    orders = read_csv_rows(orders_path)
    riders = read_csv_rows(riders_path)

    node_ids = set(int(r["node_id"]) for r in nodes)

    # Basic counts
    n_nodes = len(nodes)
    n_edges = len(edges)
    n_tts = len(tts)
    n_orders = len(orders)
    n_riders = len(riders)

    # Checks: edges endpoints exist
    bad_edges = []
    for r in edges:
        u = int(r["u"]); v = int(r["v"])
        if u not in node_ids or v not in node_ids:
            bad_edges.append((u, v))

    # Checks: travel times refer to valid edges endpoints
    bad_tts = 0
    max_t = -1
    for r in tts:
        t = int(float(r["t_min"]))
        u = int(r["u"]); v = int(r["v"])
        if u not in node_ids or v not in node_ids:
            bad_tts += 1
        if t > max_t:
            max_t = t

    # Checks: orders
    bad_orders = 0
    order_times = []
    origin_counter = Counter()
    dest_counter = Counter()
    for r in orders:
        t = int(float(r["t_min"]))
        o = int(r["origin"]); d = int(r["dest"])
        if o not in node_ids or d not in node_ids or o == d:
            bad_orders += 1
        order_times.append(t)
        origin_counter[o] += 1
        dest_counter[d] += 1

    # Checks: riders
    bad_riders = 0
    rider_starts = []
    rider_ends = []
    speed_factors = []
    for r in riders:
        s = int(float(r["start_min"]))
        e = int(float(r["end_min"]))
        init = int(r["init_node"])
        sf = float(r["speed_factor"])
        if init not in node_ids or s < 0 or e < 0 or e < s or (duration > 0 and e > duration):
            bad_riders += 1
        rider_starts.append(s)
        rider_ends.append(e)
        speed_factors.append(sf)

    # Summary text
    lines = []
    lines.append("=== DATA GENERATION STATS ===")
    lines.append(f"Output dir: {out_dir}")
    lines.append(f"Duration (min): {duration}")
    lines.append("")
    lines.append("Files:")
    lines.append(f"- nodes.csv: {n_nodes} rows")
    lines.append(f"- edges.csv: {n_edges} rows")
    lines.append(f"- travel_times.csv: {n_tts} rows (max t_min={max_t})")
    lines.append(f"- orders.csv: {n_orders} rows")
    lines.append(f"- riders.csv: {n_riders} rows")
    lines.append("")
    lines.append("Sanity checks:")
    lines.append(f"- edges with invalid endpoints: {len(bad_edges)}")
    lines.append(f"- travel_time rows with invalid endpoints: {bad_tts}")
    lines.append(f"- orders invalid (node missing or origin==dest): {bad_orders}")
    lines.append(f"- riders invalid (time/node range): {bad_riders}")
    lines.append("")

    if order_times:
        lines.append("Orders time range:")
        lines.append(f"- min t_min: {min(order_times)}")
        lines.append(f"- max t_min: {max(order_times)}")
        lines.append("")
        top_o = origin_counter.most_common(5)
        top_d = dest_counter.most_common(5)
        lines.append(f"Top-5 origin nodes: {top_o}")
        lines.append(f"Top-5 dest nodes:   {top_d}")
        lines.append("")

    if rider_starts:
        lines.append("Riders time range:")
        lines.append(f"- min start_min: {min(rider_starts)}")
        lines.append(f"- max start_min: {max(rider_starts)}")
        lines.append(f"- min end_min:   {min(rider_ends)}")
        lines.append(f"- max end_min:   {max(rider_ends)}")
        lines.append("")
        sf_counter = Counter(speed_factors)
        lines.append(f"Speed_factor distribution: {dict(sf_counter)}")
        lines.append("")

    stats_path = out_dir / "stats.txt"
    stats_path.write_text("\n".join(lines), encoding="utf-8")

    print("=== Stats written ===")
    print("Saved:", stats_path)
    print(lines[0])
    print("Sanity:", f"bad_edges={len(bad_edges)} bad_tts={bad_tts} bad_orders={bad_orders} bad_riders={bad_riders}")


if __name__ == "__main__":
    main()
