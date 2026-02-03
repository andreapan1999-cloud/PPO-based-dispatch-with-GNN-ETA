from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any

import csv
import math
import networkx as nx

from .module1_config import load_config


@dataclass
class NetworkData:
    G: nx.Graph
    node_coords: Dict[int, Tuple[float, float]]  # node_id -> (x, y) in km (synthetic grid)


def build_synthetic_grid(width: int, height: int, node_spacing_km: float) -> NetworkData:
    """
    Build an undirected grid graph with Euclidean edge lengths in km.

    Node id scheme: id = y * width + x
    """
    if width <= 1 or height <= 1:
        raise ValueError("width and height must be > 1")

    G = nx.Graph()
    node_coords: Dict[int, Tuple[float, float]] = {}

    # Add nodes
    for y in range(height):
        for x in range(width):
            nid = y * width + x
            xx = x * node_spacing_km
            yy = y * node_spacing_km
            G.add_node(nid)
            node_coords[nid] = (xx, yy)

    # Add edges (right and down to avoid duplicates)
    for y in range(height):
        for x in range(width):
            nid = y * width + x

            def add_edge_if(nx_x: int, nx_y: int) -> None:
                if 0 <= nx_x < width and 0 <= nx_y < height:
                    nid2 = nx_y * width + nx_x
                    x1, y1 = node_coords[nid]
                    x2, y2 = node_coords[nid2]
                    length_km = math.hypot(x2 - x1, y2 - y1)
                    G.add_edge(nid, nid2, length_km=length_km)

            add_edge_if(x + 1, y)
            add_edge_if(x, y + 1)

    return NetworkData(G=G, node_coords=node_coords)


def build_network_from_config(cfg: Dict[str, Any]) -> NetworkData:
    net_cfg = cfg.get("network", {})
    net_type = net_cfg.get("type", "synthetic")

    if net_type != "synthetic":
        raise NotImplementedError("Only synthetic grid is implemented in this step.")

    syn = net_cfg.get("synthetic", {})
    layout = syn.get("layout", "grid")
    if layout != "grid":
        raise NotImplementedError("Only grid layout is implemented.")

    width = int(syn.get("width", 10))
    height = int(syn.get("height", 10))
    node_spacing = float(syn.get("node_spacing", 1.0))

    return build_synthetic_grid(width=width, height=height, node_spacing_km=node_spacing)


def save_network_csv(net: NetworkData, out_dir: str | Path) -> Tuple[Path, Path]:
    """
    Save nodes.csv and edges.csv.

    nodes.csv: node_id,x_km,y_km
    edges.csv: u,v,length_km
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes_path = out_dir / "nodes.csv"
    edges_path = out_dir / "edges.csv"

    with nodes_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "x_km", "y_km"])
        for nid, (x, y) in sorted(net.node_coords.items()):
            w.writerow([nid, x, y])

    with edges_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["u", "v", "length_km"])
        for u, v, data in net.G.edges(data=True):
            w.writerow([u, v, float(data.get("length_km", 0.0))])

    return nodes_path, edges_path


def main() -> None:
    cfg = load_config("config.yaml")
    net = build_network_from_config(cfg)

    out_dir = cfg.get("output", {}).get("dir", "outputs")
    nodes_path, edges_path = save_network_csv(net, out_dir)

    print("=== Network built ===")
    print("Nodes:", net.G.number_of_nodes())
    print("Edges:", net.G.number_of_edges())
    print("Saved:", nodes_path)
    print("Saved:", edges_path)


if __name__ == "__main__":
    main()
