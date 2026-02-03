from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import csv
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
# --- GNN calibration global cache (process-level) ---
_GNN_CALIB_CACHE = {}

class DispatchEnv(gym.Env):
    """
    Dispatch Environment (clean rebuild)

    Observation: [t, #pending_orders, #idle_riders]
    Action:
      0 = do nothing
      1 = dispatch first order to first idle rider
    """

    metadata = {"render_modes": []}

    def __init__(self, data_dir: str, duration: int, use_gnn_eta: bool = False,
             gnn_ckpt: str = "outputs/eta_gnn.pt",
             calibrate_gnn: bool = True, calib_samples: int = 400,
             verbose: bool = False):

        super().__init__()
        self.verbose = bool(verbose)

        self.data_dir = Path(data_dir)
        self.duration = int(duration)

        # ---- load CSV first (needed by gnn init & calib) ----
        self.orders = self._load_orders(self.data_dir / "orders.csv")
        self.riders = self._load_riders(self.data_dir / "riders.csv")
        self.node_xy = self._load_nodes(self.data_dir / "nodes.csv")
        self.travel_time = self._load_travel_times(self.data_dir / "travel_times.csv")
        self.G = self._build_graph(self.data_dir / "edges.csv")

        # ---- gnn flags ----
        self.use_gnn_eta = bool(use_gnn_eta)
        self.gnn_ckpt = Path(gnn_ckpt)
        self.calibrate_gnn = bool(calibrate_gnn)
        self.calib_samples = int(calib_samples)

        # ---- default calib params ----
        self.gnn_alpha = 1.0
        self.gnn_beta = 0.0
        self.gnn_bucket_scale = None
        self.edge_scale = None

        # ---- init gnn + calibration (cached) ----
        if self.use_gnn_eta:
            self._init_gnn_eta()

            if self.calibrate_gnn:
                cache_key = (str(self.gnn_ckpt.resolve()), str(self.data_dir.resolve()), int(self.duration), int(self.calib_samples))
                cached = _GNN_CALIB_CACHE.get(cache_key)

                if cached is not None:
                    # reuse cached calibration
                    self.gnn_alpha = cached.get("gnn_alpha", 1.0)
                    self.gnn_beta = cached.get("gnn_beta", 0.0)
                    self.gnn_bucket_scale = cached.get("gnn_bucket_scale", None)
                    self.edge_scale = cached.get("edge_scale", None)
                else:
                    # compute once
                    self._fit_gnn_calibration()
                    _GNN_CALIB_CACHE[cache_key] = {
                        "gnn_alpha": self.gnn_alpha,
                        "gnn_beta": self.gnn_beta,
                        "gnn_bucket_scale": getattr(self, "gnn_bucket_scale", None),
                        "edge_scale": getattr(self, "edge_scale", None),
                    }

        
        self.t = 0
        self.pending_orders: List[Dict[str, Any]] = []
        self.idle_riders: List[Dict[str, Any]] = []
        self.rider_pos: Dict[int, int] = {}

        self.observation_space = spaces.Box(
            low=0.0,
            high=np.array([self.duration, 10000, 10000], dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.busy_riders = []              # list of rider dicts
        self.busy_until = {}               # rider_id -> t_min
        self.orders_ep = None
        self.riders_ep = None
        self._orders_by_t = None
        self._riders_by_t = None
        
       
 
    # ---------------- CSV loaders ----------------
    def _load_orders(self, path: Path) -> List[Dict[str, Any]]:
        orders = []
        with path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                orders.append(
                    {
                        "order_id": int(row["order_id"]),
                        "t_min": int(row["t_min"]),
                        "origin": int(row["origin"]),
                        "dest": int(row["dest"]),
                    }
                )
        return orders

    def _load_riders(self, path: Path) -> List[Dict[str, Any]]:
        riders = []
        with path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                riders.append(
                    {
                        "rider_id": int(row["rider_id"]),
                        "start_min": int(row["start_min"]),
                        "end_min": int(row["end_min"]),
                        "init_node": int(row["init_node"]),
                        "speed_factor": float(row["speed_factor"]),
                    }
                )
        return riders

    def _load_nodes(self, path: Path) -> Dict[int, tuple[float, float]]:
        node_xy: Dict[int, tuple[float, float]] = {}
        with path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                nid = int(row["node_id"])
                node_xy[nid] = (float(row["x_km"]), float(row["y_km"]))
        return node_xy

    def _load_travel_times(self, path: Path):
        """
        travel_time[(u, v)] = list of (t_min, travel_time_min), sorted by t_min
        """
        table: Dict[tuple[int, int], list[tuple[int, float]]] = {}
        with path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                u = int(row["u"])
                v = int(row["v"])
                t = int(float(row["t_min"]))
                tt = float(row["travel_time_min"])
                table.setdefault((u, v), []).append((t, tt))

        for k in table:
            table[k].sort(key=lambda x: x[0])
        return table
    def _build_graph(self, edges_path: Path) -> nx.DiGraph:
        G = nx.DiGraph()
        with edges_path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                u = int(row["u"])
                v = int(row["v"])
                # store edge; weight will be time-dependent at query time
                G.add_edge(u, v)
        return G

    # ---------------- Gym API ----------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            rng = np.random.default_rng(int(seed))
        else:
            rng = np.random.default_rng()

        self._ep_rng = rng

        self.t = 0
        self.pending_orders = []
        self.idle_riders = []
        self.rider_pos = {}
        self.busy_riders = []
        self.busy_until = {}

        orders = list(self.orders)
        riders = list(self.riders)

        rng.shuffle(orders)
        rng.shuffle(riders)

        self.orders_ep = orders
        self.riders_ep = riders

        orders_by_t = {}
        for o in self.orders_ep:
            t0 = int(o["t_min"])
            orders_by_t.setdefault(t0, []).append(o)
        self._orders_by_t = orders_by_t

        riders_by_t = {}
        for r in self.riders_ep:
            t0 = int(r["start_min"])
            riders_by_t.setdefault(t0, []).append(r)
        self._riders_by_t = riders_by_t

        return self._get_obs(), {}

    def _eta_min(self, a: int, b: int, speed_kmph: float = 18.0) -> float:
        (x1, y1) = self.node_xy[a]
        (x2, y2) = self.node_xy[b]
        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return (dist / speed_kmph) * 60.0

    def _edge_eta_time_dependent_lookup(self, u: int, v: int, t_min: int) -> float:
        """
        CSV lookup only (no GNN).
        Finds the latest time bucket <= t_min for edge (u,v).
        """
        key = (u, v)
        entries = self.travel_time.get(key)
        if not entries:
            # fallback if missing
            return 9999.0

        # entries: list[(t_bucket, tt_min)] sorted by t_bucket asc
        # choose latest bucket not exceeding t_min
        best_tt = entries[0][1]
        for tb, tt in entries:
            if tb <= t_min:
                best_tt = tt
            else:
                break
        return float(best_tt)

    def _edge_eta_time_dependent(self, u: int, v: int, t_min: int) -> float:
        # baseline: CSV lookup
        if not getattr(self, "use_gnn_eta", False):
            return float(self._edge_eta_time_dependent_lookup(u, v, t_min))

        # GNN: raw -> calibrated
        raw = float(self._edge_eta_gnn_raw(u, v, t_min))

        bucket_size = 45
        b = int((t_min // bucket_size) * bucket_size)
        scale = getattr(self, "gnn_bucket_scale", {}).get(b, self.gnn_alpha)

        edge_s = getattr(self, "edge_scale", {}).get((u, v), 1.0)
        eta = scale * raw * edge_s

        return max(0.1, float(eta))




    def _edge_eta_gnn_raw(self, u: int, v: int, t_min: int) -> float:
        import torch

        t_feat = torch.tensor(
            [[t_min / max(1.0, float(self.duration))]],
            dtype=torch.float32,
            device=self.torch_device,
        )
        uv = torch.tensor([[u, v]], dtype=torch.long, device=self.torch_device)

        with torch.no_grad():
            pred = self.gnn_model(self.gnn_data.x, self.gnn_data.edge_index, uv, t_feat)
        return float(pred.item())



    def _init_gnn_eta(self) -> None:
        import torch
        from torch_geometric.data import Data
        from models.eta_gnn import EtaGNN

        # device
        self.torch_device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # node features (x,y)
        num_nodes = max(self.node_xy.keys()) + 1
        x = torch.zeros((num_nodes, 2), dtype=torch.float32)
        for nid, (xx, yy) in self.node_xy.items():
            x[nid, 0] = float(xx)
            x[nid, 1] = float(yy)

        # edge_index from graph
        us, vs = [], []
        for (u, v) in self.G.edges():
            us.append(int(u))
            vs.append(int(v))
        edge_index = torch.tensor([us, vs], dtype=torch.long)

        self.gnn_data = Data(x=x, edge_index=edge_index).to(self.torch_device)

        self.gnn_model = EtaGNN(in_dim=2, hid_dim=64).to(self.torch_device)
        self.gnn_model.load_state_dict(
            torch.load(self.gnn_ckpt, map_location=self.torch_device)
        )
        self.gnn_model.eval()

    def _fit_gnn_calibration(self) -> None:
        import random
        from collections import defaultdict

        keys = list(self.travel_time.keys())
        if not keys:
            return

        bucket_size = 45
        T_choices = list(range(0, int(self.duration), 15))
        random.shuffle(keys)

        # -------- pass 1: fit bucket scales --------
        b_num = defaultdict(float)
        b_den = defaultdict(float)

        samples = []
        n = 0
        for (u, v) in keys:
            for t in T_choices:
                y_true = float(self._edge_eta_time_dependent_lookup(u, v, t))
                y_pred = float(self._edge_eta_gnn_raw(u, v, t))
                b = int((t // bucket_size) * bucket_size)

                b_num[b] += y_pred * y_true
                b_den[b] += y_pred * y_pred

                samples.append((u, v, t, y_true, y_pred))
                n += 1
                if n >= int(self.calib_samples):
                    break
            if n >= int(self.calib_samples):
                break

        self.gnn_bucket_scale = {}
        for b, den in b_den.items():
            scale = 1.0 if den < 1e-9 else (b_num[b] / den)
            if scale <= 0:
                scale = 1.0
            self.gnn_bucket_scale[int(b)] = float(scale)

        # fallback alpha
        scales = list(self.gnn_bucket_scale.values())
        self.gnn_alpha = float(sum(scales) / max(1, len(scales)))
        self.gnn_beta = 0.0

        # -------- pass 2: fit edge scales using bucket-scaled preds --------
        e_num = defaultdict(float)
        e_den = defaultdict(float)

        for (u, v, t, y_true, y_pred) in samples:
            b = int((t // bucket_size) * bucket_size)
            scale_t = self.gnn_bucket_scale.get(b, self.gnn_alpha)

            e_num[(u, v)] += y_true
            e_den[(u, v)] += (scale_t * y_pred)

        self.edge_scale = {}
        for k, den in e_den.items():
            if den < 1e-9:
                continue
            s = e_num[k] / den
            if s < 0.5:
                s = 0.5
            if s > 2.0:
                s = 2.0
            self.edge_scale[k] = float(s)

        if self.verbose:
            print(
                f"[GNN-CALIB] bucket_scales={self.gnn_bucket_scale} "
                f"(fallback alpha={self.gnn_alpha:.4f}) using {len(samples)} samples | "
                f"edge_scales: {len(self.edge_scale)} edges"
            )
    


    def _shortest_eta(self, src: int, dst: int, t: int) -> float:
        """
        Time-dependent shortest path ETA from src to dst at time t.
        Edge weight is travel time from self.travel_time lookup.
        """
        if src == dst:
            return 0.0

        def w(u: int, v: int, d: Dict[str, Any]) -> float:
            return float(self._edge_eta_time_dependent(u, v, t))

        try:
            return float(nx.shortest_path_length(self.G, src, dst, weight=w, method="dijkstra"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # fallback to Euclidean if disconnected
            return float(self._eta_min(src, dst))

    def _get_obs(self):
        return np.array(
            [self.t, len(self.pending_orders), len(self.idle_riders)],
            dtype=np.float32,
        )
    def step(self, action: int):
        if self._orders_by_t is None or self._riders_by_t is None:
            self.reset()

        reward = 0.0

        self.t += 1

        newly_idle = []
        for r in self.busy_riders:
            rid = r["rider_id"]
            if self.busy_until.get(rid, 10**9) <= self.t:
                newly_idle.append(r)

        if newly_idle:
            self.busy_riders = [r for r in self.busy_riders if r not in newly_idle]
            self.idle_riders.extend(newly_idle)

        for o in self._orders_by_t.get(self.t, []):
            self.pending_orders.append(o)

        for r in self._riders_by_t.get(self.t, []):
            self.idle_riders.append(r)
            self.rider_pos[r["rider_id"]] = r["init_node"]

        if action == 1:
            if self.pending_orders and self.idle_riders:
                order = self.pending_orders.pop(0)
                rider = self.idle_riders.pop(0)

                rid = rider["rider_id"]
                rider_node = self.rider_pos[rid]
                t_now = self.t

                eta_to_pickup = self._shortest_eta(rider_node, order["origin"], t_now) / rider["speed_factor"]
                eta_to_dropoff = self._shortest_eta(order["origin"], order["dest"], t_now) / rider["speed_factor"]

                reward -= 2.0 * (eta_to_pickup + eta_to_dropoff)

                self.rider_pos[rid] = order["dest"]

                import math
                total_eta = eta_to_pickup + eta_to_dropoff
                self.busy_until[rid] = self.t + int(math.ceil(total_eta))
                self.busy_riders.append(rider)
            else:
                reward -= 0.1

        reward -= 0.05 * float(len(self.pending_orders))

        terminated = self.t >= self.duration
        truncated = False
        info = {}

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info