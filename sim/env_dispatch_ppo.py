# sim/env_dispatch_ppo.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from sim.env_dispatch import DispatchEnv


class DispatchPPOEnv(gym.Env):
    """
    PPO-friendly dispatch env.

    Action:
      0 = do nothing
      1..(R*O) = choose (r_i, o_j) in a fixed candidate matrix

    Observation (flat vector):
      [t_norm, pending_cnt_norm, idle_cnt_norm,
       rider_feats (R*4), order_feats (O*5)]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_dir: str = "outputs",
        duration: int = 180,
        use_gnn_eta: bool = True,
        calibrate_gnn: bool = True,
        calib_samples: int = 2000,
        R: int = 4,
        O: int = 4,
        invalid_action_penalty: float = -2.0,
        wait_penalty: float = -0.1,
    ):
        super().__init__()

        self.base = DispatchEnv(
            data_dir=data_dir,
            duration=duration,
            use_gnn_eta=use_gnn_eta,
            calibrate_gnn=calibrate_gnn,
            calib_samples=calib_samples,
            verbose=False,
        )

        self.R = int(R)
        self.O = int(O)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.wait_penalty = float(wait_penalty)

        # action: 0..R*O
        self.action_space = spaces.Discrete(self.R * self.O + 1)

        obs_dim = 3 + self.R * 4 + self.O * 5
        self.observation_space = spaces.Box(
            low=-1e9,
            high=1e9,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    # ---------- feature helpers ----------
    def _rider_feat(self, rider: Dict[str, Any], t: int) -> List[float]:
        rid = int(rider["rider_id"])
        node = int(self.base.rider_pos.get(rid, int(rider["init_node"])))
        x, y = self.base.node_xy[node]
        return [
            t / max(1.0, float(self.base.duration)),
            float(x),
            float(y),
            float(rider["speed_factor"]),
        ]

    def _order_feat(self, order: Dict[str, Any], t: int) -> List[float]:
        age = max(0, t - int(order["t_min"]))
        ox, oy = self.base.node_xy[int(order["origin"])]
        dx, dy = self.base.node_xy[int(order["dest"])]
        return [
            age / 180.0,
            float(ox),
            float(oy),
            float(dx),
            float(dy),
        ]

    def _get_candidates(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        riders = self.base.idle_riders[: self.R]
        orders = self.base.pending_orders[: self.O]
        return riders, orders

    def _obs(self) -> np.ndarray:
        t = int(self.base.t)

        riders, orders = self._get_candidates()

        # top-level scalars
        t_norm = t / max(1.0, float(self.base.duration))
        p_norm = len(self.base.pending_orders) / 100.0
        i_norm = len(self.base.idle_riders) / 100.0

        vec: List[float] = [t_norm, p_norm, i_norm]

        # pad riders
        for k in range(self.R):
            if k < len(riders):
                vec.extend(self._rider_feat(riders[k], t))
            else:
                vec.extend([0.0, 0.0, 0.0, 0.0])

        # pad orders
        for k in range(self.O):
            if k < len(orders):
                vec.extend(self._order_feat(orders[k], t))
            else:
                vec.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        return np.asarray(vec, dtype=np.float32)

    # ---------- gym API ----------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.base.reset(seed=seed)
        return self._obs(), info

    def step(self, action: int):
        """
        We advance time by 1 minute each step (base.step will also advance).
        Dispatch happens at current time.
        """
        reward = 0.0

        riders, orders = self._get_candidates()

        if action == 0:
            pass
        else:
            idx = int(action) - 1
            ri = idx // self.O
            oi = idx % self.O

            if ri >= len(riders) or oi >= len(orders):
                reward += self.invalid_action_penalty
            else:
                rider = riders[ri]
                order = orders[oi]

                rid = int(rider["rider_id"])
                rider_node = int(self.base.rider_pos.get(rid, int(rider["init_node"])))

                o = int(order["origin"])
                d = int(order["dest"])

                # shortest path ETA with time dependency (already gnn-calibrated inside base)
                eta1 = float(self.base._shortest_eta(rider_node, o, self.base.t)) / float(rider["speed_factor"])
                eta2 = float(self.base._shortest_eta(o, d, self.base.t)) / float(rider["speed_factor"])
                total_eta = eta1 + eta2

                # perform assignment:
                # remove that order from pending_orders (exact object)
                # remove rider from idle
                try:
                    self.base.pending_orders.remove(order)
                except ValueError:
                    pass
                try:
                    self.base.idle_riders.remove(rider)
                except ValueError:
                    pass

                # mark rider busy until time t + ceil(total_eta)
                busy_until = int(self.base.t + max(1, int(np.ceil(total_eta))))
                self.base.busy_riders.append(rider)
                self.base.busy_until[rid] = busy_until
                self.base.rider_pos[rid] = d

                # reward shaping: encourage short travel
                reward += 1.0
                reward += -0.01 * float(total_eta)

        # waiting penalty for queue
        reward += self.wait_penalty * float(len(self.base.pending_orders))

        # advance time by one step using base dynamics (adds arrivals, releases busy riders, etc.)
        obs2, r2, terminated, truncated, info = self.base.step(0)

        # we ignore base reward r2 (keep PPO env reward definition consistent)
        return self._obs(), float(reward), bool(terminated), bool(truncated), info