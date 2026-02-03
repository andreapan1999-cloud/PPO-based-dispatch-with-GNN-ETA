# models/dispatch_bipartite_gnn.py
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


class DispatchBiGNN(nn.Module):
    """
    Hetero bipartite GNN:
      nodes: rider, order
      edges: rider->order candidate edges
    Output:
      score for each candidate edge (lower = better)
    """

    def __init__(self, rider_dim: int, order_dim: int, hidden: int = 64):
        super().__init__()

        self.rider_mlp = nn.Sequential(
            nn.Linear(rider_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.order_mlp = nn.Sequential(
            nn.Linear(order_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        self.conv1 = HeteroConv(
            {
                ("rider", "to", "order"): SAGEConv((hidden, hidden), hidden),
                ("order", "rev_to", "rider"): SAGEConv((hidden, hidden), hidden),
            },
            aggr="sum",
        )
        self.conv2 = HeteroConv(
            {
                ("rider", "to", "order"): SAGEConv((hidden, hidden), hidden),
                ("order", "rev_to", "rider"): SAGEConv((hidden, hidden), hidden),
            },
            aggr="sum",
        )

        # edge scorer: concat rider_emb + order_emb -> score
        self.scorer = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        xr = self.rider_mlp(data["rider"].x)
        xo = self.order_mlp(data["order"].x)

        x_dict = {"rider": xr, "order": xo}

        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        # score edges of type rider->order
        edge_index = data[("rider", "to", "order")].edge_index
        src = edge_index[0]
        dst = edge_index[1]
        h = torch.cat([x_dict["rider"][src], x_dict["order"][dst]], dim=-1)
        score = self.scorer(h).squeeze(-1)  # [num_edges]
        return score