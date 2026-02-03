import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class EtaGNN(nn.Module):
    """
    Minimal edge regression:
    - Node encoder: GraphSAGE over the road graph
    - Edge regressor: predicts travel_time_min for (u,v,t)
    """

    def __init__(self, in_dim: int, hid_dim: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, hid_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(hid_dim * 2 + 1, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1),
        )

    def forward(self, x, edge_index, uv_pairs, t_feat):
        # x: [N, in_dim]
        # edge_index: [2, E]
        # uv_pairs: [B, 2] (u,v)
        # t_feat: [B, 1] normalized t in [0,1]
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()

        u = uv_pairs[:, 0]
        v = uv_pairs[:, 1]
        z = torch.cat([h[u], h[v], t_feat], dim=-1)
        y_hat = self.edge_mlp(z).squeeze(-1)  # [B]
        return y_hat
