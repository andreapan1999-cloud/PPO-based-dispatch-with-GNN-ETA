import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from models.eta_gnn import EtaGNN


def infer_num_nodes(nodes_csv: Path) -> int:
    mx = -1
    with nodes_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            mx = max(mx, int(row["node_id"]))
    return mx + 1


def load_node_xy(nodes_csv: Path, num_nodes: int) -> torch.Tensor:
    xy = torch.zeros((num_nodes, 2), dtype=torch.float32)
    with nodes_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames or []
        if "x" in fields and "y" in fields:
            for row in r:
                nid = int(row["node_id"])
                xy[nid, 0] = float(row["x"])
                xy[nid, 1] = float(row["y"])
        else:
            # fallback: normalized node id
            nid = torch.arange(num_nodes, dtype=torch.float32).unsqueeze(-1)
            xy = torch.cat([nid / max(1.0, float(num_nodes - 1)), torch.zeros_like(nid)], dim=-1)
    return xy


def load_edge_index(edges_csv: Path) -> torch.Tensor:
    us, vs = [], []
    with edges_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            us.append(int(row["u"]))
            vs.append(int(row["v"]))
    return torch.tensor([us, vs], dtype=torch.long)


def load_edge_samples(travel_times_csv: Path):
    # returns uv [M,2], t [M,1], y [M]
    uvs, ts, ys = [], [], []
    with travel_times_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            uvs.append((int(row["u"]), int(row["v"])))
            ts.append(int(row["t_min"]))
            ys.append(float(row["travel_time_min"]))
    uv = torch.tensor(uvs, dtype=torch.long)
    t = torch.tensor(ts, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(ys, dtype=torch.float32)
    return uv, t, y


def main():
    data_dir = Path("outputs")
    nodes_csv = data_dir / "nodes.csv"
    edges_csv = data_dir / "edges.csv"
    tt_csv = data_dir / "travel_times.csv"

    assert nodes_csv.exists() and edges_csv.exists() and tt_csv.exists(), "Missing outputs/*.csv; run data_gen first."

    num_nodes = infer_num_nodes(nodes_csv)
    x = load_node_xy(nodes_csv, num_nodes)
    edge_index = load_edge_index(edges_csv)
    uv, t, y = load_edge_samples(tt_csv)

    # normalize time to [0,1]
    t = t / max(1.0, float(t.max().item()))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = Data(x=x, edge_index=edge_index).to(device)
    uv = uv.to(device)
    t = t.to(device)
    y = y.to(device)

    model = EtaGNN(in_dim=data.x.size(-1), hid_dim=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train/val split
    M = y.numel()
    perm = torch.randperm(M, device=device)
    split = int(0.8 * M)
    tr, va = perm[:split], perm[split:]

    for epoch in range(1, 201):
        model.train()
        pred = model(data.x, data.edge_index, uv[tr], t[tr])
        loss = F.mse_loss(pred, y[tr])
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                vpred = model(data.x, data.edge_index, uv[va], t[va])
                vloss = F.mse_loss(vpred, y[va]).item()
            print(f"epoch {epoch:03d} train_mse={loss.item():.4f} val_mse={vloss:.4f}")

    out = Path("outputs") / "eta_gnn.pt"
    torch.save(model.state_dict(), out)
    print("Saved:", out)


if __name__ == "__main__":
    main()
