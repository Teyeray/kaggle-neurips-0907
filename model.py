# model.py
# WD-MPNN encoder + multi-task head + official wMAE loss + checkpoint utils
# with Dropout & optional Label Noise
# PyTorch Geometric >= 2.3

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool

EPS = 1e-12

# ---------- Utils ----------

def compute_task_weights(tasks: List[str],
                         n_dict: Dict[str, Union[int, float]],
                         r_dict: Dict[str, Union[int, float]]) -> Dict[str, float]:
    """
    Compute per-task weights w_i according to the competition's official formula:

        w_i = (1 / r_i) * ( K * sqrt(1/n_i) / sum_j sqrt(1/n_j) )

    Args:
      tasks: list of task names
      n_dict: {task: n_i} number of samples for each task
      r_dict: {task: r_i} value range for each task
    Returns:
      dict {task: w_i}
    """
    K = len(tasks)
    inv_sqrt_ns = []
    for t in tasks:
        n_i = max(float(n_dict[t]), EPS)
        inv_sqrt_ns.append((1.0 / n_i) ** 0.5)
    denom = max(sum(inv_sqrt_ns), EPS)

    weights: Dict[str, float] = {}
    for idx, t in enumerate(tasks):
        r_i = max(float(r_dict[t]), EPS)
        inv_r = 1.0 / r_i
        inv_sqrt_n = inv_sqrt_ns[idx]
        w_i = inv_r * (K * inv_sqrt_n / denom)
        weights[t] = float(w_i)
    return weights

# ---------- Encoder ----------

class WDMPNNEncoder(MessagePassing):
    """
    Weighted Directed Message Passing Neural Network (wD-MPNN).
    Edge-centered message passing + node update + graph pooling.
    """

    def __init__(self,
                node_feat_dim: int,
                edge_feat_dim: int,
                hidden_dim: int,
                num_edge_layers: int = 3,
                pool: str = "mean"):
        super().__init__(aggr='add')
        assert pool in {"mean", "add"}
        self.hidden_dim = hidden_dim
        self.num_edge_layers = num_edge_layers
        self.pool = pool

        # initial edge hidden
        self.W_i = nn.Linear(node_feat_dim + edge_feat_dim, hidden_dim)
        # edge update
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        # node update
        self.W_o = nn.Linear(node_feat_dim + hidden_dim, hidden_dim)

        self.edge_ln = nn.LayerNorm(hidden_dim)
        self.node_ln = nn.LayerNorm(hidden_dim)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        src, dst = edge_index
        if edge_weight is None:
            edge_weight = torch.ones(edge_attr.size(0), device=x.device, dtype=x.dtype)

        # init edge hidden
        h = torch.cat([x[src], edge_attr], dim=-1)
        h = F.relu(self.W_i(h))
        h = self.edge_ln(h)

        # edge updates
        for _ in range(self.num_edge_layers):
            m = torch.zeros_like(h)
            m = m.index_add(0, dst, edge_weight.unsqueeze(-1) * self.W_h(h))
            h = F.relu(h + m)
            h = self.edge_ln(h)

        # node update
        agg = torch.zeros(x.size(0), self.hidden_dim, device=x.device, dtype=x.dtype)
        agg = agg.index_add(0, dst, edge_weight.unsqueeze(-1) * h)
        node_input = torch.cat([x, agg], dim=-1)
        node_hidden = F.relu(self.W_o(node_input))
        node_hidden = self.node_ln(node_hidden)

        # pooling
        if self.pool == "mean":
            graph_repr = global_mean_pool(node_hidden, batch)
        else:
            graph_repr = global_add_pool(node_hidden, batch)
        return graph_repr

# ---------- Heads ----------

class MultiTaskHead(nn.Module):
    """
    Multi-task regression head.
    If tasks is None -> single-task: returns Tensor[B]
    If tasks is list -> per-task heads: returns Dict[str, Tensor[B]]
    """

    def __init__(self,
                 hidden_dim: int,
                 mlp_hidden: List[int],
                 tasks: Optional[List[str]] = None,
                 output_dim: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.tasks = tasks

        def build_mlp() -> nn.Sequential:
            layers: List[nn.Module] = []
            in_dim = hidden_dim
            for h in mlp_hidden:
                layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
                in_dim = h
            layers.append(nn.Linear(in_dim, output_dim))
            return nn.Sequential(*layers)

        if tasks is None:
            self.head = build_mlp()
        else:
            self.heads = nn.ModuleDict({t: build_mlp() for t in tasks})

    def forward(self, graph_repr: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if hasattr(self, "head"):
            return self.head(graph_repr).squeeze(-1)
        else:
            return {t: self.heads[t](graph_repr).squeeze(-1) for t in self.tasks}

# ---------- Full Model ----------

class WDMPNNModel(nn.Module):
    """Full WD-MPNN with official wMAE loss + Dropout + Label Noise."""

    def __init__(self,
                 node_feat_dim: int,
                 edge_feat_dim: int,
                 hidden_dim: int,
                 mlp_hidden: List[int],
                 tasks: Optional[List[str]] = None,
                 output_dim: int = 1,
                 pool: str = "mean",
                 num_edge_layers: int = 3,
                 dropout: float = 0.1,
                 use_label_noise: bool = False,
                 label_noise_scale: float = 0.01):
        super().__init__()
        self.tasks = tasks if tasks is not None else []
        self.encoder = WDMPNNEncoder(node_feat_dim, edge_feat_dim,
                                     hidden_dim, num_edge_layers, pool)
        self.head = MultiTaskHead(hidden_dim, mlp_hidden, tasks,
                                  output_dim=output_dim, dropout=dropout)

        self._w: Dict[str, float] = {t: 1.0 for t in self.tasks}
        self.use_label_noise = use_label_noise
        self.label_noise_scale = label_noise_scale

    def forward(self, data) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        edge_weight = getattr(data, "edge_weight", None)
        g = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch, edge_weight=edge_weight)
        return self.head(g)

    # ----- task weights -----
    def set_task_stats(self,
                       n_dict: Dict[str, Union[int, float]],
                       r_dict: Dict[str, Union[int, float]]) -> None:
        if not self.tasks: return
        self._w = compute_task_weights(self.tasks, n_dict, r_dict)

    def get_task_weights(self) -> Dict[str, float]:
        return dict(self._w)

    # ----- loss -----
    def compute_wmae_loss(self,
                          outputs: Dict[str, torch.Tensor],
                          targets: Dict[str, torch.Tensor],
                          mask: Optional[Dict[str, torch.Tensor]] = None
                          ) -> Tuple[torch.Tensor, Dict[str, float]]:
        some_task = self.tasks[0]
        B = outputs[some_task].shape[0]
        total = outputs[some_task].new_tensor(0.0)
        per_task_mae: Dict[str, float] = {}
        w = self._w

        for t in self.tasks:
            yhat, y = outputs[t], targets[t]

            # label noise injection
            if self.use_label_noise:
                noise = self.label_noise_scale * y.std() * torch.randn_like(y)
                y = y + noise

            if mask is not None:
                m = mask[t].bool()
                if m.sum() == 0:
                    per_task_mae[t] = float("nan")
                    continue
                abs_err = (yhat[m] - y[m]).abs()
                mae_t = abs_err.mean()
                contrib = w[t] * abs_err.sum()
            else:
                abs_err = (yhat - y).abs()
                mae_t = abs_err.mean()
                contrib = w[t] * abs_err.sum()

            per_task_mae[t] = mae_t.item()
            total = total + contrib

        loss = total / max(B, 1)
        return loss, per_task_mae

    # ----- checkpoint utils -----
    def save_checkpoint(self, optimizer, path: str = "checkpoint.pt"):
        torch.save({
            "model_state": self.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, path)

    def load_checkpoint(self, optimizer, path: str = "checkpoint.pt"):
        ckpt = torch.load(path)
        self.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])

    # ----- helper -----
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self):
        return {
            "tasks": self.tasks,
            "weights": self._w,
            "use_label_noise": self.use_label_noise,
            "label_noise_scale": self.label_noise_scale
        }

__all__ = [
    "WDMPNNEncoder",
    "MultiTaskHead",
    "WDMPNNModel",
    "compute_task_weights"
]
