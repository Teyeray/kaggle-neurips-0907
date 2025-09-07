# train.py
# Notebook-friendly training utils
# No auto-execution, only functions

import torch
import torch.optim as optim
from typing import Dict, List
from model import WDMPNNModel
from data_preparation import TARGETS, QM9_TASKS

# -----------------------------
# 核心训练函数
# -----------------------------

def train_one_epoch(model, loader, optimizer, device, tasks, use_mask=True):
    model.train()
    total_loss, per_task_mae, count = 0.0, {t: 0.0 for t in tasks}, 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)

        if hasattr(batch, "mask") and use_mask:
            mask = {t: batch.mask[:, i] for i, t in enumerate(tasks)}
            targets = {t: batch.y[:, i] for i, t in enumerate(tasks)}
        else:
            mask = None
            targets = {t: batch.y[:, i] for i, t in enumerate(tasks)}

        loss, mae_dict = model.compute_wmae_loss(outputs, targets, mask=mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for t in mae_dict:
            if mae_dict[t] == mae_dict[t]:
                per_task_mae[t] += mae_dict[t]
        count += 1

    avg_loss = total_loss / max(count, 1)
    for t in per_task_mae: per_task_mae[t] /= max(count, 1)
    return avg_loss, per_task_mae


def evaluate(model, loader, device, tasks, use_mask=True):
    model.eval()
    total_loss, per_task_mae, count = 0.0, {t: 0.0 for t in tasks}, 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)

            if hasattr(batch, "mask") and use_mask:
                mask = {t: batch.mask[:, i] for i, t in enumerate(tasks)}
                targets = {t: batch.y[:, i] for i, t in enumerate(tasks)}
            else:
                mask = None
                targets = {t: batch.y[:, i] for i, t in enumerate(tasks)}

            loss, mae_dict = model.compute_wmae_loss(outputs, targets, mask=mask)
            total_loss += loss.item()
            for t in mae_dict:
                if mae_dict[t] == mae_dict[t]:
                    per_task_mae[t] += mae_dict[t]
            count += 1

    avg_loss = total_loss / max(count, 1)
    for t in per_task_mae: per_task_mae[t] /= max(count, 1)
    return avg_loss, per_task_mae

# -----------------------------
# 高层接口
# -----------------------------

def fit(model, train_loader, val_loader, optimizer, device, tasks, epochs=10, use_mask=True):
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, epochs + 1):
        tr_loss, tr_mae = train_one_epoch(model, train_loader, optimizer, device, tasks, use_mask)
        val_loss, val_mae = evaluate(model, val_loader, device, tasks, use_mask)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        print(f"[Epoch {epoch}] Train {tr_loss:.4f}, Val {val_loss:.4f}")
    return history

# -----------------------------
# Pretrain & Finetune
# -----------------------------

def pretrain_qm9(loader, dataset, hidden_dim=128, lr=1e-3, epochs=5, device="cuda"):
    tasks = QM9_TASKS
    node_dim, edge_dim = dataset.num_node_features, dataset.num_edge_features

    model = WDMPNNModel(node_feat_dim=node_dim,
                        edge_feat_dim=edge_dim,
                        hidden_dim=hidden_dim,
                        mlp_hidden=[128, 64],
                        tasks=tasks).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = fit(model, loader, loader, optimizer, device, tasks, epochs, use_mask=False)
    return model, history


def finetune_polymer(loader, dataset, hidden_dim=128, lr=1e-3, epochs=5, device="cuda", pretrained_model=None):
    tasks = TARGETS
    node_dim, edge_dim = dataset[0].x.size(-1), dataset[0].edge_attr.size(-1)

    if pretrained_model is not None:
        model = pretrained_model
        model.tasks = tasks
    else:
        model = WDMPNNModel(node_feat_dim=node_dim,
                            edge_feat_dim=edge_dim,
                            hidden_dim=hidden_dim,
                            mlp_hidden=[128, 64],
                            tasks=tasks).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = fit(model, loader, loader, optimizer, device, tasks, epochs, use_mask=True)
    return model, history
