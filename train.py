# train.py
# 通用训练脚本：支持 QM9 pretrain + Polymer finetune
# 只定义函数，不自动执行，方便 Notebook 使用

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from typing import Dict, List, Tuple

from model import WDMPNNModel
from data_preparation import load_polymer_dataset, load_qm9_dataset, TARGETS, QM9_TASKS

# -----------------------------
# 核心训练函数
# -----------------------------

def train_one_epoch(model, loader, optimizer, device, tasks, use_mask=True):
    model.train()
    total_loss = 0.0
    per_task_mae = {t: 0.0 for t in tasks}
    count = 0

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
            if mae_dict[t] == mae_dict[t]:  # 忽略 nan
                per_task_mae[t] += mae_dict[t]
        count += 1

    avg_loss = total_loss / max(count, 1)
    for t in per_task_mae:
        per_task_mae[t] /= max(count, 1)

    return avg_loss, per_task_mae


def evaluate(model, loader, device, tasks, use_mask=True):
    model.eval()
    total_loss = 0.0
    per_task_mae = {t: 0.0 for t in tasks}
    count = 0

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
    for t in per_task_mae:
        per_task_mae[t] /= max(count, 1)

    return avg_loss, per_task_mae

# -----------------------------
# 高层接口：fit
# -----------------------------

def fit(model, train_loader, val_loader, optimizer, device, tasks,
        epochs=10, use_mask=True, verbose=True):
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_mae = train_one_epoch(model, train_loader, optimizer, device, tasks, use_mask)
        val_loss, val_mae = evaluate(model, val_loader, device, tasks, use_mask)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        if verbose:
            print(f"[Epoch {epoch}] Train Loss={tr_loss:.4f}, Val Loss={val_loss:.4f}")
            print(f"  Train MAE: {tr_mae}")
            print(f"  Val   MAE: {val_mae}")

    return history

# -----------------------------
# Pretrain & Finetune
# -----------------------------

def pretrain_qm9(root="data/QM9", hidden_dim=128, batch_size=64, epochs=5, lr=1e-3, device="cuda"):
    train_loader, dataset = load_qm9_dataset(root=root, batch_size=batch_size)
    tasks = QM9_TASKS

    node_dim = dataset.num_node_features
    edge_dim = dataset.num_edge_features

    model = WDMPNNModel(
        node_feat_dim=node_dim,
        edge_feat_dim=edge_dim,
        hidden_dim=hidden_dim,
        mlp_hidden=[128, 64],
        tasks=tasks,
        dropout=0.2,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = fit(model, train_loader, train_loader, optimizer, device, tasks, epochs, use_mask=False)

    return model, history


def finetune_polymer(train_csv, hidden_dim=128, batch_size=32, epochs=5, lr=1e-3, device="cuda", pretrained_model=None):
    loader, dataset = load_polymer_dataset(train_csv, batch_size=batch_size)
    tasks = TARGETS

    # 假设和 QM9 用的模型结构一致
    node_dim = dataset[0].x.size(-1)
    edge_dim = dataset[0].edge_attr.size(-1)

    if pretrained_model is not None:
        model = pretrained_model
        model.tasks = tasks  # 切换任务
        model.head = model.head.__class__(hidden_dim, [128, 64], tasks=tasks, dropout=0.2).to(device)
    else:
        model = WDMPNNModel(
            node_feat_dim=node_dim,
            edge_feat_dim=edge_dim,
            hidden_dim=hidden_dim,
            mlp_hidden=[128, 64],
            tasks=tasks,
            dropout=0.2,
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = fit(model, loader, loader, optimizer, device, tasks, epochs, use_mask=True)

    return model, history
