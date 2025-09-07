# train.py
# 基础训练工具 (支持普通训练 + Optuna)
# 保存 & 加载最佳模型

import torch
import torch.optim as optim
from typing import Dict, List, Tuple
from tqdm import tqdm
import wandb
import os

from model import WDMPNNModel
from data_preparation import TARGETS, QM9_TASKS


# -----------------------------
# 单轮训练
# -----------------------------
def train_one_epoch(model, loader, optimizer, device, tasks, use_mask=True, epoch=None, epochs=None):
    model.train()
    total_loss, per_task_mae, count = 0.0, {t: 0.0 for t in tasks}, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
    for batch in pbar:
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
            if mae_dict[t] == mae_dict[t]:  # 过滤 nan
                per_task_mae[t] += mae_dict[t]
        count += 1

        gpu_mem = f"{torch.cuda.memory_allocated()/1024**2:.0f}MB" if device == "cuda" else "CPU"
        pbar.set_postfix(loss=loss.item(), gpu=gpu_mem)

    avg_loss = total_loss / max(count, 1)
    for t in per_task_mae: 
        per_task_mae[t] /= max(count, 1)
    return avg_loss, per_task_mae


# -----------------------------
# 验证
# -----------------------------
def evaluate(model, loader, device, tasks, use_mask=True, epoch=None, epochs=None):
    model.eval()
    total_loss, per_task_mae, count = 0.0, {t: 0.0 for t in tasks}, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False)
    with torch.no_grad():
        for batch in pbar:
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
# fit (带 early stopping + W&B + 可选保存路径)
# -----------------------------
def fit(model, train_loader, val_loader, optimizer, device, tasks, epochs=10,
        use_mask=True, patience=10, run_wandb=False, project="qm9-pretrain",
        print_all_tasks=False, ckpt_path=None):

    if run_wandb:
        wandb.init(project=project, config={
            "epochs": epochs,
            "tasks": tasks,
            "device": device,
            "params": sum(p.numel() for p in model.parameters() if p.requires_grad)
        })

    print(f"✅ Device: {device} | Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M | Tasks: {len(tasks)}")

    history = {"train_loss": [], "val_loss": []}
    best_val, patience_counter = float("inf"), 0

    # 保存路径：Optuna 传入 ckpt_path，否则默认
    save_path = ckpt_path if ckpt_path is not None else "checkpoint_best.pt"
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_mae = train_one_epoch(model, train_loader, optimizer, device, tasks, use_mask, epoch, epochs)
        val_loss, val_mae = evaluate(model, val_loader, device, tasks, use_mask, epoch, epochs)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        # 打印日志
        if print_all_tasks:
            mae_str = " | ".join([f"{t}:{val_mae[t]:.3f}" for t in tasks])
        else:
            mae_str = " | ".join([f"{t}:{val_mae[t]:.3f}" for t in list(tasks)[:3]])

        print(f"[Epoch {epoch}/{epochs}] Train {tr_loss:.3f}, Val {val_loss:.3f} | {mae_str}")

        # Early stopping + 保存最佳模型
        if val_loss < best_val:
            best_val, patience_counter = val_loss, 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered")
                break

    if run_wandb:
        wandb.finish()
    return history


# -----------------------------
# QM9 预训练
# -----------------------------
def pretrain_qm9(hidden_dim=128, lr=1e-3, epochs=5, device="cuda",
                 num_edge_layers=3, dropout=0.1, run_wandb=False,
                 train_loader=None, val_loader=None, dataset=None,
                 print_all_tasks=False, ckpt_path=None):

    tasks = QM9_TASKS
    node_dim, edge_dim = dataset.num_node_features, dataset.num_edge_features

    model = WDMPNNModel(
        node_feat_dim=node_dim,
        edge_feat_dim=edge_dim,
        hidden_dim=hidden_dim,
        mlp_hidden=[256, 128],
        tasks=tasks,
        num_edge_layers=num_edge_layers,
        dropout=dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = fit(model, train_loader, val_loader, optimizer, device, tasks,
                  epochs=epochs, use_mask=False, run_wandb=run_wandb,
                  project="qm9-pretrain",
                  print_all_tasks=print_all_tasks,
                  ckpt_path=ckpt_path)

    return model, history


# -----------------------------
# Polymer 微调
# -----------------------------
def finetune_polymer(train_loader, val_loader, dataset,
                     hidden_dim=128, lr=1e-3, epochs=20, device="cuda",
                     pretrained_model=None, num_edge_layers=3, dropout=0.1, run_wandb=True,
                     ckpt_path=None):

    tasks = TARGETS
    node_dim, edge_dim = dataset[0].x.size(-1), dataset[0].edge_attr.size(-1)

    if pretrained_model is not None:
        model = pretrained_model
        model.tasks = tasks
    else:
        model = WDMPNNModel(
            node_feat_dim=node_dim,
            edge_feat_dim=edge_dim,
            hidden_dim=hidden_dim,
            mlp_hidden=[256, 128],
            tasks=tasks,
            num_edge_layers=num_edge_layers,
            dropout=dropout
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = fit(model, train_loader, val_loader, optimizer, device, tasks,
                  epochs=epochs, use_mask=True, run_wandb=run_wandb,
                  project="polymer-finetune",
                  ckpt_path=ckpt_path)

    return model, history


# -----------------------------
# 加载最佳模型（统一接口）
# -----------------------------
def load_best_model(path: str, dataset, hidden_dim=128,
                    num_edge_layers=3, dropout=0.1, device="cuda", tasks=None):
    """
    从指定路径加载最佳模型
    - path: 保存的 checkpoint 文件路径
    - dataset: 用于获取 node_dim / edge_dim
    - hidden_dim, num_edge_layers, dropout: 构建模型所需参数（需和训练时一致）
    - tasks: 任务列表（默认 QM9_TASKS）
    """
    if tasks is None:
        tasks = QM9_TASKS

    node_dim = dataset.num_node_features
    edge_dim = dataset.num_edge_features

    model = WDMPNNModel(
        node_feat_dim=node_dim,
        edge_feat_dim=edge_dim,
        hidden_dim=hidden_dim,
        mlp_hidden=[256, 128],
        tasks=tasks,
        num_edge_layers=num_edge_layers,
        dropout=dropout
    ).to(device)

    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"✅ Loaded best model from {path}")
    return model
