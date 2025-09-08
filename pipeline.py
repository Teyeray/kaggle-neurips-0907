#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

sys.path.append("kaggle/input/neurips-open-polymer-prediction-2025")
base_path = "kaggle/input/neurips-open-polymer-prediction-2025/"
supplement_path = "kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/"
tc_smiles_path = "kaggle/input/tc-smiles/"
smiles_extra_data_path = "kaggle/input/smiles-extra-data/"


# In[2]:


import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import optuna
import wandb

from data_preparation import load_qm9_dataset
from train import pretrain_qm9


# In[3]:


from torch_geometric.datasets import QM9

dataset = QM9(root="data/qm9")
print(dataset)


# In[4]:


# -----------------------------
# 环境设置
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Training on {device}")


# In[5]:


# -----------------------------
# 数据加载
# -----------------------------
loader, dataset = load_qm9_dataset(root="data/qm9", batch_size=8172)
n = len(dataset)
train_set, val_set = random_split(dataset, [int(0.9*n), n - int(0.9*n)])
train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
val_loader = DataLoader(val_set, batch_size=256)


# In[6]:


# -----------------------------
# Optuna 目标函数
# -----------------------------
def objective(trial):
    ckpt_path = f"checkpoints/trial_{trial.number}_best.pt"
    model, hist = pretrain_qm9(
        train_loader=train_loader,
        val_loader=val_loader,
        dataset=dataset,
        hidden_dim=trial.suggest_categorical("hidden_dim", [128, 256, 512, 1024]),
        lr=trial.suggest_loguniform("lr", 1e-4, 1e-2),
        num_edge_layers=trial.suggest_int("num_edge_layers", 2, 32),
        dropout=trial.suggest_float("dropout", 0.0, 0.3),
        epochs=35,
        device=device,
        run_wandb=False,
        ckpt_path=ckpt_path
    )
    return min(hist["val_loss"])


# In[7]:


import optuna

storage_path = "sqlite:///optuna_qm9.db"  # 保存在当前目录
study = optuna.create_study(
    direction="minimize",
    study_name="qm9_pretrain",
    storage=storage_path,
    load_if_exists=True
)


# In[8]:


# -----------------------------
# 运行超参搜索
# -----------------------------
study.optimize(objective, n_trials=30)

print("✅ 最优超参数:", study.best_trial.params)
print("📉 最优验证集 loss:", study.best_value)


# In[ ]:





# In[ ]:




