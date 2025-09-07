# data_preparation.py
# 支持 Kaggle Polymer Dataset + QM9 Dataset
# 输出格式兼容 WD-MPNN model.py
# PyTorch Geometric >= 2.3

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import rdPartialCharges
from torch_geometric.datasets import QM9

# -----------------------------
# 通用图增强 transform
# -----------------------------

def graph_augment(data, node_drop_prob=0.0, edge_drop_prob=0.0, noise_std=0.0):
    """简单的图增强：随机节点mask、边dropout、全局特征扰动"""
    if node_drop_prob > 0 and data.x is not None:
        mask = torch.rand(data.x.size(0)) > node_drop_prob
        data.x = data.x.clone()
        data.x[~mask] = 0.0

    if edge_drop_prob > 0 and data.edge_index is not None:
        E = data.edge_index.size(1)
        keep = torch.rand(E) > edge_drop_prob
        data.edge_index = data.edge_index[:, keep]
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[keep]

    if noise_std > 0 and hasattr(data, "global_feats"):
        data.global_feats = data.global_feats + noise_std * torch.randn_like(data.global_feats)

    return data

# -----------------------------
# Kaggle Polymer Dataset
# -----------------------------

TARGETS = ["Tg", "FFV", "Tc", "Density", "Rg"]

def make_smile_canonical(smile):
    """清洗并标准化 SMILES"""
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return np.nan
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return np.nan

def create_graph_from_smiles(smiles):
    """SMILES -> Graph (x, edge_index, edge_attr)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # 计算 Gasteiger 电荷（可选）
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    # 节点特征
    x_feats = []
    for atom in mol.GetAtoms():
        x_feats.append([
            atom.GetAtomicNum(),
            int(atom.GetIsAromatic()),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            atom.GetDegree(),
            atom.GetTotalNumHs(),
            int(atom.IsInRing()),
            atom.GetMass(),
        ])
    x = torch.tensor(x_feats, dtype=torch.float)

    # 边特征
    e_idx, e_feats = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        feats = [
            b.GetBondTypeAsDouble(),
            int(b.GetIsConjugated()),
            int(b.IsInRing()),
            float(b.GetStereo()),
        ]
        e_idx += [[i, j], [j, i]]
        e_feats += [feats, feats]

    edge_index = torch.tensor(e_idx, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(e_feats, dtype=torch.float)

    return x, edge_index, edge_attr

class PolymerDataset(InMemoryDataset):
    """Kaggle Polymer Dataset"""

    def __init__(self, df: pd.DataFrame, transform=None):
        super().__init__(None, transform)
        self.data_list = []
        for _, row in df.iterrows():
            smi = row["SMILES"]
            try:
                x, ei, ea = create_graph_from_smiles(smi)
            except Exception as e:
                continue
            y_vals, mask_vals = [], []
            for t in TARGETS:
                val = row.get(t, np.nan)
                if pd.isna(val):
                    y_vals.append(0.0)
                    mask_vals.append(0.0)
                else:
                    y_vals.append(float(val))
                    mask_vals.append(1.0)
            data = Data(
                x=x,
                edge_index=ei,
                edge_attr=ea,
                y=torch.tensor(y_vals, dtype=torch.float),
                mask=torch.tensor(mask_vals, dtype=torch.float),
                smiles=smi,
            )
            self.data_list.append(data)

    def len(self): return len(self.data_list)
    def get(self, idx): return self.data_list[idx]

def load_polymer_dataset(train_csv, batch_size=32, transform=None):
    """加载 Kaggle Polymer Dataset"""
    df = pd.read_csv(train_csv)
    df["SMILES"] = df["SMILES"].apply(make_smile_canonical)
    df = df[df["SMILES"].notnull()].reset_index(drop=True)
    dataset = PolymerDataset(df, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset

# -----------------------------
# QM9 Dataset
# -----------------------------

QM9_TASKS = [
    "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "U0", "U", "H",
    "G", "Cv", "u0_atom", "u_atom", "h_atom", "g_atom", "A", "B", "C"
]

def load_qm9_dataset(root="data/QM9", batch_size=64, transform=None):
    """加载 QM9 数据集"""
    dataset = QM9(root=root, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset

# -----------------------------
# main 测试
# -----------------------------

if __name__ == "__main__":
    # Kaggle Polymer
    print("📂 Load Polymer Dataset")
    poly_loader, poly_dataset = load_polymer_dataset("train.csv", batch_size=4, transform=graph_augment)
    batch = next(iter(poly_loader))
    print(f"Polymer batch: x={batch.x.shape}, edge_index={batch.edge_index.shape}, y={batch.y.shape}, mask={batch.mask.shape}")

    # QM9
    print("\n📂 Load QM9 Dataset")
    qm9_loader, qm9_dataset = load_qm9_dataset(root="data/QM9", batch_size=4, transform=graph_augment)
    batch = next(iter(qm9_loader))
    print(f"QM9 batch: x={batch.x.shape}, edge_index={batch.edge_index.shape}, y={batch.y.shape}")
