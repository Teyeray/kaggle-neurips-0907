# data_preparation.py
# Notebook-friendly: no __main__, all functions callable
# Supports Kaggle Polymer Dataset + QM9 Dataset

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from rdkit import Chem
from pathlib import Path
from rdkit.Chem import rdPartialCharges
from torch_geometric.datasets import QM9


#输入数据
def add_extra_data(df_train: pd.DataFrame, df_extra: pd.DataFrame, target: str, source_name: str = "extra") -> pd.DataFrame:
    """
    将外部数据 merge 到 train：
    - 如果 train 里有相同 SMILES 且 target 缺失，用 extra 填补
    - 如果 extra 里有新 SMILES，追加到 train
    - 对同一个 SMILES target 出现多个值时取均值
    - 打印出重复 SMILES 的数量 & 差异
    """
    print(f"[INFO] Working on {source_name} (target={target})")
    before = len(df_train)

    # 只保留 SMILES + target
    df_extra = df_extra[['SMILES', target]].dropna(subset=[target]).copy()

    # 聚合外部数据，避免重复
    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()

    # 检查重复
    common = set(df_train['SMILES']) & set(df_extra['SMILES'])
    if common:
        merged_common = pd.merge(
            df_train[df_train['SMILES'].isin(common)][['SMILES', target]],
            df_extra[df_extra['SMILES'].isin(common)],
            on='SMILES',
            suffixes=('_train', '_extra')
        )
        diffs = (merged_common[f"{target}_train"] - merged_common[f"{target}_extra"]).abs()
        print(f"  重复 SMILES: {len(merged_common)}, 其中 {sum(diffs>1e-6)} 条数值不同 (mean diff={diffs.mean():.3f})")

    # merge
    df_train = df_train.merge(df_extra, on='SMILES', how='outer', suffixes=('', '_extra'))

    # 填补缺失
    mask = df_train[target].isna() & df_train[f"{target}_extra"].notna()
    df_train.loc[mask, target] = df_train.loc[mask, f"{target}_extra"]

    # 删除多余列
    if f"{target}_extra" in df_train.columns:
        df_train = df_train.drop(columns=[f"{target}_extra"])

    after = len(df_train)
    print(f"  [{target}] 增强: {after - before:+d} 条新样本, 填补 {mask.sum()} 条缺失")

    return df_train

def get_data_paths():
    BASE_PATH = Path(os.getenv('NEURIPS_DATA_PATH', 'kaggle/input/neurips-open-polymer-prediction-2025'))
    EXTRA_BASE = Path(os.getenv('EXTRA_DATA_BASE', 'kaggle/input/smiles-extra-data'))
    TC_BASE = Path(os.getenv('TC_DATA_BASE', 'kaggle/input/tc-smiles'))

    paths = {
        'train_csv': Path(os.getenv('TRAIN_CSV_PATH', BASE_PATH / 'train.csv')),
        'test_csv': Path(os.getenv('TEST_CSV_PATH', BASE_PATH / 'test.csv')),
        'sample_submission': Path(os.getenv('SAMPLE_SUBMISSION_PATH', BASE_PATH / 'sample_submission.csv')),

        'tc_data': Path(os.getenv('TC_DATA_PATH', TC_BASE / 'Tc_SMILES.csv')),
        'tg_jcim_data': Path(os.getenv('TG_JCIM_PATH', EXTRA_BASE / 'JCIM_sup_bigsmiles.csv')),
        'tg_excel_data': Path(os.getenv('TG_EXCEL_PATH', EXTRA_BASE / 'data_tg3.xlsx')),
        'density_data': Path(os.getenv('DENSITY_PATH', EXTRA_BASE / 'data_dnst1.xlsx')),
        'ffv_data': Path(os.getenv('FFV_DATA_PATH', BASE_PATH / 'train_supplement' / 'dataset4.csv')),
        'dataset1': Path(os.getenv('DATASET1_PATH', BASE_PATH / 'train_supplement' / 'dataset1.csv')),
        'dataset2': Path(os.getenv('DATASET2_PATH', BASE_PATH / 'train_supplement' / 'dataset2.csv')),
        'dataset3': Path(os.getenv('DATASET3_PATH', BASE_PATH / 'train_supplement' / 'dataset3.csv')),
    }
    return paths
# -----------------------------
# 通用图增强 transform
# -----------------------------

def graph_augment(data, node_drop_prob=0.0, edge_drop_prob=0.0, noise_std=0.0):
    """随机节点mask、边dropout、全局特征扰动"""
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
    """Kaggle Polymer Dataset（训练/推理通用）
       - 若 df 有 'id' 列，则把 id 存为 graph-level 属性 data.uid: LongTensor([id])
       - 若 df 无目标列或该目标缺失，则 y=0、mask=0（推理场景）
       - 记录无法构图的样本到 self.dropped
    """
    def __init__(self, df: pd.DataFrame, transform=None):
        super().__init__(None, transform)
        self.data_list = []
        self.dropped = []   # list of (id_or_None, smiles, reason)

        for _, row in df.iterrows():
            smi = row["SMILES"]
            sid = int(row["id"]) if "id" in df.columns and pd.notna(row["id"]) else None

            try:
                x, ei, ea = create_graph_from_smiles(smi)
            except Exception as e:
                self.dropped.append((sid, smi, f"graph_build_error:{e}"))
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
            
            data.uid = torch.tensor([sid if sid is not None else -1], dtype=torch.long)

            self.data_list.append(data)

    def len(self): return len(self.data_list)
    def get(self, idx): return self.data_list[idx]

def load_polymer_dataset(train_csv_or_df, batch_size=32, transform=None, shuffle=True):
    """加载 Kaggle Polymer Dataset（训练/推理通用）
       - train_csv_or_df: 可以是 CSV 路径或已准备好的 DataFrame
       - shuffle: 训练 True；推理 False
    """
    if isinstance(train_csv_or_df, (str, Path)):
        df = pd.read_csv(train_csv_or_df)
    else:
        df = train_csv_or_df.copy()

    df["SMILES"] = df["SMILES"].apply(make_smile_canonical)
    df = df[df["SMILES"].notnull()].reset_index(drop=True)

    dataset = PolymerDataset(df, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
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
