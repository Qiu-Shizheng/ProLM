"""Data loading helpers for ProLM fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class ProLMData:
    expression: pd.DataFrame
    labels: pd.DataFrame
    protein_names: list[str]
    features_tensor: torch.Tensor
    bias_matrix: torch.Tensor


class ProteinClassificationDataset(Dataset):
    def __init__(self, expression: pd.DataFrame, labels: pd.DataFrame):
        self.expression = expression
        self.labels = labels.set_index("eid").loc[expression.index]
        self.eids = expression.index.astype(str).tolist()

    def __len__(self) -> int:
        return len(self.eids)

    def __getitem__(self, index: int):
        eid = self.eids[index]
        x = self.expression.loc[eid].values.astype(np.float32)
        y = int(self.labels.loc[eid, "label"])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long), eid


def load_sequence_embeddings(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Sequence embeddings must be provided as an .npz file.")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].astype(np.float32) for key in data.files}


def load_prolm_data(
    expression_csv: str | Path,
    labels_csv: str | Path,
    protein_embeddings_npz: str | Path,
    correlation_csv: str | Path | None = None,
    eid_col: str = "eid",
    label_col: str = "label",
) -> ProLMData:
    expression = pd.read_csv(expression_csv, dtype={eid_col: str})
    labels = pd.read_csv(labels_csv, dtype={eid_col: str})
    if eid_col not in expression.columns:
        raise ValueError(f"Expression file must contain an '{eid_col}' column.")
    if eid_col not in labels.columns or label_col not in labels.columns:
        raise ValueError(f"Labels file must contain '{eid_col}' and '{label_col}' columns.")

    expression = expression.set_index(eid_col)
    labels = labels[[eid_col, label_col]].rename(columns={eid_col: "eid", label_col: "label"})
    labels["eid"] = labels["eid"].astype(str)
    labels["label"] = labels["label"].astype(int)
    expression.index = expression.index.astype(str)
    expression = expression.apply(pd.to_numeric, errors="coerce")

    embeddings = load_sequence_embeddings(protein_embeddings_npz)
    common = [name for name in expression.columns if name in embeddings]
    corr = None
    if correlation_csv:
        corr = pd.read_csv(correlation_csv, index_col=0)
        common = [name for name in common if name in corr.index and name in corr.columns]
    if not common:
        raise ValueError("No shared proteins across expression and sequence embeddings.")

    expression = expression[common].astype(np.float32)
    expression = expression.loc[expression.index.intersection(labels["eid"])]
    labels = labels[labels["eid"].isin(expression.index)].copy()
    labels = labels.drop_duplicates(subset="eid").set_index("eid").loc[expression.index].reset_index()
    expression = expression.fillna(expression.mean(axis=0)).fillna(0.0)

    features = np.stack([embeddings[name] for name in common]).astype(np.float32)
    if corr is None:
        bias = torch.zeros((len(common) + 1, len(common) + 1), dtype=torch.float32)
    else:
        corr = corr.loc[common, common].astype(np.float32)
        corr_values = np.nan_to_num(corr.values, nan=0.0, posinf=0.0, neginf=0.0)
        bias = torch.zeros((len(common) + 1, len(common) + 1), dtype=torch.float32)
        bias[1:, 1:] = torch.tensor(corr_values, dtype=torch.float32)

    return ProLMData(
        expression=expression,
        labels=labels,
        protein_names=common,
        features_tensor=torch.tensor(features, dtype=torch.float32),
        bias_matrix=bias,
    )
