import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from .config import (
    EMBEDDINGS_PATH,
    LABELS_PATH,
    LABEL_NAMES_PATH,
    BATCH_SIZE,
    SEED,
)


class BookDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, indices: np.ndarray):
        X_sel = X[indices]
        Y_sel = Y[indices]
        self.X = torch.from_numpy(X_sel).float()
        self.Y = torch.from_numpy(Y_sel).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_arrays():
    X = np.load(EMBEDDINGS_PATH)           # (N, 384)
    Y = np.load(LABELS_PATH)              # (N, L)

    try:
        label_names = np.load(LABEL_NAMES_PATH, allow_pickle=True)  # (L,)

    except FileNotFoundError:
        num_labels = Y.shape[1]
        label_names = np.array([f"label_{i}" for i in range(num_labels)], dtype=object)

    return X, Y, label_names


def train_val_test_split_indices(num_samples: int, seed: int = SEED):
    indices = np.arange(num_samples)

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        shuffle=True,
        stratify=None,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
        stratify=None,
    )

    return train_idx, val_idx, test_idx


def get_dataloaders(X: np.ndarray, Y: np.ndarray):
    num_samples = X.shape[0]
    train_idx, val_idx, test_idx = train_val_test_split_indices(num_samples)

    train_dataset = BookDataset(X, Y, train_idx)
    val_dataset = BookDataset(X, Y, val_idx)
    test_dataset = BookDataset(X, Y, test_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
