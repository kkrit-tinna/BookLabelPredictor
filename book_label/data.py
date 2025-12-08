# book_label/data.py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from .config import (
    LABELS_PATH,
    FROZEN_DESC_PATH,
    BATCH_SIZE,
    SEED,
)


class BooksDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_arrays(embedding_path=None):
    # load arrays from npy files
    if embedding_path is None:
        embedding_path = FROZEN_DESC_PATH

    X = np.load(embedding_path)
    Y = np.load(LABELS_PATH)

    assert X.shape[0] == Y.shape[0], "x and y must have same number of rows"

    label_names = None
    return X, Y, label_names


def get_dataloaders(X, Y, batch_size=BATCH_SIZE, val_ratio=0.1, test_ratio=0.1):
    # build train val test data loaders
    dataset = BooksDataset(X, Y)
    n = len(dataset)

    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


def train_val_test_split_indices(num_samples, val_ratio=0.1, test_ratio=0.1, seed=SEED):
    # split indices into train val test
    rng = np.random.RandomState(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    n_val = int(num_samples * val_ratio)
    n_test = int(num_samples * test_ratio)
    n_train = num_samples - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx
