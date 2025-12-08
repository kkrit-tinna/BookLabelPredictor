# book_label/compare_nn_embeddings.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import random

from .config import (
    DATA_DIR,
    LABELS_PATH,
    HIDDEN_DIM,
    LR,
    EPOCHS,
    DEVICE,
    BASELINE_DESC_PATH,
    FROZEN_DESC_PATH,
    UNFROZEN_DESC_PATH,
)
from .data import get_dataloaders
from .model import NeuralLabelPredictor
from .metrics import precision_at_ks


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

EMBEDDING_FILES = {
    "baseline_desc": BASELINE_DESC_PATH,
    "frozen_desc": FROZEN_DESC_PATH,
    "unfrozen_desc": UNFROZEN_DESC_PATH,
}

K_LIST = [1, 2, 3]


def train_and_eval_single(X, Y, embedding_name):
    num_samples, input_dim = X.shape
    _, output_dim = Y.shape

    print(f"\n=== embedding: {embedding_name} ===")
    print(f"x shape: {X.shape}, y shape: {Y.shape}")
    print(f"num labels: {output_dim}")
    print(f"using device: {DEVICE}")

    (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    ) = get_dataloaders(X, Y)

    model = NeuralLabelPredictor(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        output_dim=output_dim,
    ).to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0

        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_Y = batch_Y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = loss_fn(logits, batch_Y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * batch_X.size(0)

        avg_train_loss = running_train_loss / len(train_dataset)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X = batch_X.to(DEVICE)
                batch_Y = batch_Y.to(DEVICE)
                logits = model(batch_X)
                loss = loss_fn(logits, batch_Y)
                running_val_loss += loss.item() * batch_X.size(0)

        avg_val_loss = running_val_loss / len(val_dataset)

        print(
            f"[{embedding_name}] epoch {epoch + 1}/{EPOCHS} "
            f"| train loss: {avg_train_loss:.4f} "
            f"| val loss: {avg_val_loss:.4f}"
        )

    results_rows = []

    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        p_results = precision_at_ks(model, loader, k_list=K_LIST, device=DEVICE)
        row = {
            "embedding": embedding_name,
            "split": split_name,
        }
        for k in K_LIST:
            row[f"P@{k}"] = p_results[k]
        results_rows.append(row)

    return results_rows


def main():
    all_rows = []

    Y = np.load(LABELS_PATH)

    for name, path in EMBEDDING_FILES.items():
        X = np.load(path)
        rows = train_and_eval_single(X, Y, embedding_name=name)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    csv_path = DATA_DIR / "nn_embedding_comparison.csv"
    df.to_csv(csv_path, index=False)
    print("\nsaved results to", csv_path)
    print(df)

    val_df = df[df["split"] == "val"].set_index("embedding")

    plt.figure(figsize=(6, 4))
    for k in K_LIST:
        plt.plot(val_df.index, val_df[f"P@{k}"], marker="o", label=f"P@{k}")

    plt.xlabel("embedding")
    plt.ylabel("precision")
    plt.title("neural network precision@k (val) for different embeddings")
    plt.legend()
    plt.tight_layout()
    plot_path = DATA_DIR / "nn_embedding_comparison_val.png"
    plt.savefig(plot_path)
    print("saved plot to", plot_path)


if __name__ == "__main__":
    main()
