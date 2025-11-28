import torch
import torch.nn as nn
import torch.optim as optim

from .config import (
    DEVICE,
    HIDDEN_DIM,
    LR,
    EPOCHS,
)
from .data import load_arrays, get_dataloaders
from .model import NeuralLabelPredictor
from .metrics import precision_at_k


def train_and_eval():
    X, Y, label_names = load_arrays()
    num_samples, input_dim = X.shape
    _, output_dim = Y.shape

    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"Num labels: {output_dim}")
    print(f"Using device: {DEVICE}")

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

    print(model)

    for epoch in range(EPOCHS):
        # train
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

        # val
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
            f"Epoch {epoch+1}/{EPOCHS} - "
            f"Train loss: {avg_train_loss:.4f} | "
            f"Val loss: {avg_val_loss:.4f}"
        )

    # precision@k on val / test
    for split_name, loader in [("Val", val_loader), ("Test", test_loader)]:
        for k in [1, 3, 5]:
            p_at_k = precision_at_k(model, loader, k=k, device=DEVICE)
            print(f"{split_name} Precision@{k}: {p_at_k:.4f}")


    model_path = "neural_label_predictor.pt"
    torch.save(model.state_dict(), model_path)
    print("Model saved to", model_path)


if __name__ == "__main__":
    train_and_eval()
