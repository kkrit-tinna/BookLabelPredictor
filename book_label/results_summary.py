# book_label/results_summary.py

import torch

from .config import DEVICE, HIDDEN_DIM
from .data import load_arrays, get_dataloaders
from .model import NeuralLabelPredictor
from .metrics import precision_at_k
from .baseline_cosine import cosine_baseline_precision_at_k


def main():
    # load data
    X, Y, label_names = load_arrays()
    num_samples, input_dim = X.shape
    _, output_dim = Y.shape

    (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    ) = get_dataloaders(X, Y)

    # load model
    model = NeuralLabelPredictor(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        output_dim=output_dim,
    ).to(DEVICE)

    model.load_state_dict(torch.load("neural_label_predictor.pt", map_location=DEVICE))
    model.eval()

    k_list = [1, 2, 3]

    # nn precision
    nn_val = {k: precision_at_k(model, val_loader, k=k, device=DEVICE) for k in k_list}
    nn_test = {k: precision_at_k(model, test_loader, k=k, device=DEVICE) for k in k_list}

    # cosine precision
    cos_test = cosine_baseline_precision_at_k(k_list=k_list)

    # print table
    print("\n=== Precision@k Summary ===")
    print("Split   | Model             | P@1    | P@2    | P@3")
    print("--------|-------------------|--------|--------|--------")

    print("Val     | NN (MLP)          "
          f"| {nn_val[1]:.4f} | {nn_val[2]:.4f} | {nn_val[3]:.4f}")

    print("Test    | Cosine baseline   "
          f"| {cos_test[1]:.4f} | {cos_test[2]:.4f} | {cos_test[3]:.4f}")

    print("Test    | NN (MLP)          "
          f"| {nn_test[1]:.4f} | {nn_test[2]:.4f} | {nn_test[3]:.4f}")


if __name__ == "__main__":
    main()
