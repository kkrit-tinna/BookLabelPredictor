import numpy as np
import torch

from .config import DEVICE
from .data import load_arrays
from .model import NeuralLabelPredictor


def load_trained_model():
    X, Y, label_names = load_arrays()
    _, input_dim = X.shape
    _, output_dim = Y.shape

    model = NeuralLabelPredictor(
        input_dim=input_dim,
        hidden_dim=256,
        output_dim=output_dim,
    ).to(DEVICE)

    state_dict = torch.load("neural_label_predictor.pt", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    return model, X, Y, label_names


def show_prediction_example(sample_idx: int, k: int = 5):
    model, X, Y, label_names = load_trained_model()

    x_np = X[sample_idx]
    y_np = Y[sample_idx]

    x = torch.from_numpy(x_np).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0]

    probs_np = probs.cpu().numpy()

    true_indices = np.where(y_np == 1)[0]
    topk_idx = probs_np.argsort()[-k:][::-1]
    topk_probs = probs_np[topk_idx]

    true_labels = [label_names[i] for i in true_indices]
    pred_labels = [label_names[i] for i in topk_idx]

    # count how many correct
    correct = len(set(true_indices) & set(topk_idx))

    print(f"Sample index: {sample_idx}")
    print(f"True labels:       {true_labels}")
    print(f"Pred top-{k} labels: {pred_labels}")
    print(f"Pred probs:        {[f'{p:.3f}' for p in topk_probs]}")
    print(f"Correct in top-{k}: {correct}/{len(true_indices)}")
    print("-" * 50)

    return correct, len(true_indices)


def find_good_examples(n_samples: int = 20, k: int = 5):
    """find samples where model predicts well"""
    model, X, Y, label_names = load_trained_model()

    print(f"searching for good prediction examples...\n")

    good_examples = []

    for idx in range(min(n_samples * 10, len(X))):
        x_np = X[idx]
        y_np = Y[idx]

        x = torch.from_numpy(x_np).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)[0]

        probs_np = probs.cpu().numpy()
        true_indices = set(np.where(y_np == 1)[0])
        topk_idx = set(probs_np.argsort()[-k:])

        correct = len(true_indices & topk_idx)

        if correct >= 1:
            good_examples.append((idx, correct, len(true_indices)))

        if len(good_examples) >= n_samples:
            break

    # sort by correctness
    good_examples.sort(key=lambda x: x[1], reverse=True)

    print(f"found {len(good_examples)} examples with at least 1 correct prediction\n")

    for idx, correct, total in good_examples[:5]:
        show_prediction_example(idx, k)


if __name__ == "__main__":
    find_good_examples(n_samples=20, k=5)