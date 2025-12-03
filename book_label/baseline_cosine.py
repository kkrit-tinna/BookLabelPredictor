# book_label/baseline_cosine.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .config import EMBEDDINGS_PATH, LABELS_PATH, SEED
from .data import train_val_test_split_indices


def cosine_baseline_precision_at_k(k_list=[1, 2, 3]):
    """
    cosine baseline using same label space and train/val/test split
    """

    # x: (n, 384), y: (n, 15)
    X = np.load(EMBEDDINGS_PATH)   # embeddings
    Y = np.load(LABELS_PATH)       # labels multi-hot

    print(f"X shape (embeddings): {X.shape}")
    print(f"Y shape (labels): {Y.shape}")

    num_samples = X.shape[0]
    train_idx, val_idx, test_idx = train_val_test_split_indices(num_samples, seed=SEED)

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test,  Y_test  = X[test_idx],  Y[test_idx]

    num_labels = Y.shape[1]

    # build label prototypes from train
    label_prototypes = []
    for j in range(num_labels):
        mask = Y_train[:, j] == 1
        if mask.sum() == 0:
            # fallback use train mean
            proto = X_train.mean(axis=0)
        else:
            proto = X_train[mask].mean(axis=0)
        label_prototypes.append(proto)

    label_prototypes = np.vstack(label_prototypes)  # (15, 384)
    print(f"Label prototypes shape: {label_prototypes.shape}")

    # compute cosine similarity
    similarities = cosine_similarity(X_test, label_prototypes)  # (n_test, 15)

    # compute precision@k
    results = {}
    n_test = X_test.shape[0]

    for k in k_list:
        # get top k
        topk_indices = np.argsort(similarities, axis=1)[:, -k:]  # (n_test, k)

        correct = 0
        for i in range(n_test):
            true_labels = set(np.where(Y_test[i] == 1)[0])
            pred_labels = set(topk_indices[i])
            correct += len(true_labels & pred_labels)

        precision = correct / (n_test * k)
        results[k] = precision
        print(f"[cosine 15-label] precision@{k}: {precision:.4f}")

    return results


if __name__ == "__main__":
    cosine_baseline_precision_at_k()
