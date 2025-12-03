# baseline_cosine.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .config import DATA_DIR


def cosine_baseline_precision_at_k(k_list=[1, 2, 3], test_ratio=0.15, seed=42):

    desc_emb = np.load(DATA_DIR / "description_embeddings.npy")  # (N, 384)
    label_emb = np.load(DATA_DIR / "label_embeddings.npy")      # (L, 384)
    y_true = np.load(DATA_DIR / "y_multilabel.npy")             # (N, L)

    print(f"Description embeddings shape: {desc_emb.shape}")
    print(f"Label embeddings shape: {label_emb.shape}")
    print(f"Y shape: {y_true.shape}")

    # Train/Test split
    np.random.seed(seed)
    n_samples = desc_emb.shape[0]
    indices = np.random.permutation(n_samples)

    test_size = int(n_samples * test_ratio)
    test_idx = indices[:test_size]

    desc_test = desc_emb[test_idx]
    y_test = y_true[test_idx]

    similarities = cosine_similarity(desc_test, label_emb)

    # Precision@k
    results = {}
    for k in k_list:
        topk_indices = np.argsort(similarities, axis=1)[:, -k:]

        correct = 0
        for i in range(len(test_idx)):
            true_labels = set(np.where(y_test[i] == 1)[0])
            pred_labels = set(topk_indices[i])
            correct += len(true_labels & pred_labels)

        precision = correct / (len(test_idx) * k)
        results[k] = precision
        print(f"Cosine Baseline Precision@{k}: {precision:.4f}")

    return results


if __name__ == "__main__":
    cosine_baseline_precision_at_k()