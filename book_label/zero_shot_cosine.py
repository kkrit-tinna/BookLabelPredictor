# book_label/zero_shot_cosine.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .config import DATA_DIR, EMBEDDINGS_PATH, LABELS_PATH, SEED
from .data import train_val_test_split_indices

# one embedding per label in current label space
LABEL_EMB_PATH = DATA_DIR / "label_embeddings_prototype.npy"


def precision_at_k_subset(similarities, Y, k, label_subset=None):
    # precision@k, optionally restricted to a subset of labels
    n_test, num_labels = similarities.shape
    topk_indices = np.argsort(similarities, axis=1)[:, -k:]

    if label_subset is not None:
        label_subset = set(label_subset)

    correct = 0
    for i in range(n_test):
        true_labels = set(np.where(Y[i] == 1)[0])
        pred_labels = set(topk_indices[i])

        if label_subset is not None:
            true_labels = true_labels & label_subset
            pred_labels = pred_labels & label_subset

        correct += len(true_labels & pred_labels)

    precision = correct / (n_test * k)
    return precision


def zero_shot_cosine_precision(k_list=None):
    # cosine similarity between desc embeddings and label embeddings
    if k_list is None:
        k_list = [1, 2, 3]

    X = np.load(EMBEDDINGS_PATH)        # (n_books, d)
    Y = np.load(LABELS_PATH)            # (n_books, n_labels)
    label_emb = np.load(LABEL_EMB_PATH) # (n_labels, d)

    print(f"x shape (embeddings): {X.shape}")
    print(f"y shape (labels): {Y.shape}")
    print(f"label embeddings shape: {label_emb.shape}")

    assert X.shape[0] == Y.shape[0], "x and y must have same number of rows"
    assert label_emb.shape[0] == Y.shape[1], "label embeddings must match label count"

    num_samples = X.shape[0]
    train_idx, val_idx, test_idx = train_val_test_split_indices(
        num_samples, seed=SEED
    )

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    # define seen and unseen labels based on train and test
    freq_train = Y_train.sum(axis=0)
    freq_test = Y_test.sum(axis=0)

    seen_labels = np.where(freq_train > 0)[0]
    unseen_labels = np.where((freq_train == 0) & (freq_test > 0))[0]

    print(f"num labels: {Y.shape[1]}")
    print(f"seen labels: {len(seen_labels)}")
    print(f"unseen labels (train freq = 0, test freq > 0): {len(unseen_labels)}")

    # cosine similarity on test set
    similarities = cosine_similarity(X_test, label_emb)  # (n_test, n_labels)

    results_all = {}
    results_seen = {}
    results_unseen = {}

    for k in k_list:
        p_all = precision_at_k_subset(similarities, Y_test, k, label_subset=None)
        p_seen = precision_at_k_subset(similarities, Y_test, k, label_subset=seen_labels)
        p_unseen = precision_at_k_subset(
            similarities, Y_test, k, label_subset=unseen_labels
        )

        results_all[k] = p_all
        results_seen[k] = p_seen
        results_unseen[k] = p_unseen

        print(f"\nprecision@{k}:")
        print(f"  all labels   : {p_all:.4f}")
        print(f"  seen labels  : {p_seen:.4f}")
        print(f"  unseen labels: {p_unseen:.4f}")

    return {
        "all": results_all,
        "seen": results_seen,
        "unseen": results_unseen,
    }


if __name__ == "__main__":
    zero_shot_cosine_precision()