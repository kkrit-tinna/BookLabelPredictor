# book_label/build_label_names.py

import numpy as np
from .config import DATA_DIR, LABELS_PATH, LABEL_NAMES_PATH

def build_label_names():

    Y = np.load(LABELS_PATH)
    num_labels = Y.shape[1]
    print("Y shape:", Y.shape, "| num_labels:", num_labels)

    label_to_idx_path = DATA_DIR / "label_to_idx.npy"
    label_to_idx = np.load(label_to_idx_path, allow_pickle=True).item()
    print("len(label_to_idx):", len(label_to_idx))

    names_by_idx = [None] * num_labels
    for name, idx in label_to_idx.items():
        if 0 <= idx < num_labels:
            names_by_idx[idx] = name

    for i in range(num_labels):
        if names_by_idx[i] is None:
            names_by_idx[i] = f"label_{i}"

    label_names = np.array(names_by_idx, dtype=object)
    np.save(LABEL_NAMES_PATH, label_names)

    print("Saved label_names.npy to", LABEL_NAMES_PATH)
    print("first 10 labels:", label_names[:10])

if __name__ == "__main__":
    build_label_names()
