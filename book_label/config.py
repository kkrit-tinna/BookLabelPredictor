# book_label/config.py

from pathlib import Path

# paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# core data
LABELS_PATH = DATA_DIR / "y_multilabel.npy"
LABEL_TO_IDX_PATH = DATA_DIR / "label_to_idx.npy"

# embedding files
BASELINE_DESC_PATH = DATA_DIR / "description_embeddings.npy"
FROZEN_DESC_PATH = DATA_DIR / "frozen_description_embeddings.npy"
UNFROZEN_DESC_PATH = DATA_DIR / "unfrozen_description_embeddings.npy"

# default embedding for cosine baseline
EMBEDDINGS_PATH = FROZEN_DESC_PATH

# training config
DEVICE = "cpu"
HIDDEN_DIM = 512
LR = 1e-3
EPOCHS = 15
BATCH_SIZE = 256
SEED = 42
