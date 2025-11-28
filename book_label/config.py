import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"

EMBEDDINGS_PATH = DATA_DIR / "description_embeddings.npy"
LABELS_PATH = DATA_DIR / "y_multilabel.npy"
LABEL_NAMES_PATH = DATA_DIR / "label_names.npy"

# train
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 15
HIDDEN_DIM = 256
SEED = 42

DEVICE = "cpu"
