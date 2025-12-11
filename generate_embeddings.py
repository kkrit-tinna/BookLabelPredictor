import os
import numpy as np

from sentence_transformers import SentenceTransformer
from preprocessing import get_or_create_core_data

DESC_EMB_PATH = "description_embeddings.npy"
LABEL_EMB_PATH = "label_embeddings.npy"
Y_MULTILABEL_PATH = "y_multilabel.npy"
DESC_EMB_FROZEN_PATH = "description_embeddings_frozen.npy"
DESC_EMB_UNFROZEN_PATH = "description_embeddings_unfrozen.npy"


def generate_sbert_embeddings(force: bool = False):
    if (
        not force
        and os.path.exists(DESC_EMB_PATH)
        and os.path.exists(DESC_EMB_FROZEN_PATH)
        and os.path.exists(DESC_EMB_UNFROZEN_PATH)
        and os.path.exists(LABEL_EMB_PATH)
        and os.path.exists(Y_MULTILABEL_PATH)
    ):
        return

    core = get_or_create_core_data()
    df = core["df"]
    Y = core["Y"]
    class_names = core["class_names"]

    descriptions = df["description"].tolist()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    desc_emb = model.encode(
        descriptions,
        show_progress_bar=True,
        batch_size=32,
    )

    np.save(DESC_EMB_PATH, desc_emb)

    if force or not os.path.exists(DESC_EMB_FROZEN_PATH):
        np.save(DESC_EMB_FROZEN_PATH, desc_emb)

    if force or not os.path.exists(DESC_EMB_UNFROZEN_PATH):
        np.save(DESC_EMB_UNFROZEN_PATH, desc_emb)

    label_emb = model.encode(
        class_names,
        show_progress_bar=True,
        batch_size=32,
    )
    np.save(LABEL_EMB_PATH, label_emb)

    np.save(Y_MULTILABEL_PATH, Y)