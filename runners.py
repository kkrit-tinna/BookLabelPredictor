import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from preprocessing import get_or_create_core_data
from models import (
    LogisticTFIDFModel,
    CosineLabelEmbeddingModel,
    PrototypeEmbeddingModel,
    NeuralLabelPredictor,
)
from metrics import precision_at_ks, precision_at_k_per_sample, precision_at_k_per_sample_np
from generate_embeddings import generate_sbert_embeddings


DESC_EMB_BASELINE_PATH = "description_embeddings.npy"
LABEL_EMB_PATH = "label_embeddings.npy"
Y_MULTILABEL_PATH = "y_multilabel.npy"
DESC_EMB_FROZEN_PATH = "description_embeddings_frozen.npy"
DESC_EMB_UNFROZEN_PATH = "description_embeddings_unfrozen.npy"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_baseline_embeddings():
    try:
        X_all = np.load(DESC_EMB_BASELINE_PATH)
        label_emb = np.load(LABEL_EMB_PATH)
    except FileNotFoundError:
        generate_sbert_embeddings()
        X_all = np.load(DESC_EMB_BASELINE_PATH)
        label_emb = np.load(LABEL_EMB_PATH)
    return X_all, label_emb

def get_frozen_embeddings():
    try:
        return np.load(DESC_EMB_FROZEN_PATH)
    except FileNotFoundError:
        generate_sbert_embeddings()
        return np.load(DESC_EMB_FROZEN_PATH)

def get_unfrozen_embeddings():
    try:
        return np.load(DESC_EMB_UNFROZEN_PATH)
    except FileNotFoundError:
        generate_sbert_embeddings()
        return np.load(DESC_EMB_UNFROZEN_PATH)

def precision_at_k_np(y_true, y_scores, k):
    y_true = (y_true > 0).astype(int)
    n_samples, n_labels = y_true.shape
    topk_idx = np.argpartition(y_scores, -k, axis=1)[:, -k:]
    precisions = []
    for i in range(n_samples):
        true_labels = np.where(y_true[i] == 1)[0]
        if len(true_labels) == 0:
            precisions.append(0.0)
            continue
        pred_labels = set(topk_idx[i])
        tp = len(pred_labels.intersection(true_labels))
        precisions.append(tp / float(k))
    return float(np.mean(precisions))


def precision_at_ks_np(y_true, y_scores, k_list=(1, 2, 3)):
    return {k: precision_at_k_np(y_true, y_scores, k) for k in k_list}

def run_logreg_tfidf():
    core = get_or_create_core_data()
    df = core["df"]
    Y = core["Y"]

    train_idx = core["train_idx"]
    val_idx = core["val_idx"]
    test_idx = core["test_idx"]

    descriptions = np.array(df["description"].tolist())
    X_train_text = descriptions[train_idx]
    X_val_text = descriptions[val_idx]
    X_test_text = descriptions[test_idx]

    y_train = Y[train_idx]
    y_val = Y[val_idx]
    y_test = Y[test_idx]

    model = LogisticTFIDFModel(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        n_jobs=1,
    )

    model.fit(X_train_text, y_train)

    k_list = [1, 2, 3]
    val_scores = model.predict_proba(X_val_text)
    val_p_at_k = precision_at_ks_np(y_val, val_scores, k_list=k_list)

    for k in k_list:
        print(f"Val Precision@1: {val_p_at_k[1]:.4f}")
        break

    for k in k_list:
        print(f"Val Precision@{k}: {val_p_at_k[k]:.4f}")

    test_scores = model.predict_proba(X_test_text)
    test_p_at_k = precision_at_ks_np(y_test, test_scores, k_list=k_list)

    for k in k_list:
        print(f"Test Precision@{k}: {test_p_at_k[k]:.4f}")

    test_p_ki = precision_at_k_per_sample_np(y_test, test_scores)
    print(f"Test Precision@k_i (per-sample): {test_p_ki:.4f}")


def run_cosine_sbert():
    core = get_or_create_core_data()
    Y = core["Y"]

    train_idx = core["train_idx"]
    val_idx = core["val_idx"]
    test_idx = core["test_idx"]

    X_all, label_emb = get_baseline_embeddings()
    assert X_all.shape[0] == Y.shape[0]

    X_train = X_all[train_idx]
    X_val = X_all[val_idx]
    X_test = X_all[test_idx]

    y_train = Y[train_idx]
    y_val = Y[val_idx]
    y_test = Y[test_idx]

    model = CosineLabelEmbeddingModel(label_embeddings=label_emb)
    model.fit(X_train, y_train)

    k_list = [1, 2, 3]
    val_scores = model.predict_proba(X_val)
    val_p_at_k = precision_at_ks_np(y_val, val_scores, k_list=k_list)

    for k in k_list:
        print(f"Val Precision@{k}: {val_p_at_k[k]:.4f}")
    test_scores = model.predict_proba(X_test)
    test_p_at_k = precision_at_ks_np(y_test, test_scores, k_list=k_list)
    for k in k_list:
        print(f"Test Precision@{k}: {test_p_at_k[k]:.4f}")

    test_p_ki = precision_at_k_per_sample_np(y_test, test_scores)
    print(f"Test Precision@k_i (per-sample): {test_p_ki:.4f}")


def run_prototype_model():
    core = get_or_create_core_data()
    Y = core["Y"]

    train_idx = core["train_idx"]
    val_idx = core["val_idx"]
    test_idx = core["test_idx"]

    X_all, _ = get_baseline_embeddings()
    assert X_all.shape[0] == Y.shape[0]

    X_train = X_all[train_idx]
    X_val = X_all[val_idx]
    X_test = X_all[test_idx]
    y_train = Y[train_idx]
    
    y_val = Y[val_idx]
    y_test = Y[test_idx]
    model = PrototypeEmbeddingModel()
    model.fit(X_train, y_train)
    k_list = [1, 2, 3]
    val_scores = model.predict_proba(X_val)
    val_p_at_k = precision_at_ks_np(y_val, val_scores, k_list=k_list)
    for k in k_list:
        print(f"Val Prototype Precision@{k}: {val_p_at_k[k]:.4f}")
    test_scores = model.predict_proba(X_test)
    test_p_at_k = precision_at_ks_np(y_test, test_scores, k_list=k_list)
    for k in k_list:
        print(f"Test Prototype Precision@{k}: {test_p_at_k[k]:.4f}")
        
    test_p_ki = precision_at_k_per_sample_np(y_test, test_scores)
    print(f"Test Precision@k_i (per-sample): {test_p_ki:.4f}")


def train_nn_on_embeddings(
    X_all,
    Y,
    train_idx,
    val_idx,
    test_idx,
    hidden_dim=256,
    lr=1e-3,
    epochs=10,
    batch_size=128,
    update_embeddings: bool = False,
    save_updated_path: str | None = None,
    early_stopping: bool = True,
    patience: int = 2,
):
    
    """
    If update_embeddings = False:
        - X_all is treated as fixed input features (frozen embeddings)

    If update_embeddings = True:
        - X_all is considered a trainable parameter and updated during training.
        - saves updated embeddings to the input file path (default: "description_embeddings_unfrozen.npy")
    """
    n_samples, input_dim = X_all.shape
    _, output_dim = Y.shape

    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    test_idx = np.asarray(test_idx)

    y_train = Y[train_idx]
    y_val = Y[val_idx]
    y_test = Y[test_idx]

    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    if update_embeddings:
        embeddings = torch.nn.Parameter(
            torch.tensor(X_all, dtype=torch.float32, device=DEVICE)
        )

        train_idx_t = torch.tensor(train_idx, dtype=torch.long)
        val_idx_t = torch.tensor(val_idx, dtype=torch.long)
        test_idx_t = torch.tensor(test_idx, dtype=torch.long)

        train_ds = torch.utils.data.TensorDataset(train_idx_t, y_train_t)
        val_ds = torch.utils.data.TensorDataset(val_idx_t, y_val_t)
        test_ds = torch.utils.data.TensorDataset(test_idx_t, y_test_t)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        with torch.no_grad():

            X_val_eval = embeddings[val_idx_t].detach()
            X_test_eval = embeddings[test_idx_t].detach()

        val_eval_ds = torch.utils.data.TensorDataset(X_val_eval, y_val_t)
        test_eval_ds = torch.utils.data.TensorDataset(X_test_eval, y_test_t)

        val_eval_loader = torch.utils.data.DataLoader(val_eval_ds, batch_size=batch_size, shuffle=False)
        test_eval_loader = torch.utils.data.DataLoader(test_eval_ds, batch_size=batch_size, shuffle=False)

    else:
        X_train = X_all[train_idx]
        X_val = X_all[val_idx]
        X_test = X_all[test_idx]

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)

        train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        val_ds = torch.utils.data.TensorDataset(X_val_t, y_val_t)
        test_ds = torch.utils.data.TensorDataset(X_test_t, y_test_t)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        val_eval_loader = val_loader
        test_eval_loader = test_loader

    # creates the neural network architecture
    model = NeuralLabelPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=0.3,
    ).to(DEVICE)

    if update_embeddings:
        # treats embeddings as a changeable parameter
        optimizer = optim.Adam(
            list(model.parameters()) + [embeddings],
            lr=lr,
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for batch in train_loader:
            if update_embeddings:
                batch_idx, batch_Y = batch
                batch_idx = batch_idx.to(DEVICE)
                batch_X = embeddings[batch_idx]
            else:
                batch_X, batch_Y = batch
                batch_X = batch_X.to(DEVICE)

            batch_Y = batch_Y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = loss_fn(logits, batch_Y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * batch_Y.size(0)

        avg_train_loss = running_train_loss / len(train_ds)

        # validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if update_embeddings:
                    batch_idx, batch_Y = batch
                    batch_idx = batch_idx.to(DEVICE)
                    batch_X = embeddings[batch_idx]
                else:
                    batch_X, batch_Y = batch
                    batch_X = batch_X.to(DEVICE)

                batch_Y = batch_Y.to(DEVICE)
                logits = model(batch_X)
                loss = loss_fn(logits, batch_Y)
                running_val_loss += loss.item() * batch_Y.size(0)

        avg_val_loss = running_val_loss / len(val_ds)

        print(
            f"[NN] epoch {epoch+1}/{epochs}"
            f" | train loss: {avg_train_loss:.4f}"
            f" | val loss: {avg_val_loss:.4f}"
        )
        # stops training early if validation loss stops improving for input(patience, default: 2) number of epochs
        if avg_val_loss < best_val_loss - 1e-5:   
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if early_stopping and epochs_no_improve >= patience:
                print(
                    f"[NN] Early stopping at epoch {epoch+1} "
                    f"(best val loss so far: {best_val_loss:.4f})"
                )
                break

    # metrics
    k_list = [1, 2, 3]
    print("\n[NN] Validation metrics:")
    val_p_at_k = precision_at_ks(model, val_eval_loader, k_list=k_list, device=DEVICE)

    for k in k_list:
        print(f"Val Precision@{k}: {val_p_at_k[k]:.4f}")
    val_p_per_sample = precision_at_k_per_sample(model, val_eval_loader, device=DEVICE)
    print(f"Val Precision@k_i (per-sample): {val_p_per_sample:.4f}")

    print("\n[NN] Test metrics:")
    test_p_at_k = precision_at_ks(model, test_eval_loader, k_list=k_list, device=DEVICE)
    for k in k_list:
        print(f"Test Precision@{k}: {test_p_at_k[k]:.4f}")

    test_p_per_sample = precision_at_k_per_sample(model, test_eval_loader, device=DEVICE)
    print(f"Test Precision@k_i (per-sample): {test_p_per_sample:.4f}")

    # saves updated ebeddings to the input file path
    if update_embeddings and save_updated_path is not None:
        updated = embeddings.detach().cpu().numpy()
        np.save(save_updated_path, updated)

# Runner for the neural network projection using frozen embeddings
def run_nn_frozen():
    core = get_or_create_core_data()
    Y = core["Y"]
    train_idx = core["train_idx"]
    val_idx = core["val_idx"]
    test_idx = core["test_idx"]

    X_all = get_frozen_embeddings()
    assert X_all.shape[0] == Y.shape[0]

    print("Running NN on frozen SBERT embeddings")
    train_nn_on_embeddings(
        X_all,
        Y,
        train_idx,
        val_idx,
        test_idx,
        update_embeddings=False,
    )

# Runner for the neural network projection using unfrozen embeddings(loss is backpropagated to embeddings)
def run_nn_unfrozen():
    core = get_or_create_core_data()
    Y = core["Y"]
    train_idx = core["train_idx"]
    val_idx = core["val_idx"]
    test_idx = core["test_idx"]

    X_all = get_unfrozen_embeddings()
    assert X_all.shape[0] == Y.shape[0]

    print("Running NN on UNFROZEN SBERT embeddings (will update and save)")
    train_nn_on_embeddings(
        X_all,
        Y,
        train_idx,
        val_idx,
        test_idx,
        update_embeddings=True,
        save_updated_path=DESC_EMB_UNFROZEN_PATH,
    )