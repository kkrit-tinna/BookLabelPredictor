# models.py

import numpy as np
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


class LogisticTFIDFModel:
    """
    Multilabel logistic regression on top of TF-IDF features.

    - Vectorizes raw text with TfidfVectorizer.
    - Trains a One-vs-Rest logistic classifier over all labels.
    """

    def __init__(
        self,
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        n_jobs=1,
    ):
        """
        Configure TF-IDF + logistic regression hyperparameters.

        - Controls vocabulary size, n-gram range, and stop word handling.
        - Sets logistic regularization, iterations, and parallelism.
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.n_jobs = n_jobs

        self.vectorizer = None
        self.classifier = None
        self._train_label_mask = None
        self.n_labels_ = None

    def fit(self, texts, y_full):
        """
        Fit TF-IDF vectorizer and multilabel logistic regression.

        - Learns the TF-IDF vocabulary from training texts.
        - Trains only on labels that have at least one positive example.
        """
        texts = list(texts)
        y_full = np.asarray(y_full)
        n_samples, n_labels = y_full.shape
        self.n_labels_ = n_labels

        # TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words=self.stop_words,
        )
        X_train = self.vectorizer.fit_transform(texts)

        # masks labels with no examples in the train set
        col_sums = y_full.sum(axis=0)
        mask = col_sums > 0
        self._train_label_mask = mask

        y_reduced = y_full[:, mask]
        print(f"[TFIDF] Training on {y_reduced.shape[1]} of {n_labels} labels")

        base_lr = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
        )
        self.classifier = OneVsRestClassifier(base_lr, n_jobs=self.n_jobs)
        self.classifier.fit(X_train, y_reduced)
        return self

    def _predict_reduced(self, texts):
        """
        Predict probabilities for the subset of trainable labels.
        This allows the classifier to only predict on labels it has seen in the train space

        - Transforms input texts to TF-IDF space.
        - Runs the trained One-vs-Rest classifier to get p(y|x).
        """
        X = self.vectorizer.transform(list(texts))
        return self.classifier.predict_proba(X)

    def predict_proba(self, texts):
        """
        Predict probabilities for all labels

        - Fills trained label positions with model probabilities.
        - Sets unseen-label probabilities to zero. 
        """
        
        probs_reduced = self._predict_reduced(texts)
        n_samples = probs_reduced.shape[0]

        # after training on a reduced labelset, reforms returned probability array to full train labelset
        probs_full = np.zeros((n_samples, self.n_labels_), dtype=np.float32)
        probs_full[:, self._train_label_mask] = probs_reduced
        return probs_full


class NeuralLabelPredictor(nn.Module):
    """
    Feed-forward MLP for multilabel prediction on dense features.

    - Two hidden layers with batch norm, ReLU, and dropout.
    - Outputs raw logits for each label.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        """
        Initialize the MLP architecture.

        - input_dim: feature dimension of the embeddings.
        - output_dim: number of labels to predict.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """
        Run a batch of inputs through the network.

        - x has shape (batch_size, input_dim).
        - Returns logits of shape (batch_size, output_dim).
        """
        return self.net(x)

class CosineLabelEmbeddingModel:
    """
    Zero-shot cosine model using label name embeddings.

    - Normalizes label embeddings to unit vectors.
    - Scores each description embedding by cosine similarity to all labels.
    """

    def __init__(self, label_embeddings):
        """
        Initialize the cosine model with precomputed label embeddings.

        - Expects an array of shape (n_labels, d).
        - Row-normalizes the embeddings for cosine scoring.
        """
        # label_embeddings: np.ndarray of shape (n_labels, d)
        self.label_embeddings = label_embeddings / (
            np.linalg.norm(label_embeddings, axis=1, keepdims=True) + 1e-8
        )

    def predict_proba(self, X): 
        """
        Compute cosine similarity scores to label embeddings.

        - Normalizes each input embedding to unit length.
        - Returns a score matrix of shape (n_samples, n_labels).
        """
        X = np.asarray(X)
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        scores = X_norm @ self.label_embeddings.T
        return scores


class PrototypeEmbeddingModel:
    """
    Prototype-based cosine model.

    - Uses description embeddings + train labels to build one prototype per label
      as the mean embedding of all positive examples for that label.
    - Then uses cosine similarity to these prototypes for prediction.
    """

    def __init__(self):
        """
        Initialize an empty prototype container.

        - Prototypes are learned lazily in fit().
        """
        self.prototypes = None  # (n_labels, d)

    def fit(self, X_train, y_train):
        """
        Build label prototypes from training data.

        - Averages embeddings of all positive examples per label.
        - Falls back to global mean if a label has no positives.
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        n_train, d = X_train.shape
        n_train_y, n_labels = y_train.shape
        assert n_train == n_train_y

        prototypes = []
        for j in range(n_labels):
            mask = y_train[:, j] == 1
            if np.sum(mask) == 0:
                # if no positives (unseen label), uses the global mean as the prototype
                proto = X_train.mean(axis=0)
            else:
                proto = X_train[mask].mean(axis=0)
            prototypes.append(proto)

        prototypes = np.vstack(prototypes)
        # normalize
        self.prototypes = prototypes / (
            np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8
        )
        return self

    def predict_proba(self, X):
        """
        Score inputs by cosine similarity to label prototypes.

        - Normalizes input embeddings row-wise.
        - Returns a score matrix of shape (n_samples, n_labels).
        """
        if self.prototypes is None:
            raise RuntimeError("PrototypeEmbeddingModel.fit must be called before predict_proba.")

        X = np.asarray(X)
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        scores = X_norm @ self.prototypes.T
        return scores
