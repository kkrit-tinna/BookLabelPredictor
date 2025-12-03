import os
import random
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from joblib import parallel_backend

import preprocessing as prep

BOOKS_DATASET_PATH = "BooksDataset.csv"


def fit_tfidf_and_optional_lsa(texts, max_features = 10000, ngram_range = (1, 2), min_df = 2, use_lsa = True, n_components = 300):
    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features = max_features, ngram_range = ngram_range, stop_words = 'english', min_df = min_df)
    X_sparse = vectorizer.fit_transform(texts)
    print(f"TF-IDF matrix shape: {X_sparse.shape}")

    if use_lsa:
        print(f"Fitting SVD with n_components = {n_components}...")
        svd = TruncatedSVD(n_components = n_components, random_state = 42, n_iter = 7)
        X_proj = svd.fit_transform(X_sparse)
        norms = np.linalg.norm(X_proj, axis = 1, keepdims = True)
        norms[norms == 0] = 1.0
        X_proj = X_proj / norms
        print(f"LSA projected shape: {X_proj.shape}")
        return vectorizer, svd, X_sparse, X_proj
    else:
        return vectorizer, None, X_sparse, None


def model_evaluation(y_true, scores, per_sample_k = True, fixed_k = 3):
    n, m = y_true.shape
    assert scores.shape == (n, m)
    per_sample_scores = np.zeros(n, dtype = float)
    valid_mask = y_true.sum(axis = 1) > 0
    desc_args = np.argsort(scores, axis = 1)[:, ::-1]
    for i in range(n):
        true_count = int(y_true[i].sum())
        if true_count == 0:
            per_sample_scores[i] = 0.0
            continue
        k = true_count if per_sample_k else fixed_k
        k = max(1, min(m, k))
        topk_idx = desc_args[i, :k]
        correct = int(np.sum(y_true[i, topk_idx]))
        per_sample_scores[i] = correct / float(true_count)
    mean_score = per_sample_scores[valid_mask].mean() if np.any(valid_mask) else 0.0
    return mean_score, per_sample_scores


def tfidf_tester(
    cleaned_df,
    sample_size = 60000,
    test_size = 0.2,
    no_of_examples = 10,
    random_state = 7,
    use_all = False,
    use_lsa = True,
    lsa_components = 300
):

    if use_all:
        sample_size = len(cleaned_df)
    else:
        sample_size = min(sample_size, len(cleaned_df))

    print(f"Sampling {sample_size} examples (use_all = {use_all})")
    random.seed(random_state)
    indices = random.sample(range(len(cleaned_df)), sample_size)
    sample_df = cleaned_df.iloc[indices].reset_index(drop = True)

    descriptions_all = sample_df["description"].tolist()
    categories_all = sample_df["categories"].tolist()

    mlb = MultiLabelBinarizer()
    Y_all = mlb.fit_transform(categories_all)
    print(f"Built MultiLabel Binarizer with {len(mlb.classes_)} classes")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        descriptions_all, Y_all, test_size = test_size, random_state = random_state
    )
    print(f"Train/test split: {len(X_train_raw)}/{len(X_test_raw)}")

    vectorizer, svd, X_train_sparse, X_train_proj = fit_tfidf_and_optional_lsa(
        X_train_raw, use_lsa = use_lsa, n_components = lsa_components
    )

    X_test_sparse = vectorizer.transform(X_test_raw)
    X_test_proj = svd.transform(X_test_sparse) if (svd is not None) else None

    n_train = y_train.shape[0]
    col_sums = y_train.sum(axis = 0)
    valid_mask = (col_sums > 0) & (col_sums < n_train)
    y_train_reduced = y_train[:, valid_mask]
    y_test_reduced = y_test[:, valid_mask]
    reduced_classes = mlb.classes_[valid_mask]

    print("Original label count:", len(mlb.classes_))
    print("Usable label count:  ", len(reduced_classes))

    if X_train_proj is not None:
        X_train_feats = X_train_proj
        X_test_feats = X_test_proj
    else:
        X_train_feats = X_train_sparse
        X_test_feats = X_test_sparse

    classifier = OneVsRestClassifier(
        LogisticRegression(max_iter = 1000, n_jobs = -1, class_weight = "balanced")
    )
    print("Starting classifier training...")
    with parallel_backend("threading"):
        classifier.fit(X_train_feats, y_train_reduced)
    print("Classifier training complete")

    Y_scores = classifier.predict_proba(X_test_feats)
    mean_adj, per_sample = model_evaluation(y_test_reduced, Y_scores, per_sample_k = True)

    print("Prediction complete")
    print(f"Adjusted precision (per-sample k): {mean_adj:.4f}")
    print(f"Sample size: {sample_size}")
    print(f"Train examples: {len(X_train_raw)}")
    print(f"Test examples:  {len(X_test_raw)}")
    print(f"Label tokens:   {len(reduced_classes)}\n")

    for i in range(min(no_of_examples, len(X_test_raw))):
        print("----- Example", i + 1, "-----")
        print("Description:")
        print(X_test_raw[i][:300], "...\n")
        true_vec = y_test_reduced[i]
        true_idx = np.where(true_vec == 1)[0]
        true_labels = [reduced_classes[j] for j in true_idx]
        print("True labels:", true_labels)
        scores = Y_scores[i]
        top_k = max(1, int(true_vec.sum()))
        top_indices = np.argsort(scores)[-top_k:][::-1]
        print("Predicted labels:")
        for idx in top_indices:
            label = reduced_classes[idx]
            score = scores[idx]
            print(f"  {label} (score = {score:.3f})")
        print()


if __name__ == "__main__":
    cleaned_df = prep.load_descriptions_genres(BOOKS_DATASET_PATH)
    mapping = {row["description"]: row["categories"] for _, row in cleaned_df.iterrows()}
    print(f"Total of {len(mapping)} descriptions with genres")

    tfidf_tester(
        cleaned_df,
        sample_size = 20000,
        test_size = 0.2,
        no_of_examples = 10,
        random_state = 7,
        use_all = True,
        use_lsa = True,
        lsa_components = 300
    )
