import pandas as pd
import random
import numpy as np
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from joblib import parallel_backend

BOOKS_DATASET_PATH = "BooksDataset.csv"

def split_categories(category_string):

    category_string = category_string.strip()
    if category_string == "":
        return []
    genre_tokens = [p.strip() for p in category_string.split(",")]
    genre_tokens = [p for p in genre_tokens if p]
    cleaned_tokens = []
    for p in genre_tokens:
        low = p.lower()
        if low in {"nan", "none"}:
            continue
        if low == "general":
            continue
        cleaned_tokens.append(p)
    return cleaned_tokens

def load_descriptions_genres(csv_path):
    cache_path = csv_path + ".cleaned.pkl"
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)
    
    df = pd.read_csv(csv_path)
    df["Description"] = df["Description"].astype(str).str.strip()
    df["Category"] = df["Category"].astype(str).str.strip()

    mask_non_empty = (df["Description"] != "") & (df["Category"] != "")

    df = df[mask_non_empty]
    df["categories"] = df["Category"].apply(split_categories)
    df = df[df["categories"].map(len) > 0]

    cleaned_df = df[["Description", "categories"]].rename(columns={
        "Description": "description"
    }).reset_index(drop=True)

    cleaned_df.to_pickle(cache_path)
    return cleaned_df

def tfidf_tester(
    csv_path,
    sample_size,
    test_size,
    no_of_examples,
    random_state = 7,
    use_all = False
):
    cleaned_df = load_descriptions_genres(csv_path)
    if use_all:
        sample_size = len(cleaned_df)
    else:
        sample_size = min(sample_size, len(cleaned_df))
    random.seed(random_state)
    indices = random.sample(range(len(cleaned_df)), sample_size)
    sample_df = cleaned_df.iloc[indices].reset_index(drop=True)
    descriptions_all = sample_df["description"].tolist()
    categories_all  = sample_df["categories"].tolist()
    mlb = MultiLabelBinarizer()

    Y_all = mlb.fit_transform(categories_all)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        descriptions_all,
        Y_all,
        test_size = test_size,
        random_state = random_state
    )

    vectorizer = TfidfVectorizer(
        max_features = 10000,
        ngram_range = (1, 2),
        stop_words = 'english'
    )

    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    n_train = y_train.shape[0]
    col_sums = y_train.sum(axis=0)

    valid_mask = (col_sums > 0) & (col_sums < n_train)
    y_train_reduced = y_train[:, valid_mask]
    y_test_reduced  = y_test[:, valid_mask]
    reduced_classes = mlb.classes_[valid_mask]

    print("Original label count:", len(mlb.classes_))
    print("Usable label count:  ", len(reduced_classes))

    classifier = OneVsRestClassifier(LogisticRegression(
        max_iter = 1000,
        n_jobs = -1,
        class_weight = "balanced"
    ))

    with parallel_backend("threading"):
        classifier.fit(X_train, y_train_reduced)
        
    Y_scores = classifier.predict_proba(X_test)

    print(f"Sample size: {sample_size}")
    print(f"Train examples: {len(X_train_raw)}")
    print(f"Test examples:  {len(X_test_raw)}")
    print(f"Label tokens:   {len(reduced_classes)}\n")

    for i in range(min(no_of_examples, X_test.shape[0])):
        print("----- Example", i + 1, "-----")
        print("Description:")

        print(X_test_raw[i][:300], "...\n")
        true_vec = y_test_reduced[i]
        true_idx = np.where(true_vec == 1)[0]
        true_labels = [reduced_classes[j] for j in true_idx]

        print("True labels:", true_labels)

        scores = Y_scores[i]
        top_k = 3
        top_indices = np.argsort(scores)[-top_k:][::-1]

        print("Predicted top-3 labels:")

        for idx in top_indices:
            label = reduced_classes[idx]
            score = scores[idx]
            print(f"  {label} (score={score:.3f})")
        print()

cleaned_df = load_descriptions_genres(BOOKS_DATASET_PATH)
mapping = {
    row["description"]: row["categories"]
    for _, row in cleaned_df.iterrows()
}
print(f"Total of {len(mapping)} descriptions with genres")

# Uncomment this code to run the TF-IDF, log-Reg baseline.

tfidf_tester(
    BOOKS_DATASET_PATH,
    sample_size = 5000,
    test_size = 0.2,
    no_of_examples = 10,
    random_state = 7,
    use_all = True
    )
