# book_label/embedding_generation.py

import os
import re
import html
import pandas as pd
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from .config import DATA_DIR


def build_embeddings_and_labels():
    os.makedirs(DATA_DIR, exist_ok=True)

    # load dataset
    csv_path = DATA_DIR / "BooksDatasetClean.csv"
    df = pd.read_csv(csv_path)
    print(f"original dataset shape: {df.shape}")

    # remove rows with missing category or description
    df = df.dropna(subset=['Category', 'Description'])
    print(f"after removing missing: {df.shape}")

    # add book_id
    df['book_id'] = df.index

    def preprocess_category(cat):
        if pd.isna(cat):
            return None
        cat = str(cat).strip()
        cat = cat.replace(' & ', ', ')
        categories = [c.strip() for c in cat.split(',')]

        cleaned = []
        for c in categories:
            if c.lower().strip() == 'general':
                continue
            c = re.sub(r'\(.*?\)', '', c)
            c = re.sub(r'[()]|\d+', '', c)
            c = c.replace("'s", "s").replace("-", " ")
            c = c.lower().strip()
            c = ' '.join(c.split())
            if c:
                cleaned.append(c)
        return ', '.join(cleaned) if cleaned else None

    df['Category_Cleaned'] = df['Category'].apply(preprocess_category)
    df = df[df['Category_Cleaned'].notna()]
    print(f"after cleaning categories: {df.shape}")

    df_exploded = df.copy()
    df_exploded['Category_Split'] = df_exploded['Category_Cleaned'].str.split(', ')
    df_exploded = df_exploded.explode('Category_Split')
    df_exploded = df_exploded[df_exploded['Category_Split'].notna()]
    df_exploded = df_exploded.reset_index(drop=True)

    print(f"after exploding: {df_exploded.shape}")
    print(f"unique categories: {df_exploded['Category_Split'].nunique()}")

    # lemmatization

    nlp = spacy.load("en_core_web_sm")

    def lemmatize_category(cat):
        if pd.isna(cat) or not cat:
            return None
        doc = nlp(cat)
        lemmas = [token.lemma_ for token in doc if not token.is_punct]
        canonical = " ".join(lemmas).strip()
        return canonical if canonical else None

    df_exploded['Category_Lemmatized'] = df_exploded['Category_Split'].apply(lemmatize_category)
    df_exploded = df_exploded[df_exploded['Category_Lemmatized'].notna()].copy()

    print(f"unique categories after lemmatization: {df_exploded['Category_Lemmatized'].nunique()}")

    df_exploded = df_exploded.drop_duplicates(subset=['book_id', 'Category_Lemmatized']).reset_index(drop=True)
    print(f"after deduplication: {len(df_exploded)}")

    min_frequency = 4
    category_counts = df_exploded['Category_Lemmatized'].value_counts()
    filtered_categories = category_counts[category_counts >= min_frequency].index.tolist()
    df_exploded = df_exploded[df_exploded['Category_Lemmatized'].isin(filtered_categories)]
    print(f"after filtering low-freq categories: {len(df_exploded)}")

    df_exploded['Final_Label'] = df_exploded['Category_Lemmatized']

    def clean_description(text):
        if pd.isna(text) or text == '':
            return ''
        text = str(text)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = html.unescape(text)
        text = text.replace('Ã¢â‚¬â„¢', "'")
        text = text.replace('Ã¢â‚¬Å"', '"')
        text = text.replace('Ã¢â‚¬', '"')
        text = text.replace('Ã¢â‚¬"', '—')
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    df['Description_Cleaned'] = df['Description'].apply(clean_description)
    print(f"total books: {len(df)}")
    print(f"empty descriptions: {(df['Description_Cleaned'] == '').sum()}")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # unique labels
    unique_labels = sorted(df_exploded['Final_Label'].unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    print(f"num unique labels: {len(unique_labels)}")

    # label embeddings
    label_embeddings = model.encode(unique_labels, show_progress_bar=True)
    np.save(DATA_DIR / 'label_embeddings.npy', label_embeddings)
    np.save(DATA_DIR / 'label_to_idx.npy', label_to_idx)
    print(f"label embeddings shape: {label_embeddings.shape}")

    descriptions = df['Description_Cleaned'].tolist()
    description_embeddings = model.encode(descriptions, show_progress_bar=True, batch_size=32)
    np.save(DATA_DIR / 'description_embeddings.npy', description_embeddings)
    print(f"description embeddings shape: {description_embeddings.shape}")

    # multi-label target matrix
    n_books = len(df)
    n_labels = len(unique_labels)
    y_multilabel = np.zeros((n_books, n_labels), dtype=np.float32)

    # map book_id to row index
    book_ids = df['book_id'].tolist()
    bookid_to_row = {bid: i for i, bid in enumerate(book_ids)}

    for _, row in df_exploded.iterrows():
        book_id = row['book_id']
        label = row['Final_Label']
        if book_id in bookid_to_row and label in label_to_idx:
            y_multilabel[bookid_to_row[book_id], label_to_idx[label]] = 1.0

    np.save(DATA_DIR / 'y_multilabel.npy', y_multilabel)
    print(f"y_multilabel shape: {y_multilabel.shape}")
    print(f"avg labels per book: {y_multilabel.sum(axis=1).mean():.2f}")

if __name__ == "__main__":
    build_embeddings_and_labels()