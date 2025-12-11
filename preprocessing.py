#preprocessing.py

import os
import re
import html
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from nltk.corpus import wordnet as wn 

RAW_DATA_PATH = "BooksDataset.csv"        # raw dataset
HYPERNYM_DF_PATH = "df_exploded.pkl"      # preprocessed data output
CORE_DATA_PATH = "core_data.pkl"          # stored data read from dataset    

RANDOM_STATE = 7

PROTECTED_CATEGORIES = {
    # Fiction types
    "fiction", "science fiction", "fantasy", "mystery and detective", "romance",
    "thriller", "suspense", "historical", "horror", "western", "literary",
    # Age-specific
    "juvenile fiction", "juvenile nonfiction", "young adult fiction",
    "young adult nonfiction", "adult",
    # Major nonfiction genres
    "biography and autobiography", "history", "religion", "philosophy",
    "science", "reference", "education", "business and economic",
    # Art forms
    "poetry", "drama", "essay", "short story",
    # Core subject areas
    "political science", "social science", "psychology", "medical",
    "technology and engineering", "mathematics", "computer",
}

def split_categories(category_string: str) -> List[str]:
    """
    Split a raw category string into tokenized category labels.
    Mirrors your earlier logic: strip, drop empties, ignore obvious junk.
    """
    if pd.isna(category_string):
        return []

    category_string = str(category_string).strip()
    if category_string == "":
        return []

    genre_tokens = [p.strip() for p in category_string.split(",")]
    genre_tokens = [p for p in genre_tokens if p]

    cleaned_tokens = []
    for p in genre_tokens:
        low = p.lower()
        if low in {"nan", "none"}:
            continue
        # drop the 'general' category as it is uninformative
        if low == "general":
            continue
        cleaned_tokens.append(low)  # normalize to lowercase
    return cleaned_tokens


def clean_description(text: str) -> str:
    """
    Basic cleaning for description text before TF-IDF / SBERT embedding.
    Preserves semantic content while removing noise.
    """
    if pd.isna(text) or text == "":
        return ""

    text = str(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Decode HTML entities (e.g., &amp; → &, &quot; → ")
    text = html.unescape(text)

    # Fix common encoding issues
    text = text.replace("â€™", "'")
    text = text.replace("â€œ", '"')
    text = text.replace("â€", '"')
    text = text.replace("â€\"", "—")

    # Collapse multiple whitespaces/newlines to single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def get_meaningful_hypernym(word: str, max_depth: int = 4) -> str | None:
    MANUAL_HYPERNYMS = {
        # Sports
        "baseball": "sport",
        "basketball": "sport",
        "football": "sport",
        "golf": "sport",
        "soccer": "sport",
        "tennis": "sport",
        "hockey": "sport",
        "volleyball": "sport",
        "boxing": "sport",
        "wrestling": "sport",
        "swimming": "sport",
        "cycling": "sport",
        "running": "sport",
        "fishing": "sport",
        "sport and recreation": "sport",

        # Food/Cooking
        "cooking": "food",
        "cook": "food",
        "pasta": "food",
        "dessert": "food",
        "beverage": "food",
        "wine": "food",
        "bread": "food",
        "soup": "food",
        "salad": "food",

        # Religion
        "christian": "religion",
        "religious": "religion",
        "christianity": "religion",

        # Fiction/Mystery - map to fiction parent
        "mystery": "fiction",
        "thriller": "fiction",
        "contemporary": "fiction",

        # Social sciences
        "historical": "history",
        "political": "political science",
        "psychology": "social science",
        "sociology": "social science",
    }

    if word in MANUAL_HYPERNYMS:
        return MANUAL_HYPERNYMS[word]

    MEANINGFUL_TERMS = {
        "fiction", "nonfiction",
        "sport", "athletics", "game", "athletic game", "field game",
        "court game", "ball game", "outdoor sport", "contact sport",
        "food", "dish", "nutriment", "course", "ingredient", "produce",
        "meal", "foodstuff", "beverage", "drink", "nourishment",
        "science", "natural science", "social science", "applied science",
        "art", "fine art", "music", "dance", "performing art", "visual art",
        "craft", "hobby", "health", "medicine", "medical specialty",
        "education", "business", "religion", "philosophy", "travel",
        "animal", "plant", "nature", "technology", "computer",
        "drama", "comedy", "activity", "diversion", "entertainment",
        "history", "political science",
        "life science", "physical science", "therapy",
        "instruction", "learning", "discipline", "commerce", "trade",
        "organism", "living thing", "flora", "vehicle", "building",
    }

    REJECT_TERMS = {
        "writing", "written communication", "literary composition",
        "abstraction", "entity", "physical entity", "object", "whole",
        "thing", "matter", "substance", "relation", "communication",
        "social relation", "attribute", "psychological feature",
        "cognition", "content", "message", "state", "possession",
    }

    compound_synsets = wn.synsets(word.replace(" ", "_"), pos="n")
    if compound_synsets:
        synsets = compound_synsets
    else:
        first_word = word.split()[0]
        synsets = wn.synsets(first_word, pos="n")

    if not synsets:
        return None

    for synset in synsets[:3]:
        current = synset
        for _depth in range(max_depth + 1):
            hypernyms = current.hypernyms()
            if not hypernyms:
                break

            hyper = hypernyms[0]
            hyper_name = hyper.name().split(".")[0].replace("_", " ")

            if hyper_name in REJECT_TERMS:
                break

            if hyper_name in MEANINGFUL_TERMS:
                return hyper_name

            current = hyper

    return None


def build_hypernym_mapping_from_df(
    df: pd.DataFrame,
    category_col: str,
    min_freq: int = 5,
) -> Dict[str, str]:

    all_tokens: List[str] = []
    for raw in df[category_col]:
        all_tokens.extend(split_categories(raw))

    freq = Counter(all_tokens)
    filtered_categories = [cat for cat, c in freq.items() if c >= min_freq]

    hypernym_mapping: Dict[str, str] = {}
    unmapped_categories: List[str] = []
    protected_count = 0
    wordnet_count = 0

    for cat in filtered_categories:
        # normalize to lowercase
        cat_norm = cat.lower()

        # skip protected categories
        if cat_norm in PROTECTED_CATEGORIES:
            hypernym_mapping[cat_norm] = cat_norm
            protected_count += 1
            continue

        hyper = get_meaningful_hypernym(cat_norm)
        if hyper:
            hypernym_mapping[cat_norm] = hyper
            wordnet_count += 1
        else:
            hypernym_mapping[cat_norm] = cat_norm
            unmapped_categories.append(cat_norm)

    print("\nHypernym mapping statistics:")
    print(f"  Total frequent categories processed: {len(hypernym_mapping)}")
    print(f"  Protected (kept as-is): {protected_count}")
    print(f"  Mapped to hypernyms:   {len([k for k, v in hypernym_mapping.items() if k != v])}")
    print(f"  Unmapped (kept orig):  {len(unmapped_categories)}")


    hypernym_groups = defaultdict(list)
    for orig, hyper in hypernym_mapping.items():
        hypernym_groups[hyper].append(orig)

    sorted_groups = sorted(hypernym_groups.items(), key=lambda x: len(x[1]), reverse=True)
    print("\nTop 10 hypernym groupings:")
    for hyper, originals in sorted_groups[:10]:
        print(f"  '{hyper}' ← {len(originals)} categories")

    if unmapped_categories:
        print("\nSample unmapped categories:")
        for cat in unmapped_categories[:10]:
            print(f"  - {cat}")

    return hypernym_mapping


def apply_hypernyms_to_row(
    raw_category_string: str,
    hypernym_mapping: Dict[str, str],
) -> List[str]:
    tokens = split_categories(raw_category_string)
    if not tokens:
        return []

    mapped: List[str] = []
    for t in tokens:
        t_norm = t.lower()
        if t_norm in PROTECTED_CATEGORIES:
            mapped.append(t_norm)
        else:
            mapped.append(hypernym_mapping.get(t_norm, t_norm))

    return sorted(set(mapped))


def build_or_load_hypernym_df(
    raw_path: str = RAW_DATA_PATH,
    pickle_path: str = HYPERNYM_DF_PATH,
    desc_col: str = "Description",
    category_col: str = "Category",
    min_freq: int = 5,
) -> pd.DataFrame:
    """
    Stage 1:
    - Load raw BooksDataset.csv.
    - Clean description text (HTML removal, decoding, whitespace fixes).
    - Tokenize each book’s category string into normalized label tokens.
    - Build a hypernym mapping using:
        * Protected categories preserved as-is,
        * Manual overrides,
        * WordNet hypernym lookup for frequent categories.
    - Map each book’s category list through the hypernym mapping.
    - Filter out rows with no valid labels or empty descriptions.
    - Construct dataframe with cleaned descriptions and mapped categories(hypernym mapped) 
    - Save preprocessed dataframe to df_exploded.pkl.
    """


    if os.path.exists(pickle_path):
        print(f"[Stage 1] Loading existing hypernym df from {pickle_path}")
        return pd.read_pickle(pickle_path)

    print(f"[Stage 1] Loading raw data from {raw_path}")
    raw_df = pd.read_csv(raw_path)

    if desc_col not in raw_df.columns:
        raise ValueError(f"Description column '{desc_col}' not found in data.")
    if category_col not in raw_df.columns:
        raise ValueError(f"Category column '{category_col}' not found in data.")

    hypernym_mapping = build_hypernym_mapping_from_df(
        raw_df,
        category_col=category_col,
        min_freq=min_freq,
    )

    descriptions: List[str] = []
    categories_list: List[List[str]] = []

    for _, row in raw_df.iterrows():
        desc_raw = row[desc_col]
        cat_raw = row[category_col]

        desc_clean = clean_description(desc_raw)
        cats_mapped = apply_hypernyms_to_row(cat_raw, hypernym_mapping)

        if desc_clean == "" or len(cats_mapped) == 0:
            continue

        descriptions.append(desc_clean)
        categories_list.append(cats_mapped)

    df_exploded = pd.DataFrame(
        {
            "description": descriptions,
            "categories": categories_list,
        }
    )

    print(f"[Stage 1] Final dataset:")
    print(f"  Rows (books): {len(df_exploded):,}")
    avg_labels = sum(len(c) for c in categories_list) / max(len(categories_list), 1)
    print(f"  Avg labels per book: {avg_labels:.2f}")

    df_exploded.to_pickle(pickle_path)
    print(f"[Stage 1] Saved hypernym df to {pickle_path}")

    return df_exploded

def get_or_create_core_data() -> dict:
    """
    Stage 2:
    - Load df_exploded from Stage 1(description + categories)
    - Build multilabel matrix Y with MultiLabelBinarizer
    - Build class_names
    - Create train/val/test index splits using the random seed value
    - Save everything to core_data.pkl
    """

    if os.path.exists(CORE_DATA_PATH):
        saved = joblib.load(CORE_DATA_PATH)

        if saved.get("random_state") == RANDOM_STATE:
            print(f"[Stage 2] Bypassing dataframe creation and loading core data from {CORE_DATA_PATH}")
            return saved

    df_exploded = build_or_load_hypernym_df()
    descriptions = df_exploded["description"].tolist()
    categories_all = df_exploded["categories"].tolist()

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(categories_all)
    class_names = mlb.classes_
    n_samples = Y.shape[0]

    print(f"[Stage 2] Built label matrix Y with shape {Y.shape}")
    print(f"[Stage 2] Number of labels: {len(class_names)}")

    indices = np.arange(n_samples)

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.2, 
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5, 
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    print(f"[Stage 2] Split sizes:")
    print(f"  Train: {len(train_idx)}")
    print(f"  Val:   {len(val_idx)}")
    print(f"  Test:  {len(test_idx)}")

    core = {
        "df": df_exploded,        
        "Y": Y,
        "class_names": class_names,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "mlb": mlb,
        "random_state": RANDOM_STATE,
    }

    joblib.dump(core, CORE_DATA_PATH)
    print(f"[Stage 2] Saved core data to {CORE_DATA_PATH}")

    return core



