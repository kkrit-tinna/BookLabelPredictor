import os
import re
import html
from typing import Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd
import spacy
import nltk
from nltk.corpus import wordnet as wn

nltk.download('wordnet', quiet = True)
nltk.download('omw-1.4', quiet = True)

try:
    nlp = spacy.load("en_core_web_sm")
    print("Loaded spaCy model: en_core_web_sm")
except Exception:
    nlp = None
    print("spaCy model not available; lemmatization will be skipped")

PROTECTED_CATEGORIES: Set[str] = {
    'fiction', 'science fiction', 'fantasy', 'mystery and detective', 'romance',
    'thriller', 'suspense', 'historical', 'horror', 'western', 'literary',
    'juvenile fiction', 'juvenile nonfiction', 'young adult fiction',
    'young adult nonfiction', 'adult',
    'biography and autobiography', 'history', 'religion', 'philosophy',
    'science', 'reference', 'education', 'business and economic',
    'poetry', 'drama', 'essay', 'short story',
    'political science', 'social science', 'psychology', 'medical',
    'technology and engineering', 'mathematics', 'computer'
}

MANUAL_HYPERNYMS: Dict[str, str] = {
    'baseball': 'sport', 'basketball': 'sport', 'football': 'sport', 'golf': 'sport',
    'soccer': 'sport', 'tennis': 'sport', 'hockey': 'sport', 'volleyball': 'sport',
    'boxing': 'sport', 'wrestling': 'sport', 'swimming': 'sport', 'cycling': 'sport',
    'running': 'sport', 'fishing': 'sport', 'sport and recreation': 'sport',
    'cooking': 'food', 'cook': 'food', 'pasta': 'food', 'dessert': 'food',
    'beverage': 'food', 'wine': 'food', 'bread': 'food', 'soup': 'food',
    'christian': 'religion', 'religious': 'religion', 'christianity': 'religion',
    'mystery': 'fiction', 'thriller': 'fiction', 'contemporary': 'fiction',
    'historical': 'history', 'political': 'political science', 'psychology': 'social science',
    'sociology': 'social science'
}

MEANINGFUL_TERMS: Set[str] = {
    'fiction', 'nonfiction', 'sport', 'food', 'science', 'art', 'music', 'health',
    'medicine', 'education', 'business', 'religion', 'history', 'political science',
    'technology', 'computer', 'reference', 'poetry', 'drama', 'travel', 'animal',
    'plant', 'nature', 'law', 'philosophy'
}

REJECT_TERMS: Set[str] = {
    'thing', 'object', 'entity', 'matter', 'whole'
}

# This file contains preprocessing functions for TF-IDF and LSA

def clean_description(text: Optional[str]) -> str:
    if pd.isna(text) or text == '':
        return ''
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = html.unescape(text)
    text = text.replace('â€™', "'").replace('â€œ', '"').replace('â€', '"').replace('â€"', '—')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_category(cat: Optional[str]) -> Optional[str]:
    if pd.isna(cat) or str(cat).strip() == '':
        return None
    s = str(cat).strip()
    s = s.replace(' & ', ', ')
    parts = [p.strip() for p in s.split(',') if p and p.strip()]
    cleaned: List[str] = []
    for c in parts:
        low = c.lower()
        if low in {'nan', 'none', 'general'}:
            continue
        c = re.sub(r'\(.*?\)', '', c)
        c = re.sub(r'[\(\)\d]+', '', c)
        c = c.replace("'s", 's').replace('-', ' ')
        c = ' '.join(c.split()).strip()
        if c:
            cleaned.append(c.lower())
    return ', '.join(cleaned) if cleaned else None


def lemmatize_category(cat: Optional[str]) -> Optional[str]:
    if pd.isna(cat) or not cat:
        return None
    if nlp is None:
        return cat
    doc = nlp(cat)
    lemmas = [token.lemma_ for token in doc if not token.is_punct]
    canonical = ' '.join(lemmas).strip()
    return canonical.lower() if canonical else None


def get_meaningful_hypernym(word: str, max_depth: int = 4) -> Optional[str]:
    if word in MANUAL_HYPERNYMS:
        return MANUAL_HYPERNYMS[word]
    if word in PROTECTED_CATEGORIES or word in MEANINGFUL_TERMS:
        return word
    synsets = wn.synsets(word.replace(' ', '_'), pos = 'n')
    if not synsets:
        first = word.split()[0]
        synsets = wn.synsets(first, pos = 'n')
    if not synsets:
        return None
    for syn in synsets[: 3]:
        current = syn
        for _ in range(max_depth + 1):
            hypers = current.hypernyms()
            if not hypers:
                break
            hyper = hypers[0]
            hyper_name = hyper.name().split('.')[0].replace('_', ' ')
            if hyper_name in REJECT_TERMS:
                break
            if hyper_name in MEANINGFUL_TERMS:
                return hyper_name
            current = hyper
    return None


def build_hypernym_mapping(categories: Iterable[str], protected: Optional[Set[str]] = None) -> Dict[str, str]:
    if protected is None:
        protected = PROTECTED_CATEGORIES
    mapping: Dict[str, str] = {}
    unmapped: List[str] = []
    for cat in categories:
        if cat in mapping:
            continue
        if cat in protected or cat.lower() in protected:
            mapping[cat] = cat
            continue
        hyper = get_meaningful_hypernym(cat)
        if hyper:
            mapping[cat] = hyper
        else:
            mapping[cat] = cat
            unmapped.append(cat)
    print(f"Built hypernym mapping: total = {len(mapping)}, unmapped = {len(unmapped)}")
    return mapping


def load_descriptions_genres(csv_path: str, min_frequency: int = 4, cache_path: Optional[str] = None, force_rebuild: bool = False) -> pd.DataFrame:
    if cache_path is None:
        cache_path = csv_path + '.cleaned.pkl'
    if (not force_rebuild) and os.path.exists(cache_path):
        print(f"Loading cleaned dataframe from cache: {cache_path}")
        df_cached = pd.read_pickle(cache_path)
        print(f"Loaded cached dataframe with {len(df_cached)} rows")
        return df_cached

    print(f"Reading raw CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Original rows: {len(df)}")

    df = df.dropna(subset = ['Category', 'Description']).reset_index(drop = True)
    df['Description'] = df['Description'].astype(str).str.strip()
    df['Category'] = df['Category'].astype(str).str.strip()

    print(f"After dropping missing categories/descriptions: {df.shape}")

    df = df[df['Description'].astype(str).str.lower().isin(['', 'nan', 'none']) == False].reset_index(drop = True)
    print(f"After removing literal empty/'nan' descriptions: {len(df)}")

    df['categories'] = df['Category'].apply(preprocess_category)
    df = df[df['categories'].map(lambda x: isinstance(x, str) and len(x) > 0)].reset_index(drop = True)
    print(f"After preprocessing categories: {df.shape}")

    df['book_id'] = df.index
    df_exploded = df.copy()
    df_exploded['Category_Split'] = df_exploded['categories'].str.split(', ')
    df_exploded = df_exploded.explode('Category_Split').reset_index(drop = True)
    df_exploded = df_exploded[df_exploded['Category_Split'].notna()].reset_index(drop = True)
    print(f"After exploding categories: {df_exploded.shape}")

    df_exploded['Category_Lemmatized'] = df_exploded['Category_Split'].apply(lemmatize_category)
    df_exploded = df_exploded[df_exploded['Category_Lemmatized'].notna()].reset_index(drop = True)
    print(f"After lemmatization/filtering: {df_exploded.shape}")

    df_exploded = df_exploded.drop_duplicates(subset = ['book_id', 'Category_Lemmatized']).reset_index(drop = True)
    print(f"After deduplication: {df_exploded.shape}")

    freq = df_exploded['Category_Lemmatized'].value_counts()
    keep = set(freq[freq >= min_frequency].index.tolist())
    df_exploded = df_exploded[df_exploded['Category_Lemmatized'].isin(keep)].reset_index(drop = True)
    print(f"After frequency filter (min_frequency = {min_frequency}): {df_exploded.shape}")

    unique_cats = sorted(df_exploded['Category_Lemmatized'].unique().tolist())
    print(f"Unique lemmatized categories to map: {len(unique_cats)}")

    hypernym_map = build_hypernym_mapping(unique_cats)

    df_exploded['Hypernym_Label'] = df_exploded['Category_Lemmatized'].map(hypernym_map)
    df_exploded['Hypernym_Label'] = df_exploded['Hypernym_Label'].astype(str).str.strip().str.lower()
    df_exploded = df_exploded.drop_duplicates(subset=['book_id', 'Hypernym_Label']).reset_index(drop=True)


    df_exploded['Description_Cleaned'] = df_exploded['Description'].apply(clean_description)
    print(f"Descriptions cleaned. Total rows: {len(df_exploded)}")

    grouped = df_exploded.groupby('book_id').agg({
        'Description_Cleaned': 'first',
        'Hypernym_Label': lambda vals: sorted(set(vals.tolist()))
    }).reset_index()

    cleaned_df = grouped.rename(columns = {'Description_Cleaned': 'description', 'Hypernym_Label': 'categories'})
    cleaned_df = cleaned_df[cleaned_df['description'].astype(str).str.strip() != ''].reset_index(drop = True)
    print(f"Aggregated cleaned dataframe shape: {cleaned_df.shape}")

    cleaned_df.to_pickle(cache_path)
    print(f"Saved cleaned dataframe to cache: {cache_path}")

    return cleaned_df

def _accumulate_categories(series):
    out = set()
    for val in series:
        parts = [p.strip() for p in val.split(',') if p and p.strip()]
        for p in parts:
            out.add(p.strip().lower())
    return sorted(out)




def load_descriptions_genres_simple(csv_path: str, cache_path: Optional[str] = None) -> pd.DataFrame:
    if cache_path is None:
        cache_path = csv_path + '.simple.cleaned.pkl'
        print(f"Loading simple cleaned dataframe from cache: {cache_path}")
        df_cached = pd.read_pickle(cache_path)
        print(f"Loaded simple cached dataframe with {len(df_cached)} rows")
        return df_cached

    print(f"Reading raw CSV for simple preprocess: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Original rows: {len(df)}")

    df = df.dropna(subset = ['Category', 'Description']).reset_index(drop = True)
    df['Description'] = df['Description'].astype(str).str.strip()
    df['Category'] = df['Category'].astype(str).str.strip()
    df = df[df['Description'].astype(str).str.lower().isin(['', 'nan', 'none']) == False].reset_index(drop = True)
    df['categories'] = df['Category'].apply(_accumulate_categories)
    df = df[df['categories'].map(lambda x: isinstance(x, str) and len(x) > 0)].reset_index(drop = True)
    df['book_id'] = df.index
    df['Description_Cleaned'] = df['Description'].apply(clean_description)