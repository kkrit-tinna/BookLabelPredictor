import pandas as pd

BOOKS_DATASET_PATH = "BooksDataset.csv"

def load_descriptions_genres(csv_path):

    df = pd.read_csv(csv_path)
    
    df["Description"] = df["Description"].astype(str).str.strip()
    df["Genre"] = df["Genre"].astype(str).str.strip()

    df.replace({"": pd.NA}, inplace=True)
    
    df_cleaned = df.dropna(subset=["Description", "Genre"])
    
    desc_to_genre = df_cleaned.set_index("Description")["Genre"].to_dict()
    
    return desc_to_genre

mapping = load_descriptions_genres(BOOKS_DATASET_PATH)

print(f"Total of {len(mapping)} descriptions with genres")