import numpy as np
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ml-100k")

COLS = ["user_id", "item_id", "rating", "timestamp"]

def load_ratings(path):
    return pd.read_csv(path, sep="\t", names=COLS, engine="python")

def load_full_dataset():
    return load_ratings(os.path.join(DATA_DIR, "u.data"))

def load_movies():
    item_cols = [
        "item_id", "title", "release_date", "video_release_date", "imdb_url",
        "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    ]
    return pd.read_csv(
        os.path.join(DATA_DIR, "u.item"),
        sep="|", names=item_cols, encoding="latin-1", engine="python",
    )

def get_u_fold_paths(fold_id):
    assert fold_id in [1, 2, 3, 4, 5]
    base = os.path.join(DATA_DIR, f"u{fold_id}.base")
    test = os.path.join(DATA_DIR, f"u{fold_id}.test")
    return base, test

def load_all_folds():
    folds = []
    for i in range(1, 6):
        base, test = get_u_fold_paths(i)
        train_df = load_ratings(base)
        test_df = load_ratings(test)
        folds.append((train_df, test_df))
    return folds

def build_user_item_matrix(df, n_users=943, n_items=1682):
    matrix = np.zeros((n_users, n_items), dtype=np.float32)
    for row in df.itertuples(index=False):
        matrix[row.user_id - 1, row.item_id - 1] = row.rating
    return matrix

def matrix_to_mask(matrix):
    return matrix > 0

if __name__ == "__main__":
    df = load_full_dataset()
    print("Dataset completo:", df.shape)
    print(df.head())

    folds = load_all_folds()
    for i, (tr, te) in enumerate(folds, start=1):
        print(f"Fold {i}: treino={tr.shape[0]}  teste={te.shape[0]}")
