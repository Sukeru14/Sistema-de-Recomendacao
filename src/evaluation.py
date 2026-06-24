import numpy as np
import time

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))

def df_to_zero_indexed_triples(df):
    arr = df[["user_id", "item_id", "rating"]].to_numpy(dtype=np.float64)
    arr[:, 0] -= 1
    arr[:, 1] -= 1
    return arr

def evaluate_memory_based(train_df, test_df, mode="user", similarity="cosine", k=30, n_users=943, n_items=1682):
    from data_utils import build_user_item_matrix
    from memory_based import MemoryBasedCF

    t0 = time.time()
    train_matrix = build_user_item_matrix(train_df, n_users, n_items)

    model = MemoryBasedCF(mode=mode, similarity=similarity, k=k)
    model.fit(train_matrix)

    test_pairs = list(zip(test_df["user_id"] - 1, test_df["item_id"] - 1))
    y_pred = model.predict_batch(test_pairs)
    y_true = test_df["rating"].to_numpy()

    elapsed = time.time() - t0
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "time_sec": elapsed,
    }

def evaluate_model_based(train_df, test_df, n_factors=20, lr=0.005, reg=0.02, n_epochs=20, n_users=943, n_items=1682, random_state=42):
    from model_based import RegularizedSVD

    t0 = time.time()
    train_triples = df_to_zero_indexed_triples(train_df)

    model = RegularizedSVD(n_factors=n_factors, lr=lr, reg=reg, n_epochs=n_epochs, random_state=random_state)
    model.fit(train_triples, n_users, n_items)

    test_pairs = list(zip(test_df["user_id"] - 1, test_df["item_id"] - 1))
    y_pred = model.predict_batch(test_pairs)
    y_true = test_df["rating"].to_numpy()

    elapsed = time.time() - t0
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "time_sec": elapsed,
        "train_rmse_history": model.train_rmse_history_,
    }

def cross_validate(folds, eval_fn, **kwargs):
    rmses, maes, times = [], [], []
    for fold_id, (train_df, test_df) in enumerate(folds, start=1):
        result = eval_fn(train_df, test_df, **kwargs)
        rmses.append(result["rmse"])
        maes.append(result["mae"])
        times.append(result["time_sec"])
        print(f"    Fold {fold_id}: RMSE={result['rmse']:.4f}  MAE={result['mae']:.4f}  "
              f"({result['time_sec']:.1f}s)")

    return {
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "time_mean": float(np.mean(times)),
        "rmse_per_fold": rmses,
        "mae_per_fold": maes,
    }
