import json
import time
import numpy as np

from data_utils import get_u_fold_paths, load_ratings
from evaluation import evaluate_memory_based, evaluate_model_based

RESULTS_DIR = "../results"

def search_memory_based():
    train_df = load_ratings(get_u_fold_paths(1)[0])
    test_df = load_ratings(get_u_fold_paths(1)[1])

    results = []

    print("=== Busca de hiperparâmetros: User-based KNN ===")
    for similarity in ["cosine", "pearson"]:
        for k in [5, 10, 20, 30, 50, 80, 120]:
            t0 = time.time()
            res = evaluate_memory_based(train_df, test_df, mode="user", similarity=similarity, k=k)
            elapsed = time.time() - t0
            print(f"  user-based sim={similarity:8s} k={k:4d}  "
                  f"RMSE={res['rmse']:.4f}  MAE={res['mae']:.4f}  ({elapsed:.1f}s)")
            results.append({"mode": "user", "similarity": similarity, "k": k, "rmse": res["rmse"], "mae": res["mae"]})

    print("\n=== Busca de hiperparâmetros: Item-based KNN ===")
    for similarity in ["cosine", "pearson"]:
        for k in [5, 10, 20, 30, 50, 80, 120]:
            t0 = time.time()
            res = evaluate_memory_based(train_df, test_df, mode="item", similarity=similarity, k=k)
            elapsed = time.time() - t0
            print(f"  item-based sim={similarity:8s} k={k:4d}  "
                  f"RMSE={res['rmse']:.4f}  MAE={res['mae']:.4f}  ({elapsed:.1f}s)")
            results.append({"mode": "item", "similarity": similarity, "k": k, "rmse": res["rmse"], "mae": res["mae"]})

    with open(f"{RESULTS_DIR}/hp_search_memory.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

def search_model_based():
    train_df = load_ratings(get_u_fold_paths(1)[0])
    test_df = load_ratings(get_u_fold_paths(1)[1])

    results = []

    print("\n=== Busca de hiperparâmetros: Regularized SVD ===")
    print("-- Variando n_factors (lr=0.005, reg=0.02, epochs=30) --")
    for n_factors in [5, 10, 20, 30, 50, 100]:
        t0 = time.time()
        res = evaluate_model_based(train_df, test_df, n_factors=n_factors, lr=0.005, reg=0.02, n_epochs=30)
        elapsed = time.time() - t0
        print(f"  factors={n_factors:4d}  RMSE={res['rmse']:.4f}  MAE={res['mae']:.4f}  ({elapsed:.1f}s)")
        results.append({"param": "n_factors", "value": n_factors,
                         "lr": 0.005, "reg": 0.02, "n_epochs": 30,
                         "n_factors": n_factors,
                         "rmse": res["rmse"], "mae": res["mae"]})

    print("-- Variando reg (lambda) (n_factors=20, lr=0.005, epochs=30) --")
    for reg in [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        t0 = time.time()
        res = evaluate_model_based(train_df, test_df, n_factors=20, lr=0.005, reg=reg, n_epochs=30)
        elapsed = time.time() - t0
        print(f"  reg={reg:6.3f}  RMSE={res['rmse']:.4f}  MAE={res['mae']:.4f}  ({elapsed:.1f}s)")
        results.append({"param": "reg", "value": reg,
                         "lr": 0.005, "reg": reg, "n_epochs": 30, "n_factors": 20,
                         "rmse": res["rmse"], "mae": res["mae"]})

    print("-- Variando lr (n_factors=20, reg=0.02, epochs=30) --")
    for lr in [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]:
        t0 = time.time()
        res = evaluate_model_based(train_df, test_df, n_factors=20, lr=lr, reg=0.02, n_epochs=30)
        elapsed = time.time() - t0
        print(f"  lr={lr:6.4f}  RMSE={res['rmse']:.4f}  MAE={res['mae']:.4f}  ({elapsed:.1f}s)")
        results.append({"param": "lr", "value": lr,
                         "lr": lr, "reg": 0.02, "n_epochs": 30, "n_factors": 20,
                         "rmse": res["rmse"], "mae": res["mae"]})

    print("-- Variando n_epochs (n_factors=20, lr=0.005, reg=0.02) --")
    for n_epochs in [5, 10, 20, 30, 50, 80]:
        t0 = time.time()
        res = evaluate_model_based(train_df, test_df, n_factors=20, lr=0.005, reg=0.02, n_epochs=n_epochs)
        elapsed = time.time() - t0
        print(f"  epochs={n_epochs:4d}  RMSE={res['rmse']:.4f}  MAE={res['mae']:.4f}  ({elapsed:.1f}s)")
        results.append({"param": "n_epochs", "value": n_epochs,
                         "lr": 0.005, "reg": 0.02, "n_epochs": n_epochs, "n_factors": 20,
                         "rmse": res["rmse"], "mae": res["mae"]})

    with open(f"{RESULTS_DIR}/hp_search_model.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    mem_results = search_memory_based()
    model_results = search_model_based()

    best_user = min([r for r in mem_results if r["mode"] == "user"], key=lambda r: r["rmse"])
    best_item = min([r for r in mem_results if r["mode"] == "item"], key=lambda r: r["rmse"])
    print("\n=== Melhores configurações (holdout, fold u1) ===")
    print("User-based:", best_user)
    print("Item-based:", best_item)

    by_factors = [r for r in model_results if r["param"] == "n_factors"]
    best_factors = min(by_factors, key=lambda r: r["rmse"])
    print("Melhor n_factors (SVD):", best_factors)
