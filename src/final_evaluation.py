import json
import time

from data_utils import load_all_folds
from evaluation import evaluate_memory_based, evaluate_model_based, cross_validate

RESULTS_DIR = "../results"

#Melhores hiperparâmetros encontrados em hyperparam_search.py (holdout fold u1)
BEST_USER = dict(mode="user", similarity="pearson", k=30)
BEST_ITEM = dict(mode="item", similarity="pearson", k=30)

BEST_SVD = dict(n_factors=50, lr=0.005, reg=0.05, n_epochs=30)
BEST_SVD_INDIVIDUAL = dict(n_factors=20, lr=0.005, reg=0.02, n_epochs=30)

def main():
    folds = load_all_folds()
    final_results = {}

    print("=== Avaliação final via 5-fold CV ===\n")

    print(f"-- User-based KNN {BEST_USER} --")
    t0 = time.time()
    res = cross_validate(folds, evaluate_memory_based, **BEST_USER)
    res["time_total"] = time.time() - t0
    res["params"] = BEST_USER
    final_results["user_based"] = res
    print(f"  RMSE = {res['rmse_mean']:.4f} ± {res['rmse_std']:.4f}")
    print(f"  MAE  = {res['mae_mean']:.4f} ± {res['mae_std']:.4f}\n")

    print(f"-- Item-based KNN {BEST_ITEM} --")
    t0 = time.time()
    res = cross_validate(folds, evaluate_memory_based, **BEST_ITEM)
    res["time_total"] = time.time() - t0
    res["params"] = BEST_ITEM
    final_results["item_based"] = res
    print(f"  RMSE = {res['rmse_mean']:.4f} ± {res['rmse_std']:.4f}")
    print(f"  MAE  = {res['mae_mean']:.4f} ± {res['mae_std']:.4f}\n")

    print(f"-- Regularized SVD (config da grade conjunta) {BEST_SVD} --")
    t0 = time.time()
    res = cross_validate(folds, evaluate_model_based, **BEST_SVD)
    res["time_total"] = time.time() - t0
    res["params"] = BEST_SVD
    final_results["svd"] = res
    print(f"  RMSE = {res['rmse_mean']:.4f} ± {res['rmse_std']:.4f}")
    print(f"  MAE  = {res['mae_mean']:.4f} ± {res['mae_std']:.4f}\n")

    print(f"-- Regularized SVD (config da busca individual, para comparação) {BEST_SVD_INDIVIDUAL} --")
    t0 = time.time()
    res = cross_validate(folds, evaluate_model_based, **BEST_SVD_INDIVIDUAL)
    res["time_total"] = time.time() - t0
    res["params"] = BEST_SVD_INDIVIDUAL
    final_results["svd_individual_search"] = res
    print(f"  RMSE = {res['rmse_mean']:.4f} ± {res['rmse_std']:.4f}")
    print(f"  MAE  = {res['mae_mean']:.4f} ± {res['mae_std']:.4f}\n")

    #Baseline simples para referência (média global)
    print("-- Baseline: previsão pela média global de notas --")
    import numpy as np
    rmses, maes = [], []
    for train_df, test_df in folds:
        global_mean = train_df["rating"].mean()
        y_true = test_df["rating"].to_numpy()
        y_pred = np.full_like(y_true, global_mean, dtype=np.float64)
        rmses.append(float(np.sqrt(np.mean((y_true - y_pred) ** 2))))
        maes.append(float(np.mean(np.abs(y_true - y_pred))))
    final_results["baseline_global_mean"] = {
        "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
        "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
    }
    print(f"  RMSE = {final_results['baseline_global_mean']['rmse_mean']:.4f}")
    print(f"  MAE  = {final_results['baseline_global_mean']['mae_mean']:.4f}\n")

    with open(f"{RESULTS_DIR}/final_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print("=== Resumo Final ===")
    print(f"{'Algoritmo':<20}{'RMSE':>12}{'MAE':>12}")
    print(f"{'Baseline (média)':<20}{final_results['baseline_global_mean']['rmse_mean']:>12.4f}"
          f"{final_results['baseline_global_mean']['mae_mean']:>12.4f}")
    print(f"{'User-based KNN':<20}{final_results['user_based']['rmse_mean']:>12.4f}"
          f"{final_results['user_based']['mae_mean']:>12.4f}")
    print(f"{'Item-based KNN':<20}{final_results['item_based']['rmse_mean']:>12.4f}"
          f"{final_results['item_based']['mae_mean']:>12.4f}")
    print(f"{'Regularized SVD':<20}{final_results['svd']['rmse_mean']:>12.4f}"
          f"{final_results['svd']['mae_mean']:>12.4f}")

    return final_results

if __name__ == "__main__":
    main()
