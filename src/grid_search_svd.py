import json
import time
import numpy as np

from data_utils import get_u_fold_paths, load_ratings
from evaluation import evaluate_model_based

RESULTS_DIR = "../results"

def main():
    train_df = load_ratings(get_u_fold_paths(1)[0])
    test_df = load_ratings(get_u_fold_paths(1)[1])

    factors_grid = [5, 10, 20, 30, 50, 100]
    reg_grid = [0.005, 0.01, 0.02, 0.05, 0.1]

    results = []
    print("=== Grid Search conjunto: n_factors x reg (lr=0.005, epochs=30) ===\n")
    print(f"{'n_factors':>10}{'reg':>8}{'RMSE':>10}{'MAE':>10}{'tempo':>8}")

    for n_factors in factors_grid:
        for reg in reg_grid:
            t0 = time.time()
            res = evaluate_model_based(train_df, test_df, n_factors=n_factors, lr=0.005, reg=reg, n_epochs=30)
            elapsed = time.time() - t0
            print(f"{n_factors:>10}{reg:>8.3f}{res['rmse']:>10.4f}{res['mae']:>10.4f}{elapsed:>7.1f}s")
            results.append({
                "n_factors": n_factors, "reg": reg, "lr": 0.005, "n_epochs": 30,
                "rmse": res["rmse"], "mae": res["mae"],
            })

    with open(f"{RESULTS_DIR}/grid_search_svd.json", "w") as f:
        json.dump(results, f, indent=2)

    best = min(results, key=lambda r: r["rmse"])
    print("\n=== Melhor combinação encontrada ===")
    print(best)

    #Compara com a config "individual" usada anteriormente (n_factors=20, reg=0.02)
    baseline_individual = next(
        r for r in results if r["n_factors"] == 20 and r["reg"] == 0.02
    )
    print("\n=== Comparação com a configuração da busca individual (n_factors=20, reg=0.02) ===")
    print(f"Busca individual:  RMSE={baseline_individual['rmse']:.4f}")
    print(f"Melhor da grade:    RMSE={best['rmse']:.4f}")
    print(f"Diferença:          {baseline_individual['rmse'] - best['rmse']:.4f}")

    return results, best

if __name__ == "__main__":
    main()
