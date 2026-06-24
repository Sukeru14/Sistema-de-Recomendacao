import json
import time
import numpy as np

from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import KFold as SurpriseKFold

from data_utils import DATA_DIR
import os

RESULTS_DIR = "../results"

def main():
    reader = Reader(line_format="user item rating timestamp", sep="\t", rating_scale=(1, 5))
    data = Dataset.load_from_file(os.path.join(DATA_DIR, "u.data"), reader=reader)

    kf = SurpriseKFold(n_splits=5, random_state=42)

    algo = SVD(n_factors=20, lr_all=0.005, reg_all=0.02, n_epochs=30, random_state=42)

    rmses, maes, times = [], [], []
    print("=== SVD via scikit-surprise (biblioteca externa) — 5-fold CV ===")
    for fold_id, (trainset, testset) in enumerate(kf.split(data), start=1):
        t0 = time.time()
        algo.fit(trainset)
        predictions = algo.test(testset)
        elapsed = time.time() - t0

        fold_rmse = accuracy.rmse(predictions, verbose=False)
        fold_mae = accuracy.mae(predictions, verbose=False)

        rmses.append(fold_rmse)
        maes.append(fold_mae)
        times.append(elapsed)
        print(f"  Fold {fold_id}: RMSE={fold_rmse:.4f}  MAE={fold_mae:.4f}  ({elapsed:.1f}s)")

    result = {
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "time_mean": float(np.mean(times)),
        "params": {"n_factors": 20, "lr_all": 0.005, "reg_all": 0.02, "n_epochs": 30},
        "source": "scikit-surprise (biblioteca externa, NÃO implementado do zero)",
    }

    print(f"\nRMSE = {result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}")
    print(f"MAE  = {result['mae_mean']:.4f} ± {result['mae_std']:.4f}")

    with open(f"{RESULTS_DIR}/svd_surprise_comparison.json", "w") as f:
        json.dump(result, f, indent=2)

    return result

if __name__ == "__main__":
    main()
