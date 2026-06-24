import json
import time
import numpy as np

from data_utils import load_all_folds, build_user_item_matrix
from evaluation import df_to_zero_indexed_triples
from memory_based import MemoryBasedCF
from model_based import RegularizedSVD
from ranking_evaluation import evaluate_ranking

RESULTS_DIR = "../results"
K_VALUES = (5, 10, 20)

BEST_USER = dict(mode="user", similarity="pearson", k=30)
BEST_ITEM = dict(mode="item", similarity="pearson", k=30)
BEST_SVD = dict(n_factors=50, lr=0.005, reg=0.05, n_epochs=30)

def cv_ranking_memory(folds, mem_kwargs, n_users=943, n_items=1682):
    per_fold = {k: {"precision_at_k": [], "recall_at_k": [], "ndcg_at_k": [], "map_at_k": []} for k in K_VALUES}

    for fold_id, (train_df, test_df) in enumerate(folds, start=1):
        train_matrix = build_user_item_matrix(train_df, n_users, n_items)
        model = MemoryBasedCF(**mem_kwargs)
        model.fit(train_matrix)

        res = evaluate_ranking(model, train_df, test_df, k_values=K_VALUES)
        for k in K_VALUES:
            for metric in ["precision_at_k", "recall_at_k", "ndcg_at_k", "map_at_k"]:
                per_fold[k][metric].append(res[k][metric])
        print(f"    Fold {fold_id} concluído.")

    return _summarize(per_fold)

def cv_ranking_svd(folds, svd_kwargs, n_users=943, n_items=1682):
    per_fold = {k: {"precision_at_k": [], "recall_at_k": [], "ndcg_at_k": [], "map_at_k": []} for k in K_VALUES}

    for fold_id, (train_df, test_df) in enumerate(folds, start=1):
        triples = df_to_zero_indexed_triples(train_df)
        model = RegularizedSVD(**svd_kwargs)
        model.fit(triples, n_users, n_items)

        res = evaluate_ranking(model, train_df, test_df, k_values=K_VALUES)
        for k in K_VALUES:
            for metric in ["precision_at_k", "recall_at_k", "ndcg_at_k", "map_at_k"]:
                per_fold[k][metric].append(res[k][metric])
        print(f"    Fold {fold_id} concluído.")

    return _summarize(per_fold)

def _summarize(per_fold):
    summary = {}
    for k in K_VALUES:
        summary[k] = {}
        for metric, values in per_fold[k].items():
            summary[k][f"{metric}_mean"] = float(np.mean(values))
            summary[k][f"{metric}_std"] = float(np.std(values))
    return summary

def main():
    folds = load_all_folds()
    final_results = {}

    print(f"=== Avaliação de Ranking via 5-fold CV (K={K_VALUES}) ===\n")

    print(f"-- User-based KNN {BEST_USER} --")
    t0 = time.time()
    final_results["user_based"] = cv_ranking_memory(folds, BEST_USER)
    print(f"  ({time.time()-t0:.1f}s)\n")

    print(f"-- Item-based KNN {BEST_ITEM} --")
    t0 = time.time()
    final_results["item_based"] = cv_ranking_memory(folds, BEST_ITEM)
    print(f"  ({time.time()-t0:.1f}s)\n")

    print(f"-- Regularized SVD {BEST_SVD} --")
    t0 = time.time()
    final_results["svd"] = cv_ranking_svd(folds, BEST_SVD)
    print(f"  ({time.time()-t0:.1f}s)\n")

    with open(f"{RESULTS_DIR}/final_ranking_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print("=== Resumo Final (Precision@K / Recall@K / NDCG@K) ===")
    header = f"{'Algoritmo':<18}"
    for k in K_VALUES:
        header += f"{'P@'+str(k):>10}{'R@'+str(k):>10}{'NDCG@'+str(k):>10}"
    print(header)

    for name, label in [("user_based", "User-based KNN"), ("item_based", "Item-based KNN"), ("svd", "Regularized SVD")]:
        row = f"{label:<18}"
        for k in K_VALUES:
            m = final_results[name][k]
            row += f"{m['precision_at_k_mean']:>10.4f}{m['recall_at_k_mean']:>10.4f}{m['ndcg_at_k_mean']:>10.4f}"
        print(row)

    return final_results

if __name__ == "__main__":
    main()
