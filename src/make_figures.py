import json
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = "../figures"
RES_DIR = "../results"

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

def load_json(name):
    with open(f"{RES_DIR}/{name}") as f:
        return json.load(f)

def fig_sensitivity_k():
    data = load_json("hp_search_memory.json")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = {
        ("user", "cosine"): ("o-", "User-based (cosseno)"),
        ("user", "pearson"): ("o--", "User-based (Pearson)"),
        ("item", "cosine"): ("s-", "Item-based (cosseno)"),
        ("item", "pearson"): ("s--", "Item-based (Pearson)"),
    }
    for (mode, sim), (style, label) in styles.items():
        subset = [r for r in data if r["mode"] == mode and r["similarity"] == sim]
        subset.sort(key=lambda r: r["k"])
        ks = [r["k"] for r in subset]
        rmses = [r["rmse"] for r in subset]
        ax.plot(ks, rmses, style, label=label, markersize=6)

    ax.set_xlabel("Número de vizinhos (k)")
    ax.set_ylabel("RMSE")
    ax.set_title("Sensibilidade do RMSE ao número de vizinhos (k)\nFiltragem Colaborativa Baseada em Memória")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/sensitivity_k_memory.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Salvo: sensitivity_k_memory.pdf")

def fig_sensitivity_svd():
    data = load_json("hp_search_model.json")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    params_info = [
        ("n_factors", "Número de fatores latentes (k)", axes[0, 0]),
        ("reg", "Regularização (λ)", axes[0, 1]),
        ("lr", "Taxa de aprendizado (lr)", axes[1, 0]),
        ("n_epochs", "Número de épocas", axes[1, 1]),
    ]

    for param, xlabel, ax in params_info:
        subset = [r for r in data if r["param"] == param]
        subset.sort(key=lambda r: r["value"])
        xs = [r["value"] for r in subset]
        rmses = [r["rmse"] for r in subset]
        ax.plot(xs, rmses, "o-", color="darkorange", markersize=6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("RMSE")
        if param == "lr":
            ax.set_xscale("log")
        ax.set_title(f"RMSE vs {xlabel}")

    fig.suptitle("Análise de Sensibilidade de Hiperparâmetros — Regularized SVD", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/sensitivity_svd.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Salvo: sensitivity_svd.pdf")

def fig_learning_curve():
    import sys
    sys.path.insert(0, ".")
    from data_utils import get_u_fold_paths, load_ratings
    from evaluation import evaluate_model_based

    train_df = load_ratings(get_u_fold_paths(1)[0])
    test_df = load_ratings(get_u_fold_paths(1)[1])

    res = evaluate_model_based(train_df, test_df, n_factors=20, lr=0.005, reg=0.02, n_epochs=40)
    history = res["train_rmse_history"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(history) + 1), history, "o-", color="steelblue", markersize=4)
    ax.set_xlabel("Época")
    ax.set_ylabel("RMSE (treino)")
    ax.set_title("Curva de Convergência do SGD — Regularized SVD\n(n_factors=20, lr=0.005, reg=0.02)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/learning_curve_svd.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Salvo: learning_curve_svd.pdf")

def fig_final_comparison():
    data = load_json("final_results.json")
    surprise = load_json("svd_surprise_comparison.json")

    labels = ["Baseline\n(média global)", "User-based\nKNN", "Item-based\nKNN", "Regularized\nSVD\n(do zero)", "SVD\n(scikit-surprise)"]
    rmse_means = [
        data["baseline_global_mean"]["rmse_mean"],
        data["user_based"]["rmse_mean"],
        data["item_based"]["rmse_mean"],
        data["svd"]["rmse_mean"],
        surprise["rmse_mean"],
    ]
    rmse_stds = [
        data["baseline_global_mean"]["rmse_std"],
        data["user_based"]["rmse_std"],
        data["item_based"]["rmse_std"],
        data["svd"]["rmse_std"],
        surprise["rmse_std"],
    ]
    mae_means = [
        data["baseline_global_mean"]["mae_mean"],
        data["user_based"]["mae_mean"],
        data["item_based"]["mae_mean"],
        data["svd"]["mae_mean"],
        surprise["mae_mean"],
    ]
    mae_stds = [
        data["baseline_global_mean"]["mae_std"],
        data["user_based"]["mae_std"],
        data["item_based"]["mae_std"],
        data["svd"]["mae_std"],
        surprise["mae_std"],
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors_rmse = ["gray", "#4C72B0", "#4C72B0", "#DD8452", "#DD8452"]
    colors_mae = ["lightgray", "#8DA9D6", "#8DA9D6", "#F0B27A", "#F0B27A"]

    bars1 = ax.bar(x - width / 2, rmse_means, width, yerr=rmse_stds, capsize=4, label="RMSE", color=colors_rmse)
    bars2 = ax.bar(x + width / 2, mae_means, width, yerr=mae_stds, capsize=4, label="MAE", color=colors_mae)

    ax.set_ylabel("Erro")
    ax.set_title("Comparação Final dos Algoritmos\n(média ± desvio padrão, 5-fold CV)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/final_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Salvo: final_comparison.pdf")

def fig_per_fold_rmse():
    data = load_json("final_results.json")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    folds = [1, 2, 3, 4, 5]

    ax.plot(folds, data["user_based"]["rmse_per_fold"], "o-", label="User-based KNN")
    ax.plot(folds, data["item_based"]["rmse_per_fold"], "s-", label="Item-based KNN")
    ax.plot(folds, data["svd"]["rmse_per_fold"], "^-", label="Regularized SVD")

    ax.set_xlabel("Fold")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE por Fold — Estabilidade entre Partições\n(5-fold CV, folds pré-definidos u1–u5)")
    ax.set_xticks(folds)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/per_fold_rmse.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Salvo: per_fold_rmse.pdf")

def fig_grid_search_heatmap():
    data = load_json("grid_search_svd.json")

    factors_grid = sorted(set(r["n_factors"] for r in data))
    reg_grid = sorted(set(r["reg"] for r in data))

    grid = np.zeros((len(reg_grid), len(factors_grid)))
    for r in data:
        i = reg_grid.index(r["reg"])
        j = factors_grid.index(r["n_factors"])
        grid[i, j] = r["rmse"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto")

    ax.set_xticks(range(len(factors_grid)))
    ax.set_xticklabels(factors_grid)
    ax.set_yticks(range(len(reg_grid)))
    ax.set_yticklabels(reg_grid)
    ax.set_xlabel("Número de fatores latentes (n_factors)")
    ax.set_ylabel("Regularização (λ)")
    ax.set_title("Busca em Grade Conjunta — RMSE do SVD\n(n_factors × λ, lr=0.005, 30 épocas)")

    for i in range(len(reg_grid)):
        for j in range(len(factors_grid)):
            val = grid[i, j]
            best = val == grid.min()
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=9, fontweight="bold" if best else "normal", color="black")
            if best:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="blue", linewidth=3))

    fig.colorbar(im, ax=ax, label="RMSE")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/grid_search_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Salvo: grid_search_heatmap.pdf")

def fig_ranking_comparison():
    data = load_json("final_ranking_results.json")
    k_values = [5, 10, 20]
    algos = [("user_based", "User-based KNN", "#4C72B0"),
             ("item_based", "Item-based KNN", "#55A868"),
             ("svd", "Regularized SVD", "#DD8452")]
    n_folds = 5

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metric_info = [("precision_at_k", "Precision@K"),
                    ("recall_at_k", "Recall@K"),
                    ("ndcg_at_k", "NDCG@K")]

    for ax, (metric_key, metric_label) in zip(axes, metric_info):
        all_means = []
        for algo_key, label, color in algos:
            means = [data[algo_key][str(k)][f"{metric_key}_mean"] for k in k_values]
            sems = [data[algo_key][str(k)][f"{metric_key}_std"] / np.sqrt(n_folds) for k in k_values]
            ax.errorbar(k_values, means, yerr=sems, marker="o", label=label, color=color, capsize=4, markersize=6, linewidth=2)
            all_means.extend(means)
        ax.set_xlabel("K")
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.set_xticks(k_values)
        span = max(all_means) - min(all_means)
        pad = max(span * 1.2, 0.005)
        ax.set_ylim(min(all_means) - pad, max(all_means) + pad)

    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Qualidade da Recomendação (Ranking) — 5-fold CV\n" "(relevante = nota real ≥ 4; barras = erro padrão da média entre folds)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/ranking_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Salvo: ranking_comparison.pdf")

def fig_ranking_bars_k10():
    data = load_json("final_ranking_results.json")
    algos = [("user_based", "User-based\nKNN"), ("item_based", "Item-based\nKNN"),
             ("svd", "Regularized\nSVD")]
    metrics = [("precision_at_k", "Precision@10"), ("recall_at_k", "Recall@10"),
               ("ndcg_at_k", "NDCG@10"), ("map_at_k", "MAP@10")]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics))
    width = 0.25
    colors = ["#4C72B0", "#55A868", "#DD8452"]

    for i, (algo_key, label) in enumerate(algos):
        means = [data[algo_key]["10"][f"{m_key}_mean"] for m_key, _ in metrics]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, means, width, label=label, color=colors[i])
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Valor da métrica")
    ax.set_title("Métricas de Qualidade da Recomendação em K=10\n(5-fold CV, relevante = nota real ≥ 4)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/ranking_bars_k10.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Salvo: ranking_bars_k10.pdf")

if __name__ == "__main__":
    fig_sensitivity_k()
    fig_sensitivity_svd()
    fig_learning_curve()
    fig_final_comparison()
    fig_per_fold_rmse()
    fig_grid_search_heatmap()
    fig_ranking_comparison()
    fig_ranking_bars_k10()
    print("\nTodas as figuras foram geradas em", FIG_DIR)
