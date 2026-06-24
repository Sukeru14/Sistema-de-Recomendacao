import numpy as np

RELEVANCE_THRESHOLD = 4.0

def _dcg(relevances):
    relevances = np.asarray(relevances, dtype=np.float64)
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(relevances / discounts))

def ndcg_at_k(recommended_items, relevant_items, k):
    top_k = recommended_items[:k]
    gains = [1.0 if item in relevant_items else 0.0 for item in top_k]
    dcg = _dcg(gains)

    ideal_gains = [1.0] * min(len(relevant_items), k)
    idcg = _dcg(ideal_gains)

    return dcg / idcg if idcg > 0 else 0.0

def precision_recall_at_k(recommended_items, relevant_items, k):
    top_k = recommended_items[:k]
    n_relevant_in_topk = sum(1 for item in top_k if item in relevant_items)

    precision = n_relevant_in_topk / k if k > 0 else 0.0
    recall = n_relevant_in_topk / len(relevant_items) if len(relevant_items) > 0 else 0.0

    return precision, recall

def average_precision_at_k(recommended_items, relevant_items, k):
    top_k = recommended_items[:k]
    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(top_k, start=1):
        if item in relevant_items:
            hits += 1
            sum_precisions += hits / i
    if len(relevant_items) == 0:
        return 0.0
    return sum_precisions / min(len(relevant_items), k)

def evaluate_ranking(model, train_df, test_df, k_values=(5, 10, 20), relevance_threshold=RELEVANCE_THRESHOLD, n_users=943, n_items=1682, predict_fn=None):
    train_seen = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    test_by_user = test_df.groupby("user_id")

    metrics = {k: {"precision": [], "recall": [], "ndcg": [], "map": []} for k in k_values}

    for user_id, group in test_by_user:
        user_idx = user_id - 1
        candidate_items = group["item_id"].to_numpy()
        candidate_ratings = group["rating"].to_numpy()

        relevant_items = set(candidate_items[candidate_ratings >= relevance_threshold])
        if len(relevant_items) == 0:
            continue  #usuário sem nenhum item relevante no teste: não contribui

        #Prevê nota para cada candidato e ordena (ranking do modelo)
        if predict_fn is not None:
            scores = [predict_fn(model, user_idx, item_id - 1) for item_id in candidate_items]
        else:
            scores = [model.predict(user_idx, item_id - 1) for item_id in candidate_items]

        order = np.argsort(-np.asarray(scores))
        ranked_items = candidate_items[order]

        max_k = max(k_values)
        if len(ranked_items) < 1:
            continue

        for k in k_values:
            precision, recall = precision_recall_at_k(ranked_items, relevant_items, k)
            ndcg = ndcg_at_k(ranked_items, relevant_items, k)
            ap = average_precision_at_k(ranked_items, relevant_items, k)

            metrics[k]["precision"].append(precision)
            metrics[k]["recall"].append(recall)
            metrics[k]["ndcg"].append(ndcg)
            metrics[k]["map"].append(ap)

    results = {}
    for k in k_values:
        results[k] = {
            "precision_at_k": float(np.mean(metrics[k]["precision"])),
            "recall_at_k": float(np.mean(metrics[k]["recall"])),
            "ndcg_at_k": float(np.mean(metrics[k]["ndcg"])),
            "map_at_k": float(np.mean(metrics[k]["map"])),
            "n_users_evaluated": len(metrics[k]["precision"]),
        }
    return results
