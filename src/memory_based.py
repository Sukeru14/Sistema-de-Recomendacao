import numpy as np

def cosine_similarity_matrix(matrix, mask):
    #Para cosseno simples, tratamos ausentes como 0 (já são 0 na matrix)
    norms = np.sqrt((matrix ** 2).sum(axis=1))
    norms[norms == 0] = 1e-9
    normalized = matrix / norms[:, None]
    sim = normalized @ normalized.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return sim

def pearson_similarity_matrix(matrix, mask):
    n = matrix.shape[0]
    row_means = np.zeros(n, dtype=np.float64)
    for i in range(n):
        observed = matrix[i][mask[i]]
        row_means[i] = observed.mean() if observed.size > 0 else 0.0

    centered = np.zeros_like(matrix, dtype=np.float64)
    for i in range(n):
        centered[i][mask[i]] = matrix[i][mask[i]] - row_means[i]

    norms = np.sqrt((centered ** 2).sum(axis=1))
    norms[norms == 0] = 1e-9
    normalized = centered / norms[:, None]
    sim = normalized @ normalized.T

    #Penaliza pares com pouquíssimo overlap (menos de 2 itens em comum)
    co_count = (mask.astype(np.int32) @ mask.astype(np.int32).T)
    sim[co_count < 2] = 0.0

    np.clip(sim, -1.0, 1.0, out=sim)
    return sim

class MemoryBasedCF:
    def __init__(self, mode="user", similarity="cosine", k=30):
        assert mode in ("user", "item")
        assert similarity in ("cosine", "pearson")
        self.mode = mode
        self.similarity = similarity
        self.k = k
        self.matrix_ = None
        self.mask_ = None
        self.sim_ = None
        self.row_means_ = None
        self.global_mean_ = None

    def fit(self, matrix):
        if self.mode == "item":
            work_matrix = matrix.T.copy()
        else:
            work_matrix = matrix.copy()

        self.mask_ = work_matrix > 0
        self.matrix_ = work_matrix.astype(np.float64)
        self.global_mean_ = self.matrix_[self.mask_].mean()

        n = self.matrix_.shape[0]
        means = np.zeros(n)
        for i in range(n):
            obs = self.matrix_[i][self.mask_[i]]
            means[i] = obs.mean() if obs.size > 0 else self.global_mean_
        self.row_means_ = means

        if self.similarity == "cosine":
            self.sim_ = cosine_similarity_matrix(self.matrix_, self.mask_)
        else:
            self.sim_ = pearson_similarity_matrix(self.matrix_, self.mask_)

        #Zera autosimilaridade para não usar o próprio vizinho na previsão
        np.fill_diagonal(self.sim_, 0.0)

        return self

    def _predict_single(self, row_idx, col_idx):
        sims = self.sim_[row_idx]
        col_mask = self.mask_[:, col_idx] #quais linhas avaliaram esta coluna

        candidate_idx = np.where(col_mask)[0]
        if candidate_idx.size == 0:
            return self.row_means_[row_idx]

        candidate_sims = sims[candidate_idx]

        #mantém só vizinhos com similaridade positiva
        pos_mask = candidate_sims > 0
        candidate_idx = candidate_idx[pos_mask]
        candidate_sims = candidate_sims[pos_mask]

        if candidate_idx.size == 0:
            return self.row_means_[row_idx]

        if candidate_idx.size > self.k:
            top_k = np.argpartition(-candidate_sims, self.k)[: self.k]
            candidate_idx = candidate_idx[top_k]
            candidate_sims = candidate_sims[top_k]

        denom = np.abs(candidate_sims).sum()
        if denom < 1e-9:
            return self.row_means_[row_idx]

        deviations = self.matrix_[candidate_idx, col_idx] - self.row_means_[candidate_idx]
        pred = self.row_means_[row_idx] + (candidate_sims @ deviations) / denom

        return pred

    def predict(self, user_id, item_id):
        if self.mode == "user":
            row_idx, col_idx = user_id, item_id
        else:
            row_idx, col_idx = item_id, user_id

        if row_idx >= self.matrix_.shape[0]:
            return self.global_mean_

        pred = self._predict_single(row_idx, col_idx)
        return float(np.clip(pred, 1.0, 5.0))

    def predict_batch(self, user_item_pairs):
        return np.array([self.predict(u, i) for u, i in user_item_pairs])
