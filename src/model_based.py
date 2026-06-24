import numpy as np
from numba import njit

@njit(cache=True)
def _sgd_epoch(triples, b_u, b_i, P, Q, global_mean, lr, reg):
    n = triples.shape[0]
    sq_err_sum = 0.0
    for idx in range(n):
        u = int(triples[idx, 0])
        i = int(triples[idx, 1])
        r = triples[idx, 2]

        pred = global_mean + b_u[u] + b_i[i] + np.dot(P[u], Q[i])
        err = r - pred

        b_u[u] += lr * (err - reg * b_u[u])
        b_i[i] += lr * (err - reg * b_i[i])

        p_u_old = P[u].copy()
        P[u] += lr * (err * Q[i] - reg * P[u])
        Q[i] += lr * (err * p_u_old - reg * Q[i])

        sq_err_sum += err * err

    return sq_err_sum / n

class RegularizedSVD:
    def __init__(self, n_factors=20, lr=0.005, reg=0.02, n_epochs=20, random_state=42, verbose=False):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.verbose = verbose

        self.global_mean_ = None
        self.b_u_ = None
        self.b_i_ = None
        self.P_ = None
        self.Q_ = None
        self.train_rmse_history_ = []

    def fit(self, train_triples, n_users, n_items):
        rng = np.random.RandomState(self.random_state)

        self.global_mean_ = train_triples[:, 2].mean()
        self.b_u_ = np.zeros(n_users, dtype=np.float64)
        self.b_i_ = np.zeros(n_items, dtype=np.float64)
        #Inicialização pequena e aleatória
        self.P_ = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q_ = rng.normal(0, 0.1, (n_items, self.n_factors))

        triples = train_triples.copy()
        self.train_rmse_history_ = []

        for epoch in range(self.n_epochs):
            rng.shuffle(triples) #ordem aleatória a cada época
            mse = _sgd_epoch(triples, self.b_u_, self.b_i_, self.P_, self.Q_, self.global_mean_, self.lr, self.reg)
            epoch_rmse = np.sqrt(mse)
            self.train_rmse_history_.append(epoch_rmse)
            if self.verbose:
                print(f"  [SVD] época {epoch + 1}/{self.n_epochs} - RMSE treino: {epoch_rmse:.4f}")

        return self

    def predict(self, user_id, item_id):
        if user_id >= len(self.b_u_) or item_id >= len(self.b_i_):
            return float(np.clip(self.global_mean_, 1.0, 5.0))

        pred = (self.global_mean_ + self.b_u_[user_id] + self.b_i_[item_id] + self.P_[user_id] @ self.Q_[item_id])
        return float(np.clip(pred, 1.0, 5.0))

    def predict_batch(self, user_item_pairs):
        return np.array([self.predict(u, i) for u, i in user_item_pairs])
