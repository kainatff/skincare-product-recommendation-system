"""
Collaborative filtering models using scikit-learn only (no surprise).

Three models are provided:
  - UserBasedCF  : user-user k-NN on the rating matrix
  - ItemBasedCF  : item-item k-NN on the transposed rating matrix
  - SVDRecommender: matrix factorisation via TruncatedSVD

All three share the same interface:
  .fit(rating_matrix)   — rating_matrix is a pd.DataFrame (users × products)
  .predict(user_id, product_id) → float
  .recommend(user_id, n) → list of (product_id, score)
  .save() / .load()
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ─── helpers ────────────────────────────────────────────────────────────────

def _save(obj, path: str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def _load(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ─── User-Based CF ──────────────────────────────────────────────────────────

class UserBasedCF:
    """
    Predicts a rating for (user, item) by finding the k most similar users
    (cosine similarity on their rating vectors) and averaging their ratings
    for that item.
    """

    def __init__(self, k: int = 10):
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=k + 1,   # +1 because user matches itself
                                   metric='cosine',
                                   algorithm='brute')
        self.matrix: pd.DataFrame | None = None          # users × products
        self._path = os.path.join(MODEL_DIR, 'user_cf.pkl')

    # ── fit ────────────────────────────────────────────────────────────────
    def fit(self, rating_matrix: pd.DataFrame):
        """
        Parameters
        ----------
        rating_matrix : pd.DataFrame
            Index = user_id, columns = product_id, values = mean rating (0 for missing).
        """
        self.matrix = rating_matrix
        self.nn.fit(rating_matrix.values)
        return self

    # ── predict ────────────────────────────────────────────────────────────
    def predict(self, user_id, product_id) -> float:
        if self.matrix is None:
            raise RuntimeError("Model not fitted yet.")
        if user_id not in self.matrix.index:
            return self.matrix[product_id].mean() if product_id in self.matrix.columns else 0.0
        if product_id not in self.matrix.columns:
            return 0.0

        user_vec = self.matrix.loc[user_id].values.reshape(1, -1)
        distances, indices = self.nn.kneighbors(user_vec)

        # indices[0][0] is the user itself — skip it
        neighbour_idx = indices[0][1:]
        neighbour_ratings = self.matrix.iloc[neighbour_idx][product_id].values

        # Only count neighbours who actually rated the product
        rated_mask = neighbour_ratings > 0
        if not rated_mask.any():
            return self.matrix[product_id].mean()

        weights = 1.0 - distances[0][1:][rated_mask]   # similarity = 1 − cosine_distance
        weights = np.maximum(weights, 0)
        total_w = weights.sum()
        if total_w == 0:
            return neighbour_ratings[rated_mask].mean()
        return float(np.dot(weights, neighbour_ratings[rated_mask]) / total_w)

    # ── recommend ──────────────────────────────────────────────────────────
    def recommend(self, user_id, n: int = 10) -> list[tuple]:
        """Return list of (product_id, predicted_score), highest score first."""
        if self.matrix is None:
            raise RuntimeError("Model not fitted yet.")

        already_rated = set()
        if user_id in self.matrix.index:
            row = self.matrix.loc[user_id]
            already_rated = set(row[row > 0].index)

        scores = [
            (pid, self.predict(user_id, pid))
            for pid in self.matrix.columns
            if pid not in already_rated
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

    # ── persistence ────────────────────────────────────────────────────────
    def save(self):
        _save(self, self._path)

    @classmethod
    def load(cls) -> 'UserBasedCF':
        path = os.path.join(MODEL_DIR, 'user_cf.pkl')
        return _load(path)


# ─── Item-Based CF ──────────────────────────────────────────────────────────

class ItemBasedCF:
    """
    Predicts a rating for (user, item) by finding the k most similar items
    (cosine similarity on their rating vectors across users) and averaging.
    """

    def __init__(self, k: int = 10):
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=k + 1,
                                   metric='cosine',
                                   algorithm='brute')
        self.matrix: pd.DataFrame | None = None          # users × products
        self._path = os.path.join(MODEL_DIR, 'item_cf.pkl')

    def fit(self, rating_matrix: pd.DataFrame):
        self.matrix = rating_matrix
        # fit on products × users (transposed)
        self.nn.fit(rating_matrix.T.values)
        return self

    def predict(self, user_id, product_id) -> float:
        if self.matrix is None:
            raise RuntimeError("Model not fitted yet.")
        if product_id not in self.matrix.columns:
            return 0.0
        if user_id not in self.matrix.index:
            return self.matrix[product_id].mean()

        pid_idx = self.matrix.columns.get_loc(product_id)
        item_vec = self.matrix.T.iloc[pid_idx].values.reshape(1, -1)
        distances, indices = self.nn.kneighbors(item_vec)

        neighbour_pids = self.matrix.columns[indices[0][1:]]
        user_ratings = self.matrix.loc[user_id, neighbour_pids].values

        rated_mask = user_ratings > 0
        if not rated_mask.any():
            return self.matrix[product_id].mean()

        weights = 1.0 - distances[0][1:][rated_mask]
        weights = np.maximum(weights, 0)
        total_w = weights.sum()
        if total_w == 0:
            return user_ratings[rated_mask].mean()
        return float(np.dot(weights, user_ratings[rated_mask]) / total_w)

    def recommend(self, user_id, n: int = 10) -> list[tuple]:
        if self.matrix is None:
            raise RuntimeError("Model not fitted yet.")

        already_rated = set()
        if user_id in self.matrix.index:
            row = self.matrix.loc[user_id]
            already_rated = set(row[row > 0].index)

        scores = [
            (pid, self.predict(user_id, pid))
            for pid in self.matrix.columns
            if pid not in already_rated
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

    def save(self):
        _save(self, self._path)

    @classmethod
    def load(cls) -> 'ItemBasedCF':
        path = os.path.join(MODEL_DIR, 'item_cf.pkl')
        return _load(path)


# ─── SVD Recommender ────────────────────────────────────────────────────────

class SVDRecommender:
    """
    Matrix factorisation via sklearn TruncatedSVD.

    The rating matrix R ≈ U · Σ · Vt.
    Predicted ratings are reconstructed as R̂ = U · Σ · Vt.
    """

    def __init__(self, n_factors: int = 100):
        self.n_factors = n_factors
        self.svd = TruncatedSVD(n_components=n_factors, random_state=42)
        self.matrix: pd.DataFrame | None = None
        self.reconstructed: np.ndarray | None = None
        self._path = os.path.join(MODEL_DIR, 'svd.pkl')

    def fit(self, rating_matrix: pd.DataFrame):
        """
        Parameters
        ----------
        rating_matrix : pd.DataFrame
            Index = user_id, columns = product_id, values = mean rating (0 for missing).
        """
        self.matrix = rating_matrix
        R = rating_matrix.values.astype(float)

        # Clamp n_components to be < min(shape)
        max_factors = min(R.shape) - 1
        n = min(self.n_factors, max_factors)
        if n != self.n_factors:
            print(f"  [SVD] Reducing n_factors from {self.n_factors} to {n} "
                  f"(matrix is {R.shape})")
            self.svd = TruncatedSVD(n_components=n, random_state=42)

        U = self.svd.fit_transform(R)           # (n_users, n_factors)
        Vt = self.svd.components_               # (n_factors, n_products)
        Sigma = np.diag(self.svd.singular_values_)

        self.reconstructed = U @ Sigma @ Vt     # (n_users, n_products) — full R̂
        return self

    def predict(self, user_id, product_id) -> float:
        if self.matrix is None or self.reconstructed is None:
            raise RuntimeError("Model not fitted yet.")
        if user_id not in self.matrix.index or product_id not in self.matrix.columns:
            return 0.0
        u_idx = self.matrix.index.get_loc(user_id)
        p_idx = self.matrix.columns.get_loc(product_id)
        return float(self.reconstructed[u_idx, p_idx])

    def recommend(self, user_id, n: int = 10) -> list[tuple]:
        if self.matrix is None or self.reconstructed is None:
            raise RuntimeError("Model not fitted yet.")

        if user_id not in self.matrix.index:
            # Cold-start: return globally highest-rated products
            mean_scores = self.matrix.mean(axis=0)
            top = mean_scores.nlargest(n)
            return list(zip(top.index, top.values))

        u_idx = self.matrix.index.get_loc(user_id)
        user_row = self.matrix.loc[user_id]
        already_rated = set(user_row[user_row > 0].index)

        scores = pd.Series(
            self.reconstructed[u_idx],
            index=self.matrix.columns
        )
        scores = scores.drop(labels=list(already_rated), errors='ignore')
        top = scores.nlargest(n)
        return list(zip(top.index, top.values))

    def save(self):
        _save(self, self._path)

    @classmethod
    def load(cls) -> 'SVDRecommender':
        path = os.path.join(MODEL_DIR, 'svd.pkl')
        return _load(path)