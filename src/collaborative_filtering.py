import pandas as pd
import numpy as np
from surprise import (
    SVD,
    KNNWithMeans,
    Dataset,
    Reader,
    accuracy
)
from surprise.model_selection import GridSearchCV
import pickle
import os


# ─────────────────────────────────────────────────────────────
# BUILD SURPRISE DATASET
# ─────────────────────────────────────────────────────────────

def build_surprise_dataset(reviews: pd.DataFrame):
    """
    Build Surprise dataset from reviews dataframe.

    Required columns:
        author_id
        product_id
        rating
    """

    print("[CF] Preparing dataset...")

    df = reviews.copy()

    required_cols = [
        "author_id",
        "product_id",
        "rating"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(
        subset=[
            "author_id",
            "product_id",
            "rating"
        ]
    )

    df["author_id"] = df["author_id"].astype(str)
    df["product_id"] = df["product_id"].astype(str)
    df["rating"] = df["rating"].astype(float)

    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(
        df[
            [
                "author_id",
                "product_id",
                "rating"
            ]
        ],
        reader
    )

    print(f"[CF] Dataset ready: {len(df)} ratings")

    return data


# ─────────────────────────────────────────────────────────────
# USER-BASED COLLABORATIVE FILTERING
# ─────────────────────────────────────────────────────────────

class UserBasedCF:

    def __init__(self, k=20):

        self.k = k

        self.model = KNNWithMeans(
            k=k,
            min_k=3,
            sim_options={
                "name": "cosine",
                "user_based": True
            },
            verbose=False
        )

        self.trainset = None

    # ---------------------------------------------------------

    def fit(self, data):

        self.trainset = data.build_full_trainset()

        self.model.fit(self.trainset)

        print(f"[UserBasedCF] Fitted with k={self.k}")

    # ---------------------------------------------------------

    def predict(self, user_id, product_id):

        try:
            pred = self.model.predict(
                str(user_id),
                str(product_id)
            )
            return pred.est

        except Exception:
            return 0.0

    # ---------------------------------------------------------

    def recommend(
        self,
        user_id: str,
        products: pd.DataFrame,
        reviewed_ids: set,
        n: int = 10
    ) -> pd.DataFrame:

        if self.trainset is None:
            return pd.DataFrame()

        candidates = [
            pid
            for pid in products["product_id"]
            if pid not in reviewed_ids
        ]

        preds = []

        for pid in candidates:

            rating = self.predict(
                user_id,
                pid
            )

            preds.append(
                (
                    pid,
                    rating
                )
            )

        preds.sort(
            key=lambda x: x[1],
            reverse=True
        )

        top = preds[:n]

        top_df = pd.DataFrame(
            top,
            columns=[
                "product_id",
                "predicted_rating"
            ]
        )

        return top_df.merge(
            products[
                [
                    "product_id",
                    "product_name",
                    "brand_name",
                    "price_usd",
                    "rating"
                ]
            ],
            on="product_id"
        ).reset_index(drop=True)

    # ---------------------------------------------------------

    def evaluate(self, testset):

        predictions = self.model.test(testset)

        rmse = accuracy.rmse(
            predictions,
            verbose=False
        )

        mae = accuracy.mae(
            predictions,
            verbose=False
        )

        print(
            f"[UserBasedCF] RMSE={rmse:.4f}  MAE={mae:.4f}"
        )

        return rmse, mae

    # ---------------------------------------------------------

    def save(
        self,
        path="models/user_cf.pkl"
    ):

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )

        with open(
            path,
            "wb"
        ) as f:

            pickle.dump(
                self,
                f
            )

    # ---------------------------------------------------------

    @classmethod
    def load(
        cls,
        path="models/user_cf.pkl"
    ):

        with open(
            path,
            "rb"
        ) as f:

            return pickle.load(f)


# ─────────────────────────────────────────────────────────────
# ITEM-BASED COLLABORATIVE FILTERING
# ─────────────────────────────────────────────────────────────

class ItemBasedCF:

    def __init__(self, k=20):

        self.k = k

        self.model = KNNWithMeans(
            k=k,
            sim_options={
                "name": "pearson_baseline",
                "user_based": False
            },
            verbose=False
        )

        self.trainset = None

    # ---------------------------------------------------------

    def fit(self, data):

        self.trainset = data.build_full_trainset()

        self.model.fit(self.trainset)

        print(f"[ItemBasedCF] Fitted with k={self.k}")

    # ---------------------------------------------------------

    def get_similar_items(
        self,
        product_id,
        n=10
    ):

        if self.trainset is None:
            return []

        try:

            inner_id = self.trainset.to_inner_iid(
                str(product_id)
            )

        except ValueError:
            return []

        neighbors = self.model.get_neighbors(
            inner_id,
            k=n
        )

        results = []

        for iid in neighbors:

            raw_id = self.trainset.to_raw_iid(iid)

            score = self.model.sim[
                inner_id
            ][
                iid
            ]

            results.append(
                (
                    raw_id,
                    score
                )
            )

        return sorted(
            results,
            key=lambda x: x[1],
            reverse=True
        )

    # ---------------------------------------------------------

    def recommend(
        self,
        user_id,
        reviews,
        products,
        n=10
    ):

        user_reviews = reviews[
            reviews["author_id"] == user_id
        ]

        if user_reviews.empty:
            return pd.DataFrame()

        top_products = (
            user_reviews
            .nlargest(
                5,
                "rating"
            )["product_id"]
            .tolist()
        )

        reviewed_ids = set(
            user_reviews["product_id"]
        )

        scored = {}

        for pid in top_products:

            neighbors = self.get_similar_items(
                pid,
                n=20
            )

            for sim_pid, score in neighbors:

                if sim_pid not in reviewed_ids:

                    scored[
                        sim_pid
                    ] = max(
                        scored.get(
                            sim_pid,
                            0
                        ),
                        score
                    )

        if not scored:
            return pd.DataFrame()

        rec_df = pd.DataFrame(
            scored.items(),
            columns=[
                "product_id",
                "score"
            ]
        )

        rec_df = rec_df.nlargest(
            n,
            "score"
        )

        return rec_df.merge(
            products[
                [
                    "product_id",
                    "product_name",
                    "brand_name",
                    "price_usd",
                    "rating"
                ]
            ],
            on="product_id"
        ).reset_index(drop=True)

    # ---------------------------------------------------------

    def evaluate(self, testset):

        preds = self.model.test(testset)

        rmse = accuracy.rmse(
            preds,
            verbose=False
        )

        mae = accuracy.mae(
            preds,
            verbose=False
        )

        print(
            f"[ItemBasedCF] RMSE={rmse:.4f}  MAE={mae:.4f}"
        )

        return rmse, mae

    # ---------------------------------------------------------

    def save(
        self,
        path="models/item_cf.pkl"
    ):

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )

        with open(
            path,
            "wb"
        ) as f:

            pickle.dump(
                self,
                f
            )

    # ---------------------------------------------------------

    @classmethod
    def load(
        cls,
        path="models/item_cf.pkl"
    ):

        with open(
            path,
            "rb"
        ) as f:

            return pickle.load(f)


# ─────────────────────────────────────────────────────────────
# SVD MATRIX FACTORIZATION
# ─────────────────────────────────────────────────────────────

class SVDRecommender:

    def __init__(
        self,
        n_factors=100,
        n_epochs=20
    ):

        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            verbose=False
        )

        self.trainset = None

    # ---------------------------------------------------------

    def fit(self, data):

        self.trainset = data.build_full_trainset()

        self.model.fit(self.trainset)

        print("[SVD] Fitted")

    # ---------------------------------------------------------

    def predict(
        self,
        user_id,
        product_id
    ):

        try:

            return self.model.predict(
                str(user_id),
                str(product_id)
            ).est

        except Exception:

            return 0.0

    # ---------------------------------------------------------

    def save(
        self,
        path="models/svd.pkl"
    ):

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )

        with open(
            path,
            "wb"
        ) as f:

            pickle.dump(
                self,
                f
            )

    # ---------------------------------------------------------

    @classmethod
    def load(
        cls,
        path="models/svd.pkl"
    ):

        with open(
            path,
            "rb"
        ) as f:

            return pickle.load(f)