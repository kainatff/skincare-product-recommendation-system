import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pickle
import os


class ContentBasedRecommender:
    """
    Content-Based Recommender using:

    - TF-IDF on ingredients + highlights + categories + brand
    - Price normalization
    - Product metadata signals
    """

    def __init__(self):

        self.tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words="english"
        )

        self.product_df = None
        self.tfidf_matrix = None
        self.feature_matrix = None
        self.product_index = None

    # -------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------
    def fit(self, products: pd.DataFrame):

        print("[ContentBased] Training model...")

        self.product_df = products.reset_index(drop=True)

        # Map product_id → row index
        self.product_index = {
            pid: i
            for i, pid in enumerate(self.product_df["product_id"])
        }

        # ---------------------------------------------------------
        # TEXT FEATURES
        # ---------------------------------------------------------
        combined_text = (
            self.product_df["ingredients"].fillna("") + " " +
            self.product_df["highlights"].fillna("") + " " +
            self.product_df["primary_category"].fillna("") + " " +
            self.product_df["secondary_category"].fillna("") + " " +
            self.product_df["tertiary_category"].fillna("") + " " +
            self.product_df["brand_name"].fillna("")
        )

        self.tfidf_matrix = self.tfidf.fit_transform(combined_text)

        # ---------------------------------------------------------
        # NUMERIC FEATURES
        # ---------------------------------------------------------
        price_scaled = MinMaxScaler().fit_transform(
            self.product_df[["price_usd"]].fillna(0)
        )

        rating_scaled = MinMaxScaler().fit_transform(
            self.product_df[["rating"]].fillna(0)
        )

        loves_scaled = MinMaxScaler().fit_transform(
            self.product_df[["loves_count"]].fillna(0)
        )

        # ---------------------------------------------------------
        # BOOLEAN FEATURES
        # ---------------------------------------------------------
        bool_cols = [
            "limited_edition",
            "new",
            "online_only",
            "out_of_stock",
            "sephora_exclusive"
        ]

        for col in bool_cols:
            if col not in self.product_df.columns:
                self.product_df[col] = 0

        bool_features = self.product_df[bool_cols].fillna(0).astype(int).values

        # ---------------------------------------------------------
        # CATEGORY ONE-HOT
        # ---------------------------------------------------------
        top_cats = (
            self.product_df["primary_category"]
            .value_counts()
            .head(20)
            .index
        )

        cat_oh = pd.get_dummies(
            self.product_df["primary_category"].where(
                self.product_df["primary_category"].isin(top_cats),
                "Other"
            )
        )

        # ---------------------------------------------------------
        # COMBINE FEATURES
        # ---------------------------------------------------------
        import scipy.sparse as sp
        from sklearn.preprocessing import normalize

        structured = np.hstack([
            price_scaled,
            rating_scaled,
            loves_scaled,
            bool_features,
            cat_oh.values
        ])

        structured_sparse = sp.csr_matrix(structured)

        combined = sp.hstack([
            self.tfidf_matrix * 0.7,
            structured_sparse * 0.3
        ])

        self.feature_matrix = normalize(combined)

        print(f"[ContentBased] Fitted on {len(self.product_df)} products.")

    # -------------------------------------------------------------
    # RECOMMEND SIMILAR PRODUCTS
    # -------------------------------------------------------------
    def recommend_for_product(self, product_id: str, n: int = 10) -> pd.DataFrame:

        if product_id not in self.product_index:
            raise ValueError(f"Product {product_id} not found.")

        idx = self.product_index[product_id]

        sims = cosine_similarity(
            self.feature_matrix[idx],
            self.feature_matrix
        ).flatten()

        sims[idx] = 0

        top_idx = np.argsort(sims)[::-1][:n]

        results = self.product_df.iloc[top_idx][
            [
                "product_id",
                "product_name",
                "brand_name",
                "primary_category",
                "price_usd",
                "rating"
            ]
        ].copy()

        results["similarity_score"] = sims[top_idx]

        return results.reset_index(drop=True)

    # -------------------------------------------------------------
    # SIMPLE USER PROFILE RECOMMENDER
    # -------------------------------------------------------------
    def recommend_by_budget(self, budget_max: float, n: int = 10) -> pd.DataFrame:

        df = self.product_df.copy()

        df = df[df["price_usd"] <= budget_max]

        if df.empty:
            return pd.DataFrame()

        df = df.sort_values(
            ["rating", "loves_count"],
            ascending=False
        )

        return df.head(n)[
            [
                "product_id",
                "product_name",
                "brand_name",
                "price_usd",
                "rating"
            ]
        ].reset_index(drop=True)

    # -------------------------------------------------------------
    # RECOMMEND FOR USER PROFILE
    # -------------------------------------------------------------
    def recommend_for_user_profile(
        self,
        skin_type,
        concerns,
        budget_max,
        n=10
    ):

        df = self.product_df.copy()

        df = df[df["price_usd"] <= budget_max].copy()

        if df.empty:
            return pd.DataFrame()

        df["score"] = (
            df["rating"].fillna(0) * 0.6 +
            np.log1p(df["loves_count"].fillna(0)) * 0.4
        )

        def match_score(row):
            text = (
                str(row.get("ingredients", "")) + " " +
                str(row.get("highlights", ""))
            ).lower()

            return sum(1 for c in concerns if c.lower() in text)

        df["score"] += df.apply(match_score, axis=1)

        if "skin_type" in df.columns:
            df["score"] += df["skin_type"].fillna("").str.lower().apply(
                lambda x: 1 if skin_type.lower() in x else 0
            )

        return df.sort_values("score", ascending=False).head(n)

    # -------------------------------------------------------------
    # SAVE / LOAD
    # -------------------------------------------------------------
    def save(self, path="models/content_based.pkl"):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)

        print(f"[ContentBased] Saved to {path}")

    @classmethod
    def load(cls, path="models/content_based.pkl"):

        with open(path, "rb") as f:
            return pickle.load(f)