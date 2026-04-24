import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle, os

# Key active ingredients to look for (explainability layer)
KEY_INGREDIENTS = [
    'niacinamide', 'retinol', 'hyaluronic acid', 'vitamin c', 'ascorbic acid',
    'salicylic acid', 'glycolic acid', 'lactic acid', 'ceramide', 'peptide',
    'squalane', 'zinc', 'azelaic acid', 'kojic acid', 'bakuchiol',
    'centella asiatica', 'panthenol', 'allantoin', 'tea tree', 'benzoyl peroxide'
]

SKIN_CONCERN_MAP = {
    'acne':        ['salicylic acid', 'niacinamide', 'zinc', 'benzoyl peroxide', 'tea tree'],
    'aging':       ['retinol', 'bakuchiol', 'peptide', 'vitamin c', 'ascorbic acid'],
    'hydration':   ['hyaluronic acid', 'ceramide', 'squalane', 'panthenol'],
    'brightening': ['vitamin c', 'ascorbic acid', 'niacinamide', 'kojic acid', 'azelaic acid'],
    'sensitive':   ['centella asiatica', 'allantoin', 'panthenol', 'ceramide'],
    'exfoliation': ['glycolic acid', 'lactic acid', 'salicylic acid']
}


class IngredientAwareHybrid:
    """
    Hybrid recommender that:
    1. Clusters products by key active ingredients (content signal)
    2. Blends with CF predicted ratings (collaborative signal)
    3. Provides ingredient-level explanations
    """

    def __init__(self, n_clusters=30, cf_weight=0.6, content_weight=0.4):
        self.n_clusters     = n_clusters
        self.cf_weight      = cf_weight
        self.content_weight = content_weight
        self.tfidf          = TfidfVectorizer(vocabulary=KEY_INGREDIENTS)
        self.kmeans         = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.product_df     = None
        self.ingredient_matrix = None
        self.cluster_labels    = None

    # ------------------------------------------------------------------ #
    def _extract_active_ingredients(self, ingredient_text: str) -> list[str]:
        text_lower = ingredient_text.lower()
        return [ing for ing in KEY_INGREDIENTS if ing in text_lower]

    def fit(self, products: pd.DataFrame):
        self.product_df = products.reset_index(drop=True)
        ingredient_text = (
            self.product_df['ingredients'].fillna('') + ' ' +
            self.product_df['highlights'].fillna('')
        ).str.lower()

        self.ingredient_matrix = self.tfidf.fit_transform(ingredient_text)
        self.ingredient_matrix_dense = normalize(self.ingredient_matrix.toarray())

        # Cluster products by active ingredient profile
        self.cluster_labels = self.kmeans.fit_predict(self.ingredient_matrix_dense)
        self.product_df = self.product_df.copy()
        self.product_df['ingredient_cluster'] = self.cluster_labels

        print(f"[Hybrid] Fitted. {self.n_clusters} ingredient clusters created.")

    # ------------------------------------------------------------------ #
    def _content_score(self, skin_concern: str, product_ids: list[str]) -> dict:
        """Score products by match with skin concern ingredient keywords."""
        concern_ingredients = SKIN_CONCERN_MAP.get(skin_concern.lower(), [])
        if not concern_ingredients:
            return {pid: 0.5 for pid in product_ids}

        scores = {}
        for pid in product_ids:
            row = self.product_df[self.product_df['product_id'] == pid]
            if row.empty:
                scores[pid] = 0.0
                continue
            text = (row['ingredients'].values[0] + ' ' +
                    row['highlights'].values[0]).lower()
            hits = sum(1 for ing in concern_ingredients if ing in text)
            scores[pid] = hits / len(concern_ingredients)
        return scores

    def recommend(
        self,
        user_id:      str,
        skin_type:    str,
        skin_concern: str,
        cf_recommender,           # a fitted CF model with .recommend()
        reviews:      pd.DataFrame,
        n:            int = 10
    ) -> pd.DataFrame:
        """
        Hybrid recommendation pipeline:
        Step 1 – Filter products suitable for user's skin type
        Step 2 – Get CF-based candidate list
        Step 3 – Score candidates by content (ingredient) match
        Step 4 – Blend scores and return top-n with explanation
        """
        # Step 1: Skin type filter
        skin_col = f'skin_type_{skin_type.lower()}'
        if skin_col in self.product_df.columns:
            eligible = self.product_df[self.product_df[skin_col] == 1].copy()
        else:
            eligible = self.product_df.copy()

        reviewed_ids = set(
            reviews[reviews['author_id'] == user_id]['product_id']
        )

        # Step 2: CF recommendations from eligible products
        cf_recs = cf_recommender.recommend(
            user_id, eligible, reviewed_ids, n=n * 3)

        if cf_recs.empty:
            # Cold-start fallback: top-rated products for this skin type
            cf_recs = eligible.nlargest(n * 3, 'rating')[
                ['product_id','rating']].rename(
                    columns={'rating': 'predicted_rating'})

        # Step 3: Content score
        candidate_ids = cf_recs['product_id'].tolist()
        content_scores = self._content_score(skin_concern, candidate_ids)

        # Step 4: Blend
        cf_recs = cf_recs.copy()
        # Normalize CF predicted ratings to [0,1]
        cf_col = 'predicted_rating' if 'predicted_rating' in cf_recs.columns else 'score'
        cf_recs['cf_norm']      = (cf_recs[cf_col] - 1) / 4  # ratings 1-5 → 0-1
        cf_recs['content_norm'] = cf_recs['product_id'].map(content_scores).fillna(0)
        cf_recs['hybrid_score'] = (
            self.cf_weight      * cf_recs['cf_norm'] +
            self.content_weight * cf_recs['content_norm']
        )
        cf_recs = cf_recs.nlargest(n, 'hybrid_score')

        # Add product info and explanation
        final = cf_recs.merge(
            self.product_df[['product_id','product_name','brand_name',
                             'price_usd','rating','ingredients']],
            on='product_id', how='left')

        final['key_ingredients_found'] = final['ingredients'].fillna('').apply(
            lambda x: ', '.join(self._extract_active_ingredients(x)) or 'N/A'
        )
        final['explanation'] = final.apply(
            lambda r: (f"Recommended for {skin_type} skin with {skin_concern} concern. "
                       f"Key actives: {r['key_ingredients_found']}."), axis=1
        )

        return final[['product_id','product_name','brand_name','price_usd',
                       'rating','hybrid_score','key_ingredients_found',
                       'explanation']].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    def get_cluster_summary(self) -> pd.DataFrame:
        """Summarise what ingredients dominate each cluster."""
        summary = []
        for c in range(self.n_clusters):
            mask = self.product_df['ingredient_cluster'] == c
            prods = self.product_df[mask]
            combined = ' '.join(prods['ingredients'].fillna('').str.lower())
            top_ings = [ing for ing in KEY_INGREDIENTS if ing in combined][:5]
            summary.append({
                'cluster': c,
                'n_products': mask.sum(),
                'top_ingredients': ', '.join(top_ings)
            })
        return pd.DataFrame(summary).sort_values('n_products', ascending=False)

    def save(self, path='models/hybrid.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path,'wb') as f: pickle.dump(self, f)

    @classmethod
    def load(cls, path='models/hybrid.pkl'):
        with open(path,'rb') as f: return pickle.load(f)