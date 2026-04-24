"""
Main training script — run this to build and save all models.
Usage: python train.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data_loader import load_data, clean_products, clean_reviews
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import (
    UserBasedCF, ItemBasedCF, SVDRecommender
)
from src.hybrid_model import IngredientAwareHybrid
from src.evaluation import evaluate_model, plot_comparison

import warnings
warnings.filterwarnings('ignore')


# ── 1. Load Cleaned Data ─────────────────────────────────────
print("=== Step 1: Loading cleaned data ===")

products, reviews = load_data(
    'data/cleaned/products.csv',
    'data/cleaned/reviews.csv'
)

print(f"Products: {products.shape}")
print(f"Reviews:  {reviews.shape}")


# ── 2. Content-Based Model ───────────────────────────────────
print("\n=== Step 2: Content-Based Model ===")

products['combined_features'] = (
    products['ingredients'].fillna('') + ' ' +
    products['highlights'].fillna('') + ' ' +
    products['primary_category'].fillna('') + ' ' +
    products['secondary_category'].fillna('') + ' ' +
    products['tertiary_category'].fillna('') + ' ' +
    products['brand_name'].fillna('')
)

cbr = ContentBasedRecommender()
cbr.fit(products)
cbr.save()
print("Content-based model saved.")


# ── 3. Collaborative Filtering ───────────────────────────────
print("\n=== Step 3: Collaborative Filtering ===")

print("Sampling reviews for CF training...")

# Drop leftover pandas index column saved as unnamed (common when CSV was
# written with df.to_csv() without index=False)
reviews = reviews.loc[:, ~reviews.columns.str.match(r'^Unnamed')]

# ── Auto-detect column names ──────────────────────────────────
_USER_ALIASES    = ['author_id', 'user_id', 'reviewer_id', 'reviewerid', 'userid', 'user']
_PRODUCT_ALIASES = ['product_id', 'item_id', 'asin', 'productid', 'prod_id', 'product']
_RATING_ALIASES  = ['rating', 'stars', 'score', 'review_rating', 'overall']

def _find_col(df, aliases, label):
    cols_lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in cols_lower:
            return cols_lower[a]
    raise KeyError(
        f"Could not find a '{label}' column. "
        f"Tried: {aliases}. Available: {list(df.columns)}"
    )

user_col    = _find_col(reviews, _USER_ALIASES,    'user_id')
product_col = _find_col(reviews, _PRODUCT_ALIASES, 'product_id')
rating_col  = _find_col(reviews, _RATING_ALIASES,  'rating')

print(f"Using columns → user='{user_col}'  product='{product_col}'  rating='{rating_col}'")

# Normalise to standard names so the rest of the code (and CF models) is consistent
reviews = reviews.rename(columns={
    user_col:    'user_id',
    product_col: 'product_id',
    rating_col:  'rating',
})

reviews_sample = reviews.sample(frac=0.25, random_state=42)
print(f"Sample size: {len(reviews_sample)}")

# Split into train / test using sklearn — no surprise needed
train_reviews, test_reviews = train_test_split(
    reviews_sample, test_size=0.2, random_state=42
)

# Build the user-item rating matrix from training data
# Shape: (n_users, n_products); missing entries filled with 0
train_matrix = (
    train_reviews
    .pivot_table(index='user_id', columns='product_id',
                 values='rating', aggfunc='mean')
    .fillna(0)
)

print(f"Rating matrix shape: {train_matrix.shape}")

user_cf = UserBasedCF(k=10)
user_cf.fit(train_matrix)
user_cf.save()
print("UserCF saved.")

item_cf = ItemBasedCF(k=10)
item_cf.fit(train_matrix)
item_cf.save()
print("ItemCF saved.")

svd = SVDRecommender(n_factors=100)
svd.fit(train_matrix)
svd.save()
print("SVD saved.")


# ── 4. Hybrid Model ──────────────────────────────────────────
print("\n=== Step 4: Hybrid Model ===")

hybrid = IngredientAwareHybrid(
    n_clusters=30,
    cf_weight=0.6,
    content_weight=0.4
)
hybrid.fit(products)
hybrid.save()

print("\nCluster summary:")
print(hybrid.get_cluster_summary().head(10).to_string(index=False))


# ── 5. Evaluation ────────────────────────────────────────────
print("\n=== Step 5: Evaluation ===")

reviews_test = reviews.sample(frac=0.2, random_state=42)

results = []
for name, model in [('UserCF', user_cf), ('ItemCF', item_cf), ('SVD', svd)]:
    r = evaluate_model(name, model, reviews_test, products)
    results.append(r)
    print(f"\n[{name}] Results:")
    for k, v in r.items():
        if k != 'model':
            print(f"  {k}: {v:.4f}")

plot_comparison(results)
print("\n=== Training complete! All models saved to models/ ===")