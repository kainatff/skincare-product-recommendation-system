"""
Main training script — run this to build and save all models.
Usage: python train.py
"""

import pandas as pd
import numpy as np
from surprise.model_selection import train_test_split as s_split

from src.data_loader import load_data, clean_products, clean_reviews
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import (
    UserBasedCF, ItemBasedCF, SVDRecommender, build_surprise_dataset
)
from src.hybrid_model import IngredientAwareHybrid
from src.evaluation import evaluate_model, plot_comparison

import warnings
warnings.filterwarnings('ignore')


# ── 1. Load & Clean ──────────────────────────────────────────
# ── 1. Load Cleaned Data ─────────────────────────────────────

print("=== Step 1: Loading cleaned data ===")

products, reviews = load_data(
    'data/cleaned/products.csv',
    'data/cleaned/reviews.csv'
)

print(f"Products: {products.shape}")
print(f"Reviews: {reviews.shape}")

# ── 2. Content-Based Model ───────────────────────────────────
print("\n=== Step 2: Content-Based Model ===")

# 👉 Create combined text features (IMPORTANT FIX)
products['combined_features'] = (
    products['ingredients'].fillna('') + ' ' +
    products['highlights'].fillna('') + ' ' +
    products['primary_category'].fillna('') + ' ' +
    products['secondary_category'].fillna('') + ' ' +
    products['tertiary_category'].fillna('') + ' ' +
    products['brand_name'].fillna('')
)

cbr = ContentBasedRecommender()
cbr.fit(products)   # make sure your model uses 'combined_features'
cbr.save()


# ── 3. Collaborative Filtering ───────────────────────────────
print("\n=== Step 3: Collaborative Filtering ===")

print("\nSampling reviews for CF training...")

reviews_sample = reviews.sample(
    frac=0.25,   # use 25% of data
    random_state=42
)

print(f"Sample size: {len(reviews_sample)}")

data = build_surprise_dataset(reviews_sample)

trainset, testset = s_split(data, test_size=0.2, random_state=42)

user_cf = UserBasedCF(k=10)
user_cf.fit(data)
user_cf.save()

item_cf = ItemBasedCF(k=10)
item_cf.fit(data)
item_cf.save()

svd = SVDRecommender(n_factors=100, n_epochs=20)
svd.fit(data)
svd.save()


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

for name, model in [
    ('UserCF', user_cf),
    ('ItemCF', item_cf),
    ('SVD', svd)
]:
    r = evaluate_model(name, model, reviews_test, products)
    results.append(r)

    print(f"\n[{name}] Results:")
    for k, v in r.items():
        if k != 'model':
            print(f"  {k}: {v:.4f}")


plot_comparison(results)

print("\n=== Training complete! All models saved to models/ ===")