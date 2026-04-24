"""
Streamlit demo app for the Skincare Recommendation System.
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import SVDRecommender
from src.hybrid_model import IngredientAwareHybrid, SKIN_CONCERN_MAP

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skincare Recommender",
    page_icon="🌿",
    layout="wide"
)
st.title("🌿 Skincare Product Recommendation System")
st.markdown("*Personalized recommendations powered by Content-Based & Collaborative Filtering*")

# ── Load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    cbr    = ContentBasedRecommender.load()
    svd    = SVDRecommender.load()
    hybrid = IngredientAwareHybrid.load()
    return cbr, svd, hybrid

@st.cache_data
def load_data_cached():
    products = pd.read_csv('data/cleaned/products.csv')
    reviews  = pd.read_csv('data/cleaned/reviews.csv')
    return products, reviews

cbr, svd, hybrid = load_models()
products, reviews = load_data_cached()

# ── Sidebar: User inputs ──────────────────────────────────────────────────────
st.sidebar.header("Your Profile")

mode = st.sidebar.radio("Recommendation mode",
    ["New User (no history)", "Existing User (by ID)"])

skin_type = st.sidebar.selectbox("Skin type",
    ["Normal", "Dry", "Oily", "Combination", "Sensitive"])

skin_concern = st.sidebar.selectbox("Primary concern",
    list(SKIN_CONCERN_MAP.keys()))

budget  = st.sidebar.slider("Max budget (USD)", 5, 200, 80)
n_recs  = st.sidebar.slider("Number of recommendations", 5, 20, 10)

user_id = None
if mode == "Existing User (by ID)":
    user_id = st.sidebar.text_input("Enter your User ID")

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🔍 Get Recommendations", type="primary")

# ── Main content ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Recommendations", "Product Lookup", "Model Comparison"])

with tab1:
    if run_btn:
        with st.spinner("Finding best products for you..."):
            if mode == "New User (no history)" or not user_id:
                # Content-based only — no CF needed
                recs = cbr.recommend_for_user_profile(
                    skin_type=skin_type,
                    concerns=[skin_concern] + list(SKIN_CONCERN_MAP.get(skin_concern, [])),
                    budget_max=budget,
                    n=n_recs
                )
                method = "Content-Based Filtering"
            else:
                # Hybrid — passes the sklearn-based SVDRecommender
                recs = hybrid.recommend(
                    user_id=user_id,
                    skin_type=skin_type,
                    skin_concern=skin_concern,
                    cf_recommender=svd,        # SVDRecommender.recommend(user_id, n)
                    reviews=reviews,
                    n=n_recs
                )
                recs = recs[recs['price_usd'] <= budget]
                method = "Ingredient-Aware Hybrid Model"

        if recs.empty:
            st.warning("No products found. Try relaxing filters.")
        else:
            st.success(f"Found {len(recs)} recommendations using **{method}**")
            for _, row in recs.iterrows():
                with st.expander(
                    f"**{row.get('product_name', 'N/A')}** — {row.get('brand_name', '')}"
                ):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Price",  f"${row.get('price_usd', 0):.2f}")
                    col2.metric("Rating", f"{row.get('rating', 0):.1f} ⭐")
                    col3.metric("Score",
                        f"{row.get('hybrid_score', row.get('relevance_score', 0)):.3f}")
                    if 'key_ingredients_found' in row:
                        st.markdown(f"**Active ingredients:** {row['key_ingredients_found']}")
                    if 'explanation' in row:
                        st.info(row['explanation'])

with tab2:
    st.subheader("Find similar products")
    product_name_input = st.text_input("Search product name")
    if product_name_input:
        matches = products[
            products['product_name'].str.contains(product_name_input, case=False, na=False)
        ]
        if not matches.empty:
            selected_pid = st.selectbox(
                "Select product",
                matches['product_id'].values,
                format_func=lambda pid: matches[
                    matches['product_id'] == pid]['product_name'].values[0]
            )
            if st.button("Find Similar"):
                sims = cbr.recommend_for_product(selected_pid, n=10)
                st.dataframe(sims, use_container_width=True)
        else:
            st.warning("No matching products found.")

with tab3:
    st.subheader("Model Performance Comparison")
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        img = mpimg.imread('reports/model_comparison.png')
        st.image(img, use_container_width=True)   # use_container_width replaces deprecated use_column_width
    except FileNotFoundError:
        st.info("Run `python train.py` first to generate evaluation charts.")