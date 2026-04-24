"""
Skincare Recommendation System — Redesigned UI
Minimal, feminine, Notion-inspired aesthetic with soft pinks and generous whitespace.
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import SVDRecommender
from src.hybrid_model import IngredientAwareHybrid, SKIN_CONCERN_MAP

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skin Ritual",
    page_icon="",
    layout="centered"
)

# ── Global styles ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #fdf8f6 !important;
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    color: #2d2520;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── Main container ── */
.block-container {
    max-width: 720px !important;
    padding: 4rem 2rem 8rem !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Typography ── */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.3rem;
    font-weight: 500;
    line-height: 1.15;
    color: #1f1a17;
    margin: 0 0 0.6rem;
    letter-spacing: -0.02em;
}
.hero-title em {
    font-style: italic;
    color:  #b8745e;;
}
.hero-sub {
 font-size: 1.05rem;
    color: #8f7f78;
    font-weight: 300;
    margin: 0 0 3.5rem;
    }

/* ── Section label ── */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #a8644d;   /* stronger contrast */
    margin: 2.5rem 0 1rem;
}

/* ── Cards ── */
.card {
    background: #ffffff;
    border: 1px solid #f0e8e3;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 0.75rem;
    transition: box-shadow 0.2s ease;
}
.card:hover {
    box-shadow: 0 4px 24px rgba(196,131,106,0.10);
}
.card-product {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: #2d2520;
    margin: 0 0 0.15rem;
}
.card-brand {
    font-size: 0.8rem;
    color: #b8a8a0;
    font-weight: 400;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.card-meta {
    display: flex;
    gap: 1.5rem;
    font-size: 0.88rem;
    color: #7a6860;
}
.card-meta span { display: flex; align-items: center; gap: 0.3rem; }
.card-ingredients {
    margin-top: 0.9rem;
    padding-top: 0.9rem;
    border-top: 1px solid #f5eeea;
    font-size: 0.83rem;
    color: #9c8b84;
    line-height: 1.6;
}
.card-pill {
    display: inline-block;
    background: #fdf0eb;
    color: #c4836a;
    border-radius: 50px;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-weight: 400;
    margin: 0 4px 4px 0;
}
[data-testid="stTabs"] button {
    color: #7a6860 !important;
    font-weight: 400 !important;
    opacity: 0.7;
    border-bottom: 2px solid transparent !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #2d2520 !important;
    font-weight: 500 !important;
    opacity: 1;
    border-bottom: 2px solid #b8745e !important;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stSelectbox"] > label,
div[data-testid="stRadio"] > label,
div[data-testid="stSlider"] > label {
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #c4836a !important;
    margin-bottom: 0.4rem !important;
}

div[data-testid="stSelectbox"] > div > div,
div[data-testid="stTextInput"] > div > div > input {
    background: #ffffff !important;
    border: 1px solid #f0e8e3 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    color: #2d2520 !important;
    box-shadow: none !important;
}

div[data-testid="stSelectbox"] > div > div:hover,
div[data-testid="stTextInput"] > div > div > input:focus {
    border-color: #c4836a !important;
    box-shadow: 0 0 0 2px rgba(196,131,106,0.12) !important;
}

/* Slider track */
[data-testid="stSlider"] .st-emotion-cache-1gv3e14,
[data-testid="stSlider"] div[role="slider"] {
    background: #c4836a !important;
}
[data-testid="stSlider"] .stSlider > div > div > div > div {
    background: #c4836a !important;
}

/* Radio buttons — force label text to always be dark and visible */
[data-testid="stRadio"] label {
    color: #2d2520 !important;
    font-size: 0.9rem !important;
}
[data-testid="stRadio"] div[role="radio"] p,
[data-testid="stRadio"] div[role="radio"] span {
    color: #2d2520 !important;
}
[data-testid="stRadio"] div[role="radio"][aria-checked="true"] div {
    background: #c4836a !important;
    border-color: #c4836a !important;
}
[data-testid="stRadio"] div[role="radio"][aria-checked="true"] p,
[data-testid="stRadio"] div[role="radio"][aria-checked="true"] span {
    color: #2d2520 !important;
    font-weight: 500 !important;
}

/* ── Primary button ── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: #2d2520 !important;
    color: #fdf8f6 !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.65rem 2.2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.04em !important;
    transition: background 0.2s ease, transform 0.15s ease !important;
    width: auto !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #c4836a !important;
    transform: translateY(-1px) !important;
}

/* Secondary button */
div[data-testid="stButton"] > button:not([kind="primary"]) {
    background: transparent !important;
    color: #c4836a !important;
    border: 1px solid #f0e8e3 !important;
    border-radius: 50px !important;
    padding: 0.5rem 1.6rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.04em !important;
}
div[data-testid="stButton"] > button:not([kind="primary"]):hover {
    border-color: #c4836a !important;
    background: #fdf0eb !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: #fdf0eb !important;
    border: 1px solid #f0e8e3 !important;
    border-radius: 12px !important;
    color: #7a6860 !important;
    font-size: 0.88rem !important;
}

/* ── Divider ── */
.soft-divider {
    border: none;
    border-top: 1px solid #f0e8e3;
    margin: 2rem 0;
}

/* ── Fade-in animation ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeUp 0.5s ease both; }
.fade-in-1 { animation-delay: 0.05s; }
.fade-in-2 { animation-delay: 0.12s; }
.fade-in-3 { animation-delay: 0.19s; }
.fade-in-4 { animation-delay: 0.26s; }
.fade-in-5 { animation-delay: 0.33s; }

/* ── Lookup tab input ── */
.stTextInput { margin-bottom: 0.5rem !important; }

/* ── Tag line beneath hero ── */
.tagline-strip {
    display: flex;
    gap: 1.5rem;
    margin-bottom: 3rem;
    flex-wrap: wrap;
}
.tagline-item {
    font-size: 0.75rem;
    color: #b8a8a0;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 400;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.tagline-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #e8cfc4;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)


# ── Load models (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    cbr    = ContentBasedRecommender.load()
    svd    = SVDRecommender.load()
    hybrid = IngredientAwareHybrid.load()
    return cbr, svd, hybrid

@st.cache_data
def load_data():
    products = pd.read_csv('data/cleaned/products.csv')
    reviews  = pd.read_csv('data/cleaned/reviews.csv')
    return products, reviews

cbr, svd, hybrid = load_models()
products, reviews = load_data()


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fade-in">
    <p class="hero-title">Your skin,<br><em>understood.</em></p>
    <p class="hero-sub">Tell us about your skin and we'll find products that actually make sense for you.</p>
    <div class="tagline-strip">
        <span class="tagline-item"><span class="tagline-dot"></span> Ingredient-aware</span>
        <span class="tagline-item"><span class="tagline-dot"></span> Skin-type matched</span>
        <span class="tagline-item"><span class="tagline-dot"></span> No fluff</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Find Products", "Explore Similar"])

# ── Tab 1: Recommendations ────────────────────────────────────────────────────
with tab1:

    st.markdown('<p class="section-label">Your skin</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        skin_type = st.selectbox(
            "Skin type",
            ["Normal", "Dry", "Oily", "Combination", "Sensitive"],
            label_visibility="visible"
        )
    with col2:
        skin_concern = st.selectbox(
            "Primary concern",
            list(SKIN_CONCERN_MAP.keys()),
            label_visibility="visible"
        )

    st.markdown('<p class="section-label" style="margin-top:1.8rem">Budget</p>', unsafe_allow_html=True)
    budget = st.slider("Max price (USD)", 5, 200, 80, label_visibility="collapsed")
    st.markdown(f'<p style="font-size:0.82rem;color:#9c8b84;margin-top:-0.5rem">Up to <strong style="color:#2d2520">${budget}</strong></p>', unsafe_allow_html=True)
    
    st.markdown('<p class="section-label" style="margin-top:1.8rem">Do you have a purchase history?</p>', unsafe_allow_html=True)
    st.markdown("""
<style>
div[data-testid="stRadio"] label span {
    color: #1f1a17 !important;
    font-size: 0.95rem !important;
    font-weight: 400 !important;
    opacity: 1 !important;
}

div[data-testid="stRadio"] label {
    color: #1f1a17 !important;
    font-weight: 500 !important;
}

div[data-testid="stRadio"] div[role="radio"][aria-checked="true"] span {
    color: #1f1a17 !important;
    font-weight: 600 !important;
}

div[data-testid="stRadio"] {
    color: #1f1a17 !important;
}
</style>
""", unsafe_allow_html=True)

    has_history = st.radio(
        "History",
        ["No, I'm new here", "Yes, use my profile"],
        label_visibility="collapsed",
        horizontal=True
    )

    user_id = None
    if has_history == "Yes, use my profile":
        st.markdown('<p class="section-label" style="margin-top:1.5rem">Your user ID</p>', unsafe_allow_html=True)
        user_id = st.text_input("User ID", placeholder="e.g. user_00423", label_visibility="collapsed")

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    run = st.button("Find my matches", type="primary")

    # ── Results ──
    if run:
        with st.spinner(""):
            if has_history == "No, I'm new here" or not user_id:
                recs = cbr.recommend_for_user_profile(
                    skin_type=skin_type,
                    concerns=[skin_concern] + list(SKIN_CONCERN_MAP.get(skin_concern, [])),
                    budget_max=budget,
                    n=8
                )
            else:
                recs = hybrid.recommend(
                    user_id=user_id,
                    skin_type=skin_type,
                    skin_concern=skin_concern,
                    cf_recommender=svd,
                    reviews=reviews,
                    n=10
                )
                recs = recs[recs['price_usd'] <= budget]

        st.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)

        if recs.empty:
            st.info("No matches found — try increasing your budget or selecting a different concern.")
        else:
            n = len(recs)
            st.markdown(
                f'<p style="font-size:0.82rem;color:#9c8b84;margin-bottom:1.5rem">'
                f'{n} product{"s" if n != 1 else ""} for <strong style="color:#2d2520">{skin_type.lower()} skin</strong> '
                f'· {skin_concern.lower()}</p>',
                unsafe_allow_html=True
            )

            for i, (_, row) in enumerate(recs.iterrows()):
                delay_class = f"fade-in fade-in-{min(i+1, 5)}"
                name   = row.get('product_name', 'Unknown product')
                brand  = row.get('brand_name', '')
                price  = row.get('price_usd', 0)
                rating = row.get('rating', 0)

                # Key ingredients as pills
                ingredients_html = ""
                if 'key_ingredients_found' in row and pd.notna(row['key_ingredients_found']):
                    raw = str(row['key_ingredients_found'])
                    pills = [
                        f'<span class="card-pill">{ing.strip()}</span>'
                        for ing in raw.split(',') if ing.strip()
                    ]
                    if pills:
                        ingredients_html = f'<div class="card-ingredients">{"".join(pills)}</div>'

                st.markdown(f"""
<div class="card {delay_class}">
    <p class="card-product">{name}</p>
    <p class="card-brand">{brand}</p>
    <div class="card-meta">
        <span>${price:.0f}</span>
        <span style="color:#e8cfc4">·</span>
        <span>{rating:.1f} &nbsp;★</span>
    </div>
    {ingredients_html}
</div>
""", unsafe_allow_html=True)


# ── Tab 2: Similar products ───────────────────────────────────────────────────
with tab2:
    st.markdown("""
<div class="fade-in" style="margin-bottom:1.5rem">
    <p style="font-size:1rem;color:#9c8b84;font-weight:300;margin:0">
        Found something you like? Search for it and we'll find more like it.
    </p>
</div>
""", unsafe_allow_html=True)

    st.markdown('<p class="section-label">Search a product</p>', unsafe_allow_html=True)
    query = st.text_input("Product name", placeholder="e.g. CeraVe Moisturizing Cream", label_visibility="collapsed")

    if query:
        matches = products[
            products['product_name'].str.contains(query, case=False, na=False)
        ]
        if not matches.empty:
            selected_pid = st.selectbox(
                "Select the product",
                matches['product_id'].values,
                format_func=lambda pid: matches[
                    matches['product_id'] == pid]['product_name'].values[0],
                label_visibility="visible"
            )
            if st.button("Show similar products"):
                sims = cbr.recommend_for_product(selected_pid, n=8)

                st.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)
                st.markdown(
                    f'<p style="font-size:0.82rem;color:#9c8b84;margin-bottom:1.5rem">'
                    f'{len(sims)} similar products found</p>',
                    unsafe_allow_html=True
                )

                for i, (_, row) in enumerate(sims.iterrows()):
                    delay_class = f"fade-in fade-in-{min(i+1, 5)}"
                    name   = row.get('product_name', 'Unknown product')
                    brand  = row.get('brand_name', '')
                    price  = row.get('price_usd', 0)
                    rating = row.get('rating', 0)

                    ingredients_html = ""
                    if 'key_ingredients_found' in row and pd.notna(row['key_ingredients_found']):
                        raw = str(row['key_ingredients_found'])
                        pills = [
                            f'<span class="card-pill">{ing.strip()}</span>'
                            for ing in raw.split(',') if ing.strip()
                        ]
                        if pills:
                            ingredients_html = f'<div class="card-ingredients">{"".join(pills)}</div>'

                    st.markdown(f"""
<div class="card {delay_class}">
    <p class="card-product">{name}</p>
    <p class="card-brand">{brand}</p>
    <div class="card-meta">
        <span>${price:.0f}</span>
        <span style="color:#e8cfc4">·</span>
        <span>{rating:.1f} &nbsp;★</span>
    </div>
    {ingredients_html}
</div>
""", unsafe_allow_html=True)
        else:
            st.info("No products found — try a shorter or different search term.")