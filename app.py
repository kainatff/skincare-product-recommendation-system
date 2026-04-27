"""
Skincare Recommendation System — Modernized UI
Warm, playful, editorial aesthetic with fixed radio button styling.
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
    page_icon="🌸",
    layout="centered"
)

# ── Global styles ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Outfit:wght@300;400;500;600&display=swap');

/* ── Reset & base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #fef6f0 !important;
    font-family: 'Outfit', sans-serif;
    font-weight: 300;
    color: #2d2520;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Main container ── */
.block-container {
    max-width: 740px !important;
    padding: 3.5rem 2.5rem 8rem !important;
}

/* ── Typography ── */
.hero-eyebrow {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #c4836a;
    margin-bottom: 0.8rem;
    display: block;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.6rem;
    font-weight: 700;
    line-height: 1.1;
    color: #1f1a17;
    margin: 0 0 1rem;
    letter-spacing: -0.02em;
}
.hero-title em {
    font-style: italic;
    color: #c4836a;
}
.hero-sub {
    font-size: 1.05rem;
    color: #8f7f78;
    font-weight: 300;
    margin: 0 0 2.5rem;
    line-height: 1.7;
    max-width: 480px;
}

/* ── Accent badge strip ── */
.badge-strip {
    display: flex;
    gap: 0.6rem;
    margin-bottom: 3rem;
    flex-wrap: wrap;
}
.badge {
    font-size: 0.72rem;
    color: #c4836a;
    background: #fff0ea;
    border: 1px solid #f5d9cc;
    border-radius: 50px;
    padding: 0.3rem 0.9rem;
    font-weight: 500;
    letter-spacing: 0.05em;
}

/* ── Section label ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #c4836a;
    margin: 2.8rem 0 0.9rem;
    display: block;
}

/* ── Info panel ── */
.info-panel {
    background: #fff8f5;
    border: 1px solid #f5d9cc;
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.8rem;
    font-size: 0.85rem;
    color: #8f7f78;
    line-height: 1.6;
}

/* ── Cards ── */
.card {
    background: #ffffff;
    border: 1px solid #f0e6e0;
    border-radius: 18px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 0.8rem;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, #c4836a, #e8a98f);
    border-radius: 0 3px 3px 0;
    opacity: 0;
    transition: opacity 0.25s ease;
}
.card:hover {
    box-shadow: 0 8px 32px rgba(196,131,106,0.12);
    transform: translateY(-2px);
}
.card:hover::before {
    opacity: 1;
}
.card-rank {
    position: absolute;
    top: 1.2rem;
    right: 1.5rem;
    font-size: 0.7rem;
    color: #e8cfc4;
    font-weight: 600;
    letter-spacing: 0.1em;
}
.card-product {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    color: #1f1a17;
    margin: 0 0 0.2rem;
    padding-right: 2.5rem;
}
.card-brand {
    font-size: 0.75rem;
    color: #c4b0a8;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
}
.card-meta {
    display: flex;
    gap: 0.5rem;
    font-size: 0.86rem;
    color: #7a6860;
    align-items: center;
    flex-wrap: wrap;
}
.meta-price {
    background: #f5f0ed;
    border-radius: 6px;
    padding: 2px 8px;
    font-weight: 600;
    color: #2d2520;
    font-size: 0.84rem;
}
.meta-rating {
    color: #c4836a;
    font-weight: 500;
}
.card-ingredients {
    margin-top: 0.9rem;
    padding-top: 0.9rem;
    border-top: 1px solid #f5eeea;
    font-size: 0.82rem;
    color: #9c8b84;
    line-height: 1.8;
}
.card-pill {
    display: inline-block;
    background: #fdf0eb;
    color: #b8745e;
    border-radius: 50px;
    padding: 2px 10px;
    font-size: 0.73rem;
    font-weight: 500;
    margin: 0 3px 3px 0;
    border: 1px solid #f5d9cc;
}

/* ── Tabs ── */
[data-testid="stTabs"] {
    border-bottom: 1px solid #f0e6e0 !important;
}
[data-testid="stTabs"] button {
    color: #b8a8a0 !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 400 !important;
    font-size: 0.92rem !important;
    border-bottom: 2px solid transparent !important;
    padding-bottom: 0.7rem !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #2d2520 !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #c4836a !important;
}

/* ── Selectbox ── */
div[data-testid="stSelectbox"] > label,
div[data-testid="stSlider"] > label {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #c4836a !important;
    margin-bottom: 0.4rem !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #ffffff !important;
    border: 1.5px solid #f0e6e0 !important;
    border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.92rem !important;
    color: #2d2520 !important;
    box-shadow: none !important;
    transition: border-color 0.2s ease !important;
}
div[data-testid="stSelectbox"] > div > div:hover {
    border-color: #c4836a !important;
}

/* ── Text input ── */
div[data-testid="stTextInput"] > label {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #c4836a !important;
}
div[data-testid="stTextInput"] > div > div > input {
    background: #ffffff !important;
    border: 1.5px solid #f0e6e0 !important;
    border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.92rem !important;
    color: #2d2520 !important;
    box-shadow: none !important;
    padding: 0.6rem 1rem !important;
}
div[data-testid="stTextInput"] > div > div > input:focus {
    border-color: #c4836a !important;
    box-shadow: 0 0 0 3px rgba(196,131,106,0.1) !important;
}
div[data-testid="stTextInput"] > div > div > input::placeholder {
    color: #c4b0a8 !important;
}

/* ── Slider ── */
[data-testid="stSlider"] div[role="slider"] {
    background: #c4836a !important;
    border-color: #c4836a !important;
}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #c4b0a8 !important;
    font-size: 0.8rem !important;
}

/* ── RADIO BUTTONS — full fix for highlight/selection issue ── */
/* Remove all browser text selection highlights */
[data-testid="stRadio"] * {
    -webkit-user-select: none !important;
    -moz-user-select: none !important;
    user-select: none !important;
}

/* Hide the default radio container label */
[data-testid="stRadio"] > label {
    display: none !important;
}

/* The outer radio group wrapper */
[data-testid="stRadio"] > div {
    display: flex !important;
    gap: 0.7rem !important;
    flex-wrap: wrap !important;
}

/* Each radio option wrapper — matches primary button style, slightly smaller */
[data-testid="stRadio"] > div > label {
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
    background: transparent !important;
    border: 1.5px solid #2d2520 !important;
    border-radius: 50px !important;
    padding: 0.55rem 1.5rem !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    color: #2d2520 !important;
    letter-spacing: 0.04em !important;
    background-clip: padding-box !important;
    box-shadow: 0 2px 8px rgba(45,37,32,0.08) !important;
}

[data-testid="stRadio"] > div > label:hover {
    background: linear-gradient(135deg, #c4836a 0%, #d4957a 100%) !important;
    border-color: #c4836a !important;
    color: #fdf8f6 !important;
    box-shadow: 0 4px 14px rgba(196,131,106,0.28) !important;
    transform: translateY(-1px) !important;
}

/* Selected state — matches "Find my matches" dark gradient */
[data-testid="stRadio"] > div > label:has(input:checked) {
    background: linear-gradient(135deg, #2d2520 0%, #3d3028 100%) !important;
    border-color: #2d2520 !important;
    color: #fdf8f6 !important;
    font-weight: 500 !important;
    box-shadow: 0 4px 16px rgba(45,37,32,0.18) !important;
}

/* Hide the actual radio circle input */
[data-testid="stRadio"] > div > label > div:first-child {
    display: none !important;
}

/* The text span inside label */
[data-testid="stRadio"] > div > label > div:last-child p,
[data-testid="stRadio"] > div > label > div:last-child span,
[data-testid="stRadio"] > div > label p,
[data-testid="stRadio"] > div > label span {
    color: inherit !important;
    font-size: inherit !important;
    font-weight: inherit !important;
    font-family: inherit !important;
    margin: 0 !important;
    background: none !important;
}

/* Override any Streamlit internal selection/highlight on checked labels */
[data-testid="stRadio"] > div > label:has(input:checked) *,
[data-testid="stRadio"] > div > label:has(input:checked) p,
[data-testid="stRadio"] > div > label:has(input:checked) span {
    color: #fdf8f6 !important;
    background: none !important;
}

/* ── Primary button ── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #2d2520 0%, #3d3028 100%) !important;
    color: #fdf8f6 !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.75rem 2.4rem !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 16px rgba(45,37,32,0.18) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #c4836a 0%, #d4957a 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(196,131,106,0.3) !important;
}

/* Secondary button */
div[data-testid="stButton"] > button:not([kind="primary"]) {
    background: transparent !important;
    color: #c4836a !important;
    border: 1.5px solid #f0e6e0 !important;
    border-radius: 50px !important;
    padding: 0.55rem 1.6rem !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 400 !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stButton"] > button:not([kind="primary"]):hover {
    border-color: #c4836a !important;
    background: #fff8f5 !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: #fff8f5 !important;
    border: 1px solid #f5d9cc !important;
    border-radius: 14px !important;
    color: #7a6860 !important;
    font-size: 0.88rem !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #c4836a !important;
}

/* ── Divider ── */
.soft-divider {
    border: none;
    border-top: 1px solid #f0e6e0;
    margin: 2.2rem 0;
}

/* ── Results header ── */
.results-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 1.5rem;
}
.results-count {
    font-size: 0.82rem;
    color: #9c8b84;
}
.results-tag {
    background: #fff0ea;
    border: 1px solid #f5d9cc;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    color: #c4836a;
    font-weight: 500;
}

/* ── Fade-in animation ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in   { animation: fadeUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both; }
.fade-in-1 { animation-delay: 0.04s; }
.fade-in-2 { animation-delay: 0.10s; }
.fade-in-3 { animation-delay: 0.16s; }
.fade-in-4 { animation-delay: 0.22s; }
.fade-in-5 { animation-delay: 0.28s; }

/* ── Decorative background blob ── */
.bg-blob {
    position: fixed;
    top: -120px;
    right: -120px;
    width: 500px;
    height: 500px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(232,175,143,0.12) 0%, transparent 70%);
    pointer-events: none;
    z-index: -1;
}
</style>

<div class="bg-blob"></div>
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
    <span class="hero-eyebrow">✦ Personalized skincare</span>
    <p class="hero-title">Your skin,<br><em>understood.</em></p>
    <p class="hero-sub">Tell us about your skin and we'll surface products that genuinely work for you — not just what's trending.</p>
    <div class="badge-strip">
        <span class="badge">✦ Ingredient-aware</span>
        <span class="badge">✦ Skin-type matched</span>
        <span class="badge">✦ No fluff</span>
        <span class="badge">✦ Budget-friendly</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Helper: render a product card ─────────────────────────────────────────────
def render_card(row, index=0):
    delay_class = f"fade-in fade-in-{min(index+1, 5)}"
    name   = row.get('product_name', 'Unknown product')
    brand  = row.get('brand_name', '')
    price  = row.get('price_usd', 0)
    rating = row.get('rating', 0)

    stars = "★" * round(rating) + "☆" * (5 - round(rating))
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
    <span class="card-rank">#{index+1:02d}</span>
    <p class="card-product">{name}</p>
    <p class="card-brand">{brand}</p>
    <div class="card-meta">
        <span class="meta-price">${price:.0f}</span>
        <span class="meta-rating">{stars}</span>
        <span style="color:#b8a8a0;font-size:0.8rem">{rating:.1f} / 5</span>
    </div>
    {ingredients_html}
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["✦  Find Products", "◈  Explore Similar"])

# ── Tab 1: Recommendations ────────────────────────────────────────────────────
with tab1:

    st.markdown('<span class="section-label">Your skin profile</span>', unsafe_allow_html=True)

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

    st.markdown('<span class="section-label" style="margin-top:2rem">Budget</span>', unsafe_allow_html=True)
    budget = st.slider("Max price (USD)", 5, 200, 80, label_visibility="collapsed")
    st.markdown(
        f'<p style="font-size:0.82rem;color:#9c8b84;margin-top:-0.3rem">'
        f'Showing products up to <strong style="color:#c4836a">${budget}</strong></p>',
        unsafe_allow_html=True
    )

    st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
    run = st.button("✦  Find my matches", type="primary")

    # ── Results ──
    if run:
        with st.spinner("Finding your perfect matches…"):
            recs = cbr.recommend_for_user_profile(
                skin_type=skin_type,
                concerns=[skin_concern] + list(SKIN_CONCERN_MAP.get(skin_concern, [])),
                budget_max=budget,
                n=8
            )

        st.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)

        if recs.empty:
            st.info("✦ No matches found — try increasing your budget or selecting a different concern.")
        else:
            n = len(recs)
            st.markdown(f"""
<div class="results-header">
    <span class="results-count">{n} product{"s" if n != 1 else ""} found</span>
    <span class="results-tag">{skin_type} skin</span>
    <span class="results-tag">{skin_concern}</span>
</div>""", unsafe_allow_html=True)

            for i, (_, row) in enumerate(recs.iterrows()):
                render_card(row, i)