import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(product_path, reviews_path):
    """Load and merge the Sephora dataset."""
    products = pd.read_csv(product_path)
    reviews = pd.read_csv(reviews_path, dtype={'author_id': str}, low_memory=False)
    print(f"Products: {products.shape} | Reviews: {reviews.shape}")
    return products, reviews


def clean_products(df):
    """Clean the products dataframe."""
    df = df.copy()
    
    # Drop duplicates
    df.drop_duplicates(subset='product_id', inplace=True)
    
    # Fill missing ingredient info
    df['ingredients'] = df['ingredients'].fillna('')
    df['highlights'] = df['highlights'].fillna('')
    
    # Standardize skin type columns (already boolean-like in this dataset)
    skin_type_cols = ['skin_type_combination', 'skin_type_dry',
                      'skin_type_normal', 'skin_type_oily', 'skin_type_sensitive']
    for col in skin_type_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    
    # Price cleaning
    df['price_usd'] = pd.to_numeric(df['price_usd'], errors='coerce')
    df['price_usd'] = df['price_usd'].fillna(df['price_usd'].median())
    
    return df


def clean_reviews(df):
    """Clean the reviews dataframe."""
    df = df.copy()
    df.drop_duplicates(subset=['author_id','product_id'], inplace=True)
    df.dropna(subset=['rating','author_id','product_id'], inplace=True)
    df['rating'] = df['rating'].astype(float)
    
    # Keep only users with at least 5 reviews (cold start mitigation)
    review_counts = df['author_id'].value_counts()
    active_users  = review_counts[review_counts >= 5].index
    df = df[df['author_id'].isin(active_users)]
    return df


def run_eda(products, reviews, output_dir='reports/'):
    """Generate basic EDA plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Rating distribution
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    reviews['rating'].value_counts().sort_index().plot(
        kind='bar', ax=axes[0], color='steelblue', edgecolor='white')
    axes[0].set_title('Rating Distribution')
    axes[0].set_xlabel('Rating')
    
    products['primary_category'].value_counts().head(10).plot(
        kind='barh', ax=axes[1], color='coral')
    axes[1].set_title('Top 10 Product Categories')
    
    review_counts = reviews.groupby('author_id').size()
    axes[2].hist(review_counts, bins=50, color='mediumseagreen', edgecolor='white')
    axes[2].set_title('Reviews per User')
    axes[2].set_xlabel('Number of Reviews')
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/eda_overview.png', dpi=150)
    plt.close()
    print(f"[EDA] Saved plots to {output_dir}")
    
    # Summary stats
    print("\n=== Products ===")
    print(products.describe(include='all').T[['count','unique','top','mean']].dropna(how='all'))
    print("\n=== Reviews ===")
    print(reviews[['rating','total_feedback_count','total_pos_feedback_count']].describe())


if __name__ == '__main__':
    products, reviews = load_data(
        'data/product_info.csv',
        'data/reviews_0-250.csv'   # merge multiple review CSVs as needed
    )
    products = clean_products(products)
    reviews  = clean_reviews(reviews)
    run_eda(products, reviews)
    
    # Save cleaned data
    os.makedirs('data/cleaned', exist_ok=True)
    products.to_csv('data/cleaned/products.csv', index=False)
    reviews.to_csv('data/cleaned/reviews.csv',  index=False)
    print("Saved cleaned files.")