import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Rating Prediction Metrics (RMSE / MAE)
# ─────────────────────────────────────────────────────────────────────────────
def compute_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def compute_mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


# ─────────────────────────────────────────────────────────────────────────────
# Ranking Metrics (Precision@K / Recall@K)
# ─────────────────────────────────────────────────────────────────────────────
def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    recommended : ordered list of product_ids (top-K)
    relevant    : set of product_ids the user actually rated >= threshold
    """
    if not recommended:
        return 0.0
    top_k = recommended[:k]
    hits  = sum(1 for pid in top_k if pid in relevant)
    return hits / k

def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits  = sum(1 for pid in top_k if pid in relevant)
    return hits / len(relevant)

def average_precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """AP@K — rewards ranking relevant items higher."""
    hits, score = 0, 0.0
    for i, pid in enumerate(recommended[:k], start=1):
        if pid in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k) if relevant else 0.0

def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Normalized Discounted Cumulative Gain @ K."""
    dcg, idcg = 0.0, 0.0
    for i, pid in enumerate(recommended[:k], start=1):
        if pid in relevant:
            dcg += 1 / np.log2(i + 1)
    ideal = min(len(relevant), k)
    for i in range(1, ideal + 1):
        idcg += 1 / np.log2(i + 1)
    return dcg / idcg if idcg > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Full Evaluation Runner
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model_name: str, recommender, reviews_test: pd.DataFrame,
                   products: pd.DataFrame, k_values=(5, 10, 20),
                   rating_threshold=4.0) -> dict:
    """
    Run a full evaluation for a recommender:
    - RMSE / MAE on rating prediction
    - Precision@K, Recall@K, NDCG@K for ranking
    """
    results = {'model': model_name}
    y_true, y_pred = [], []
    all_prec, all_rec, all_ndcg = {k: [] for k in k_values}, \
                                   {k: [] for k in k_values}, \
                                   {k: [] for k in k_values}

    users = reviews_test['author_id'].unique()
    n_eval = min(300, len(users))  # cap for speed
    sampled_users = np.random.choice(users, n_eval, replace=False)

    for uid in sampled_users:
        user_test = reviews_test[reviews_test['author_id'] == uid]
        relevant  = set(user_test[user_test['rating'] >= rating_threshold]['product_id'])
        all_ids   = set(user_test['product_id'])

        # Rating prediction
        for _, row in user_test.iterrows():
            try:
                pred = recommender.predict(str(uid), str(row['product_id']))
                y_true.append(row['rating'])
                y_pred.append(pred)
            except Exception:
                pass

        # Ranking
        try:
            reviewed_ids = set(
                reviews_test[reviews_test['author_id'] != uid]['product_id'])
            recs_df = recommender.recommend(str(uid), products, reviewed_ids, n=max(k_values))
            if recs_df.empty:
                continue
            recs_list = recs_df['product_id'].tolist()
            for k in k_values:
                all_prec[k].append(precision_at_k(recs_list, relevant, k))
                all_rec[k].append(recall_at_k(recs_list, relevant, k))
                all_ndcg[k].append(ndcg_at_k(recs_list, relevant, k))
        except Exception:
            pass

    if y_true:
        results['rmse'] = compute_rmse(y_true, y_pred)
        results['mae']  = compute_mae(y_true, y_pred)

    for k in k_values:
        if all_prec[k]:
            results[f'precision@{k}'] = np.mean(all_prec[k])
            results[f'recall@{k}']    = np.mean(all_rec[k])
            results[f'ndcg@{k}']      = np.mean(all_ndcg[k])

    return results


def plot_comparison(results_list: list[dict], output_path='reports/model_comparison.png'):
    """Bar chart comparison of all models."""
    metrics = ['rmse','mae','precision@10','recall@10','ndcg@10']
    labels  = [r['model'] for r in results_list]
    x = np.arange(len(metrics))
    width = 0.8 / len(labels)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#4C72B0','#DD8452','#55A868','#C44E52']
    for i, (result, color) in enumerate(zip(results_list, colors)):
        vals = [result.get(m, 0) for m in metrics]
        ax.bar(x + i * width, vals, width, label=result['model'], color=color, alpha=0.85)

    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Eval] Comparison chart saved to {output_path}")