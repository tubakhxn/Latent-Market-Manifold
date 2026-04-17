
from data_loader import load_market_data
from features import compute_features
from model import reduce_dimensionality
from visualizer import plot_3d_trajectory
import os
import numpy as np
import pandas as pd

def generate_synthetic_data(n_assets=5, n_days=200, seed=42):
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01', periods=n_days)
    data = []
    for asset in [f'Asset{i+1}' for i in range(n_assets)]:
        price = 100 + np.cumsum(np.random.randn(n_days) * 2 + np.random.randn() * 0.5)
        for i, d in enumerate(dates):
            data.append({'Date': d, 'Asset': asset, 'Price': price[i]})
    return pd.DataFrame(data)

if __name__ == '__main__':
    # Always use large synthetic data for demo
    print('Generating large synthetic data for visualization...')
    df = generate_synthetic_data(n_assets=5, n_days=200)
    features = compute_features(df)
    # Drop rows with NaN in key features to ensure valid PCA
    features_valid = features.dropna(subset=['Return', 'Volatility', 'Momentum'])
    # Choose method: 'pca', 'tsne', or 'umap'
    latent = reduce_dimensionality(features_valid, method='pca')
    # Color by volatility as example
    plot_3d_trajectory(latent, color_by=features_valid['Volatility'].fillna(0).values)
