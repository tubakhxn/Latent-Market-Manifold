import numpy as np
import pandas as pd

def compute_features(df):
    """
    Compute returns, rolling volatility, momentum, and cross-asset correlations.
    Assumes df has columns: Date, Asset, Price
    Returns a DataFrame with engineered features.
    """
    df = df.sort_values(['Asset', 'Date'])
    df['Return'] = df.groupby('Asset')['Price'].pct_change()
    df['Volatility'] = df.groupby('Asset')['Return'].rolling(window=20).std().reset_index(0, drop=True)
    df['Momentum'] = df.groupby('Asset')['Price'].pct_change(periods=10)
    # Pivot to wide format for correlation
    pivot = df.pivot(index='Date', columns='Asset', values='Return')
    corr = pivot.rolling(window=20).corr().groupby('Date').last()
    # Flatten correlation matrix for each date
    corr_features = corr.stack().reset_index()
    # After stacking, columns are usually: Date, Asset1, Asset2, value
    if corr_features.shape[1] == 4:
        corr_features.columns = ['Date', 'Asset1', 'Asset2', 'Correlation']
    elif corr_features.shape[1] == 3:
        corr_features.columns = ['Date', 'Asset2', 'Correlation']
        corr_features['Asset1'] = corr_features['Asset2']  # fallback for single asset
    else:
        corr_features.columns = [f'col{i}' for i in range(corr_features.shape[1]-1)] + ['Correlation']
    # Merge features back
    features = df.merge(corr_features, left_on=['Date', 'Asset'], right_on=['Date', 'Asset1'], how='left')
    return features
