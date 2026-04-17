from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import numpy as np

def reduce_dimensionality(features, method='pca', n_components=3):
    """
    Reduce features to 3D latent space using PCA, t-SNE, or UMAP.
    """
    X = features.select_dtypes(include=[np.number]).dropna(axis=1, how='any')
    n_samples, n_features = X.shape
    if n_samples < 3 or n_features < 3:
        print(f"Warning: Not enough samples/features for 3D embedding (samples={n_samples}, features={n_features}). Using min(s, f) components.")
        n_components = min(n_samples, n_features, 2) if min(n_samples, n_features) >= 2 else 1
    if method == 'pca':
        model = PCA(n_components=n_components)
    elif method == 'tsne':
        model = TSNE(n_components=n_components, random_state=42)
    elif method == 'umap':
        model = umap.UMAP(n_components=n_components, random_state=42)
    else:
        raise ValueError('Unknown method')
    latent = model.fit_transform(X)
    # If latent is 1D or 2D, pad to 3D for visualization
    if latent.shape[1] == 1:
        latent = np.hstack([latent, np.zeros((latent.shape[0], 2))])
    elif latent.shape[1] == 2:
        latent = np.hstack([latent, np.zeros((latent.shape[0], 1))])
    return latent
