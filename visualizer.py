import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_trajectory(latent, color_by, labels=None, title='3D Latent Market Manifold', colorscale='plasma'):
    """
    Plot a 3D trajectory in latent space with color and optional cluster labels using matplotlib.
    """
    x, y, z = latent[:,0], latent[:,1], latent[:,2]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    p = ax.scatter(x, y, z, c=color_by, cmap=colorscale, s=8, alpha=0.85)
    ax.plot(x, y, z, color='white', linewidth=1.5, alpha=0.7)
    ax.set_title(title, color='white')
    ax.set_xlabel('Latent X', color='white')
    ax.set_ylabel('Latent Y', color='white')
    ax.set_zlabel('Latent Z', color='white')
    ax.tick_params(colors='white')
    fig.patch.set_facecolor('black')
    cbar = fig.colorbar(p, ax=ax, pad=0.1)
    cbar.set_label('Volatility', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.show()
