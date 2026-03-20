import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.decomposition import PCA
from scipy.spatial import KDTree

def read_kmeans_binary(filename):
    """
    Reads the ECE415 binary format[cite: 1]:
    [numObjs (int32)] [numCoords (int32)] [Raw Floats (float32)...]
    """
    with open(filename, 'rb') as f:
        header = np.fromfile(f, dtype=np.int32, count=2)
        n_objs, n_coords = header
        data = np.fromfile(f, dtype=np.float32)
        return data.reshape((n_objs, n_coords))

# 1. Load the benchmark file
# This file contains 17,695 objects with 20 dimensions
X_raw = read_kmeans_binary('texture17695.bin')
N, D = X_raw.shape

# 2. PCA Projection (20D -> 2D)
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_raw)

# 3. Clustering Parameters
K = 8
MAX_ITER = 200
centroids = X_2d[np.random.choice(N, K, replace=False)]
history = [centroids.copy()]

# --- Plot Setup ---
fig, ax = plt.subplots(figsize=(12, 8), dpi=120)
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

# Pre-calculate background grid for Voronoi territories
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]

def update(frame):
    global centroids
    ax.clear()
    ax.axis('off')

    # Step A:find_nearest_cluster logic
    tree = KDTree(centroids)
    _, labels = tree.query(X_2d)
    
    # Step B: Draw Territories (Decision Boundaries)
    _, grid_labels = tree.query(grid_points)
    grid_labels = grid_labels.reshape(xx.shape)
    ax.contourf(xx, yy, grid_labels, alpha=0.15, cmap='plasma')

    # Step C: Plot Low alpha for big data
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, s=1, cmap='plasma', alpha=0.1)

    # Step D: Recalculate Means
    new_centroids = np.array([X_2d[labels == i].mean(axis=0) if np.any(labels == i) 
                             else centroids[i] for i in range(K)])
    
    # Step E: Plot Centroid Movement Trails
    history.append(new_centroids.copy())
    for i in range(K):
        path = np.array([h[i] for h in history])
        ax.plot(path[:, 0], path[:, 1], color='white', alpha=0.4, lw=1, ls=':')

    # Draw Current Centroids
    ax.scatter(new_centroids[:, 0], new_centroids[:, 1], s=180, marker='*', 
               color='white', edgecolor='cyan', linewidth=1.2, zorder=10)

    # HUD / Technical Overlay
    ax.text(0.02, 0.96, f"BIG DATA K-MEANS OPTIMIZATION\nDATASET: texture17695.bin | N={N} | D={D}", 
            transform=ax.transAxes, color='cyan', fontsize=10, fontweight='bold', fontfamily='monospace')
    
    status = "STATUS: CONVERGED" if np.allclose(centroids, new_centroids, atol=1e-5) else f"ITERATION: {frame+1}"
    ax.text(0.98, 0.02, status, transform=ax.transAxes, color='white', ha='right', fontsize=12, fontfamily='monospace')

    centroids = new_centroids

# Create and save the animation
anim = animation.FuncAnimation(fig, update, frames=MAX_ITER, interval=500, repeat=False)
plt.tight_layout()

print("Rendering GIF... this may take a minute for 17,695 points (i have an i5 4690k).")
# Saves to the current directory
anim.save('kmeans_big_data.gif', writer='pillow', fps=5) 
print("Success! kmeans_big_data.gif saved in the current folder.")
plt.show()