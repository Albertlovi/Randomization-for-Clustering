import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import calinski_harabasz_score

# Load the dataset
X = pd.read_csv('datasets/Dataset6.csv', sep=';', header=None).values  # Ensure X is a NumPy array
print(X)

# Ensure the results folder exists
results_folder = "results_Lloyd"
os.makedirs(results_folder, exist_ok=True)

# Helper function to plot and save a frame for GIF
def plot_kmeans(X, centers, labels, iteration, images):
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, label='Centers')
    ax.set_title(f"Iteration {iteration}")
    ax.legend()
    plt.close(fig)

    # Save the figure to an image and append it to the images list
    from io import BytesIO
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    images.append(Image.open(buffer))

# Lloyd's algorithm for K-Means
def kmeans_lloyds(X, k, max_iter=100, tol=1e-4):
    np.random.seed(42)  # For reproducibility
    centers = X[np.random.choice(len(X), k, replace=False)]
    labels = np.zeros(len(X))
    images = []

    for iteration in range(max_iter):
        # Step 1: Assign points to the nearest center
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)  # Compute distances to each center
        labels = np.argmin(distances, axis=1)  # Assign each point to the nearest center

        # Step 2: Capture current state for visualization
        plot_kmeans(X, centers, labels, iteration, images)

        # Step 3: Update centers
        new_centers = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]  # Handle empty clusters
            for i in range(k)
        ])

        # Step 4: Check for convergence
        if np.allclose(centers, new_centers, atol=tol):
            print(f"Converged at iteration {iteration}")
            break
        centers = new_centers

    return centers, labels, images, iteration

# Run K-Means
k = 3  # Number of clusters
final_centers, final_labels, images, final_iteration = kmeans_lloyds(X, k)

# Save the GIF
gif_filename = os.path.join(results_folder, 'kmeans_evolution.gif')
images[0].save(
    gif_filename,
    save_all=True,
    append_images=images[1:],
    loop=0,
    duration=500
)
print(f"GIF saved as '{gif_filename}'")

# Save the final clustering visualization
def save_final_plot(X, centers, labels, output_path):
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, label='Centers')
    ax.set_title(f"Final Clustering (at iteration {final_iteration})")
    ax.legend()
    fig.savefig(output_path)
    plt.close(fig)

final_plot_filename = os.path.join(results_folder, 'final_clustering.png')
save_final_plot(X, final_centers, final_labels, final_plot_filename)
print(f"Final clustering image saved as '{final_plot_filename}'")

# Calculate the Calinski–Harabasz index
ch_index = calinski_harabasz_score(X, final_labels)
print(f"Calinski–Harabasz Index: {ch_index}")




