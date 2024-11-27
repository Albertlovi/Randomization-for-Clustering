import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Plot the clustering evolution at every iteration of the experiment
def plot_kmeans_experiment(X, centers, labels, iteration, images, title_prefix=""):
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, label='Centers')
    ax.set_title(f"{title_prefix} Iteration {iteration}")
    ax.legend()
    plt.close(fig)

    # Save figure to image buffer and append to the images list
    from io import BytesIO
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    images.append(Image.open(buffer))

# Run K-Means for a specific experiment using its initial centers step by step to safe the final picture after each iteration
def kmeans_iteration_by_iteration(X, k, initial_centers, max_iter=100, tol=1e-4, title_prefix=""):
    centers = initial_centers.copy()
    labels = np.zeros(len(X))
    images = []

    for iteration in range(max_iter):
        # Step 1: Assign points to the nearest center
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)  # Compute distances to each center
        labels = np.argmin(distances, axis=1)  # Assign each point to the nearest center

        # Step 2: Capture current state for visualization
        plot_kmeans_experiment(X, centers, labels, iteration, images, title_prefix=title_prefix)

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

# Plot and save results for an experiment
def save_experiment_results(X, experiment, results_folder, experiment_type):
    initial_centers = experiment["initial_centers"]
    k = initial_centers.shape[0]

    # Run K-Means by iteration for this experiment
    final_centers, final_labels, images, final_iteration = kmeans_iteration_by_iteration(
        X, k, initial_centers, title_prefix=experiment_type
    )

    # Save GIF
    gif_filename = os.path.join(results_folder, f'{experiment_type}_evolution.gif')
    images[0].save(
        gif_filename,
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=500
    )
    print(f"GIF for {experiment_type} saved as '{gif_filename}'")

    # Save final clustering visualization
    final_plot_filename = os.path.join(results_folder, f'{experiment_type}_final_clustering.png')
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=final_labels, cmap='viridis', alpha=0.6)
    ax.scatter(final_centers[:, 0], final_centers[:, 1], c='red', marker='x', s=100, label='Centers')
    ax.set_title(f"{experiment_type} Final Clustering (at iteration {final_iteration})")
    ax.legend()
    fig.savefig(final_plot_filename)
    plt.close(fig)
    print(f"Final clustering image for {experiment_type} saved as '{final_plot_filename}'")
