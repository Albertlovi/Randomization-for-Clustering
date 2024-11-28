import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def plot_kmeans_experiment(df, centers, labels, iteration, images, title_prefix=""):
    '''
    This function plots a given clustering in a given iteration of the K-Means algorithm.
    '''
    fig, ax = plt.subplots()
    ax.scatter(df[:, 0], df[:, 1], c=labels, cmap='viridis', alpha=0.6)
    ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, label='Centers')
    ax.set_title(f"{title_prefix} iteration {iteration}")
    ax.legend()
    plt.close(fig)

    # Save figure to image buffer and append to the images list
    from io import BytesIO
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    images.append(Image.open(buffer))


def kmeans_iteration_by_iteration(df, k, initial_centers, max_iter=100, tol=1e-4, title_prefix=""):
    '''
    This funciton runs the K-Means clustering method for a specific experiment using its initial centers. At each 
    iteration of the process it saves the final picture. 
    '''
    centers = initial_centers.copy()
    labels = np.zeros(len(df))
    images = []

    for iteration in range(max_iter):
        # Assign points to the nearest center
        distances = np.linalg.norm(df[:, np.newaxis] - centers, axis=2)  
        labels = np.argmin(distances, axis=1)  

        # Capture current state for visualization
        plot_kmeans_experiment(df, centers, labels, iteration, images, title_prefix=title_prefix)

        # Update centers
        new_centers = np.array([
            df[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]  # Handle empty clusters
            for i in range(k)
        ])

        # Check for convergence
        if np.allclose(centers, new_centers, atol=tol):
            break
        centers = new_centers

    return centers, labels, images, iteration


def save_experiment_results(df, experiment, results_folder, experiment_type):
    '''
    This function saves a .gif file with the evolution of a given experiment.
    '''
    print(f'Saving results for {experiment_type}')
    initial_centers = experiment["initial_centers"]
    k = initial_centers.shape[0]

    # Run K-Means by iteration for this experiment
    final_centers, final_labels, images, final_iteration = kmeans_iteration_by_iteration(
        df, k, initial_centers, title_prefix=experiment_type
    )

    # Save GIF
    gif_filename = os.path.join(results_folder, f'{experiment_type}.gif')
    images[0].save(
        gif_filename,
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=500
    )

    # Save final clustering visualization
    final_plot_filename = os.path.join(results_folder, f'{experiment_type}.png')
    fig, ax = plt.subplots()
    ax.scatter(df[:, 0], df[:, 1], c=final_labels, cmap='viridis', alpha=0.6)
    ax.scatter(final_centers[:, 0], final_centers[:, 1], c='red', marker='x', s=100, label='Centers')
    ax.set_title(f"{experiment_type} final clustering (at iteration {final_iteration})")
    ax.legend()
    fig.savefig(final_plot_filename)
    plt.close(fig)
