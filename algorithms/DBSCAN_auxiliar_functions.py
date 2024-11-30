import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def plot_dbscan_experiment(df, labels, iteration, images, title_prefix=""):
    '''
    This function plots a given clustering at a specific iteration of DBSCAN.
    '''
    unique_labels = set(labels)
    fig, ax = plt.subplots()
    
    # Plot each cluster with a different color
    for k in unique_labels:
        if k == -1:
            # Black used for noise points
            color = 'black'
            label_name = 'Noise'
        else:
            color = plt.cm.viridis(k / max(unique_labels))
            label_name = f"Cluster {k}"
        
        class_member_mask = labels == k
        ax.scatter(
            df[class_member_mask, 0],
            df[class_member_mask, 1],
            c=[color],
            label=label_name,
            alpha=0.6
        )

    ax.set_title(f"{title_prefix} iteration {iteration}")
    ax.legend()
    plt.close(fig)

    # Save figure to image buffer and append to the images list
    from io import BytesIO
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    images.append(Image.open(buffer))


def dbscan_iteration_by_iteration(df, eps, min_samples, title_prefix=""):
    '''
    This function runs the DBSCAN algorithm and captures the clustering visualization at each iteration.
    '''
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    # Run DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(data_scaled)

    # Create images list to simulate "iterations"
    images = []
    plot_dbscan_experiment(data_scaled, labels, iteration=0, images=images, title_prefix=title_prefix)

    return labels, images


def save_dbscan_results(df, eps, min_samples, results_folder, experiment_type):
    '''
    This function saves a .gif file with the evolution of DBSCAN clustering.
    '''
    print(f'Saving results for {experiment_type}')

    # Run DBSCAN and capture visualization
    final_labels, images = dbscan_iteration_by_iteration(df, eps, min_samples, title_prefix=experiment_type)

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
    unique_labels = set(final_labels)
    for k in unique_labels:
        if k == -1:
            color = 'black'
            label_name = 'Noise'
        else:
            color = plt.cm.viridis(k / max(unique_labels))
            label_name = f"Cluster {k}"

        class_member_mask = final_labels == k
        ax.scatter(
            df[class_member_mask, 0],
            df[class_member_mask, 1],
            c=[color],
            label=label_name,
            alpha=0.6
        )
    ax.set_title(f"{experiment_type} final clustering")
    ax.legend()
    fig.savefig(final_plot_filename)
    plt.close(fig)