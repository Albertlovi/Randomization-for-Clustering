from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
from matplotlib import cm

def save_kmeans_clustering_final_plot(X, labels, centers=None, results_folder="results", experiment_type="KMeans"):
    '''
    This function saves a .png file with the final clustering visualization for KMeans,
    including cluster centers.
    
    Parameters:
    - X: The dataset (NumPy array).
    - labels: Cluster labels for the data points.
    - centers: Cluster centers (for KMeans). Default is None.
    - results_folder: Folder to save the plot.
    - experiment_type: Name for the clustering experiment (used in the plot title and filename).
    '''
    # Standardize data for consistent scaling
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = set(labels)
    
    # Plot each cluster
    for k in unique_labels:
        if k == -1:
            # Black used for noise points (if any, e.g., in DBSCAN)
            color = 'black'
            label_name = 'Noise'
        else:
            color = plt.cm.viridis(k / (max(unique_labels) if max(unique_labels) > 0 else 1))
            label_name = f"Cluster {k}"

        class_member_mask = labels == k
        ax.scatter(
            data_scaled[class_member_mask, 0],
            data_scaled[class_member_mask, 1],
            c=[color],
            label=label_name,
            alpha=0.6
        )

    # Plot cluster centers if available
    if centers is not None:
        centers_scaled = scaler.transform(centers)  # Scale the centers if data is scaled
        ax.scatter(
            centers_scaled[:, 0],
            centers_scaled[:, 1],
            c='red',
            s=200,
            marker='x',
            label='Centers'
        )

    ax.set_title(f"{experiment_type} final clustering")
    os.makedirs(results_folder, exist_ok=True)

    final_plot_filename = os.path.join(results_folder, f'{experiment_type}.png')
    fig.savefig(final_plot_filename)
    plt.close(fig)

    print(f"Final clustering plot saved as {final_plot_filename}")
    return final_plot_filename


def save_gmm(df, n_components, results_folder, experiment_type):
    '''
    This function saves a .png file with the GMM clustering and ellipsoids of the Gaussian components.
    '''
    # Ensure the results folder exists
    os.makedirs(results_folder, exist_ok=True)

    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    # Fit GMM
    gmm = GaussianMixture(n_components=n_components)
    labels = gmm.fit_predict(data_scaled)

    cmap = cm.viridis  # You can replace this with other colormaps like 'plasma', 'inferno', etc.
    colors = [cmap(i / n_components) for i in range(n_components)]  # Create a list of n_components colors


    # Create plot
    fig, ax = plt.subplots()

    # Plot the clusters
    unique_labels = set(labels)
    for k in unique_labels:
        class_member_mask = labels == k
        color = colors[k]
        ax.scatter(
            data_scaled[class_member_mask, 0],
            data_scaled[class_member_mask, 1],
            label=f"Cluster {k}",
            color=color,
            alpha=0.6
        )

    # Plot ellipsoids for each Gaussian component
    for k in range(n_components):
        # Extract the mean and covariance of the k-th Gaussian component
        mean = gmm.means_[k]
        cov = gmm.covariances_[k]
        
        # Create an ellipse based on the covariance matrix
        v, w = np.linalg.eigh(cov)  # Eigenvalues and eigenvectors of the covariance matrix
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Scale the axes of the ellipse
        angle = np.arctan2(w[1, 0], w[0, 0])  # Angle of rotation of the ellipse

        # Ellipse parameters
        ell = Ellipse(
            mean, 
            v[0], v[1], 
            angle=np.degrees(angle), 
            color='blue', 
            alpha=0.4, 
            lw=2
        )
        ax.add_patch(ell)

    ax.set_title(f"{experiment_type} final clustering")
    
    # Save final clustering visualization
    final_plot_filename = os.path.join(results_folder, f'{experiment_type}.png')
    fig.savefig(final_plot_filename)
    plt.close(fig)

    print(f"GMM clustering plot with ellipsoids saved as {final_plot_filename}")
    return final_plot_filename
