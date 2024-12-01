import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def save_dbscan_final_plot(df, eps, min_samples, results_folder, experiment_type):
    '''
    This function saves a .png file with the final clustering of DBSCAN.
    '''
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    # Run DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(data_scaled)

    # Create plot
    fig, ax = plt.subplots()
    unique_labels = set(labels)
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
            data_scaled[class_member_mask, 0],
            data_scaled[class_member_mask, 1],
            c=[color],
            label=label_name,
            alpha=0.6
        )
    ax.set_title(f"{experiment_type} final clustering")

    # Save final clustering visualization
    final_plot_filename = os.path.join(results_folder, f'{experiment_type}.png')
    fig.savefig(final_plot_filename)
    plt.close(fig)

    print(f"Final clustering plot saved as {final_plot_filename}")
    return final_plot_filename