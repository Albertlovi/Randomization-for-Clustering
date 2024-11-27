import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import calinski_harabasz_score

# Load the dataset
X = pd.read_csv('datasets/Dataset6.csv', sep=';', header=None).values  # Ensure X is a NumPy array
print(X)

# Initialize random centers
np.random.seed(53)  # For reproducibility
k = 5  # Number of clusters
initial_centers = X[np.random.choice(range(len(X)), k, replace=False)]

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

# Simulate Lloyd's algorithm manually for GIF creation
images = []
current_centers = initial_centers
labels = np.zeros(X.shape[0])

for iteration in range(15):  # Set max iterations
    # Assign points to nearest center
    distances = np.linalg.norm(X[:, np.newaxis] - current_centers, axis=2)
    labels = np.argmin(distances, axis=1)

    # Capture the current state
    plot_kmeans(X, current_centers, labels, iteration, images)

    # Update centers
    new_centers = np.array([X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else current_centers[i] for i in range(k)])
    
    # Break if centers do not change
    if np.allclose(current_centers, new_centers):
        break
    current_centers = new_centers

# Save as GIF
images[0].save(
    'kmeans_evolution.gif',
    save_all=True,
    append_images=images[1:],
    loop=0,
    duration=500
)

print("GIF saved as 'kmeans_evolution.gif'")



# Calculate the Calinski–Harabasz index
def calculate_ch_index(X, labels):
    return calinski_harabasz_score(X, labels)


# Compute the Calinski-Harabasz index
ch_index = calculate_ch_index(X, labels)
print(f"Calinski–Harabasz Index: {ch_index}")




