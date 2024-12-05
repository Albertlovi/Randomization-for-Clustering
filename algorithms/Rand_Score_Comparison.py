from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import pandas as pd

# Load the dataset
Y = pd.read_csv('datasets/Dataset1.csv', sep=';', header=None)

# Remove commas from the dataset if there are any (for example, '610,291' -> 610291)
Y = Y.applymap(lambda x: str(x).replace(',', '') if isinstance(x, str) else x)

# Convert all columns to numeric
Y = Y.apply(pd.to_numeric, errors='coerce')

# Assuming the ground truth labels are the last column of the dataset (adjust if necessary)
ground_truth = Y.iloc[:, -1].values  # Modify this if the ground truth is stored differently

# KMEANS
# Fit KMeans with 7 clusters
kmeans = KMeans(n_clusters=7, random_state=42)
labels_kmeans = kmeans.fit_predict(Y.iloc[:, :-1])  # Use all features except the last column

# Calculate the Adjusted Rand Index between the ground truth and KMeans labels
ari_kmeans = adjusted_rand_score(ground_truth, labels_kmeans)

# Print the Adjusted Rand Index
print(f"The Adjusted Rand Index between ground truth and KMeans is: {ari_kmeans}")


# GMM
# Fit GMM with 7 clusters
gmm = GaussianMixture(n_components=7, random_state=42)
labels_gmm = gmm.fit_predict(Y.iloc[:, :-1])


# Calculate the Adjusted Rand Index between the ground truth and GMM labels
ari_gmm = adjusted_rand_score(ground_truth, labels_gmm)

# Print the Adjusted Rand Index
print(f"The Adjusted Rand Index between ground truth and GMM is: {ari_gmm}")
