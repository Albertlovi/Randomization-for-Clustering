from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import pandas as pd

# Load the dataset
Y = pd.read_csv('datasets/Dataset1.csv', sep=';', header=None)

Y = Y.applymap(lambda x: str(x).replace(',', '') if isinstance(x, str) else x)
Y = Y.apply(pd.to_numeric, errors='coerce')

ground_truth = Y.iloc[:, -1].values

# KMEANS
# Fit KMeans with 7 clusters
kmeans = KMeans(n_clusters=7, random_state=42)
labels_kmeans = kmeans.fit_predict(Y.iloc[:, :-1])  # Use all features except the last column

# Calculate the Adjusted Rand Index between the ground truth and KMeans labels
ari_kmeans = adjusted_rand_score(ground_truth, labels_kmeans)

print(f"The Adjusted Rand Index between ground truth and KMeans is: {ari_kmeans}")

# GMM
# Fit GMM with 7 clusters
gmm = GaussianMixture(n_components=7, random_state=42)
labels_gmm = gmm.fit_predict(Y.iloc[:, :-1])

# Calculate the Adjusted Rand Index between the ground truth and GMM labels
ari_gmm = adjusted_rand_score(ground_truth, labels_gmm)

print(f"The Adjusted Rand Index between ground truth and GMM is: {ari_gmm}")
