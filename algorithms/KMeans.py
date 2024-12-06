import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.cluster import KMeans
from auxiliary_functions_to_plot import save_kmeans_clustering_final_plot
from sklearn.metrics import calinski_harabasz_score, silhouette_score

X = pd.read_csv('datasets/Dataset6.csv', sep=';', header=None).values

# Compute Calinski–Harabasz scores for different numbers of clusters
scores = []
n_clusters_range = range(2, 11)  

for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    score = calinski_harabasz_score(X, labels)
    scores.append(score)

plt.figure(figsize=(8, 4))
plt.plot(n_clusters_range, scores, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Calinski–Harabasz Score")
plt.title("Calinski–Harabasz Index for KMeans")

results_folder = "algorithms/Results_KMeans"
os.makedirs(results_folder, exist_ok=True)

final_plot_filename = os.path.join(results_folder, "calinski_harabasz_scores_kmeans.png")
plt.savefig(final_plot_filename)
plt.close()

# Identify the optimal number of clusters
optimal_k = n_clusters_range[scores.index(max(scores))]
print(f"Optimal number of clusters based on Calinski–Harabasz Index: {optimal_k}")


kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels_kmeans = kmeans.fit_predict(X)
centers_kmeans = kmeans.cluster_centers_

# Save clustering plot with KMeans
save_kmeans_clustering_final_plot(X, labels_kmeans, centers=centers_kmeans, results_folder=results_folder, experiment_type="KMeans")

# INTERNAL SCORES

print('The Calisnski-Harabasz score of KMeans is ', calinski_harabasz_score(X, labels_kmeans)) # C-H index

print('The Silhouette score of KMeans is', silhouette_score(X, labels_kmeans)) # Silhouette score

