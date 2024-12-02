import matplotlib.pyplot as plt
import os
from sklearn.mixture import GaussianMixture
import pandas as pd
from sklearn import mixture
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from auxiliary_functions_to_plot import save_gmm

X = pd.read_csv('datasets/Dataset6.csv', sep=';', header=None).values

scores = []
n_components_range = range(2, 11)  # Try clusters from 2 to 10

for n_components in n_components_range:
    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type="full", random_state=42)
    gmm.fit(X)
    labels = gmm.predict(X)
    score = calinski_harabasz_score(X, labels)
    scores.append(score)

# Plot the scores
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(n_components_range, scores, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Calinski–Harabasz Score")
plt.title("Calinski–Harabasz Index for GMM")

results_folder = "algorithms/Results_GMM"
os.makedirs(results_folder, exist_ok=True)

# Save the plot in the specified folder
final_plot_filename = os.path.join(results_folder, "calinski_harabasz_scores_gmm.png")
plt.savefig(final_plot_filename)

# Close the plot to release memory
plt.close()

# Identify the optimal number of clusters
optimal_k = n_components_range[scores.index(max(scores))]
print(f"Optimal number of clusters based on Calinski–Harabasz Index: {optimal_k}")

experiment_type = "GMM"

save_gmm(X, optimal_k, results_folder, experiment_type)

gmm = GaussianMixture(n_components=n_components)
labels = gmm.fit_predict(X)

# INTERNAL SCORES

print('The Calisnski-Harabasz score of GMM is ', calinski_harabasz_score(X, labels)) # C-H index

print('The Silhouette score of GMM is', silhouette_score(X, labels)) # Silhouette score






