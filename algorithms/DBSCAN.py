import os
import pandas as pd
from DBSCAN_auxiliar_functions import *
from sklearn.metrics import calinski_harabasz_score, silhouette_score

dataset_path = 'datasets/Dataset6.csv'
X = pd.read_csv(dataset_path, sep=';', header=None).values

# Define experiment parameters
results_folder = "algorithms/Results_DBSCAN"
os.makedirs(results_folder, exist_ok=True)
eps = 0.3
min_samples = 10

# Save DBSCAN results
save_dbscan_final_plot(X, eps, min_samples, results_folder, experiment_type="DBSCAN")

scaler = StandardScaler()
data_scaled = scaler.fit_transform(X)

db = DBSCAN(eps=eps, min_samples=min_samples)
labels = db.fit_predict(data_scaled)

# INTERNAL SCORES

print('The Calisnski-Harabasz score of DBSCAN is ', calinski_harabasz_score(data_scaled, labels)) # C-H index

print('The Silhouette score of DBSCAN is', silhouette_score(data_scaled, labels)) # Silhouette score
