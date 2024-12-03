import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from algorithms.KMeans_auxiliary_functions import save_experiment_results

# Load the dataset
df = pd.read_csv('datasets/Dataset6.csv', sep=';', header=None).values  
k = 5 # Number of clusters

# Variables to track the best experiment
best_ch_index = -np.inf  # Start with a very low score
best_initial_centers = None

# Perform T experiments
T = 100
for i in range(T):
    # Generate k random initial centers
    initial_centers = df[np.random.choice(len(df), k, replace=False)]
    
    # Initialize and fit K-Means with these initial centers
    kmeans = KMeans(n_clusters=k, init=initial_centers, n_init=1, max_iter=100, random_state=None)
    kmeans.fit(df)
    
    # Calculate Calinski-Harabasz score
    labels = kmeans.labels_
    ch_index = calinski_harabasz_score(df, labels)
    
    # If the current experiment has a better score, update the best experiment
    if ch_index > best_ch_index:
        best_ch_index = ch_index
        best_initial_centers = initial_centers

# Save results for the best experiment
results_folder = "algorithms/Results_Lloyd"
os.makedirs(results_folder, exist_ok=True)

# Save the best experiment results
save_experiment_results(df, {"initial_centers": best_initial_centers, "ch_index": best_ch_index}, results_folder, "Best experiment")



