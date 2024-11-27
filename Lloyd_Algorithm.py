import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from auxiliary_functions_Lloyd import save_experiment_results

# Load the dataset
X = pd.read_csv('datasets/Dataset6.csv', sep=';', header=None).values  
k = 3

# Dictionary to store experiment results
experiments = {}

# Perform T experiments
T = 100
for i in range(T):
    # Generate k random centers 
    initial_centers = X[np.random.choice(len(X), k, replace=False)]
    
    # Initialize and fit KMeans with these initial centers
    kmeans = KMeans(n_clusters=3, init=initial_centers, n_init=1, max_iter=300, random_state=None)
    kmeans.fit(X)
    
    # Calculate CH Index
    labels = kmeans.labels_
    ch_index = calinski_harabasz_score(X, labels)
    
    # Save the results in the experiments dictionary
    experiments[i] = {
        "initial_centers": initial_centers,
        "ch_index": ch_index
    }

# Get all ch values
all_ch_indices = [(i, exp["ch_index"]) for i, exp in experiments.items()]
all_ch_indices.sort(key=lambda x: x[1])  # Sort by CH Index value

# Find min, max, and median experiments
all_ch_indices = [(i, exp["ch_index"]) for i, exp in experiments.items()]
all_ch_indices.sort(key=lambda x: x[1])  # Sort by CH Index value
min_experiment = experiments[all_ch_indices[0][0]]
max_experiment = experiments[all_ch_indices[-1][0]]
median_index = len(all_ch_indices) // 2
median_experiment = experiments[all_ch_indices[median_index][0]]

# Results folder
results_folder = "results_Lloyd"
os.makedirs(results_folder, exist_ok=True)

# Save results for each type of experiment
save_experiment_results(X, min_experiment, results_folder, "Min CH Index")
save_experiment_results(X, max_experiment, results_folder, "Max CH Index")
save_experiment_results(X, median_experiment, results_folder, "Median CH Index")

