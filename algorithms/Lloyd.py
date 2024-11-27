import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from algorithms.Lloyd_auxiliary_functions import save_experiment_results


# Load the dataset
df = pd.read_csv('datasets/Dataset6.csv', sep=';', header=None).values  
k = 3 # Number of clusters
experiments = {} # Dictionary to store experiment results

# Perform T experiments
T = 100
for i in range(T):
    # Generate k random initial centers 
    initial_centers = df[np.random.choice(len(df), k, replace=False)]
    
    # Initialize and fit KMeans with these initial centers
    kmeans = KMeans(n_clusters=3, init=initial_centers, n_init=1, max_iter=100, random_state=None)
    kmeans.fit(df)
    
    # Calculate Calinski-Harabasz score
    labels = kmeans.labels_
    ch_index = calinski_harabasz_score(df, labels)
    
    # Save the results in the experiments dictionary
    experiments[i] = {
        "initial_centers": initial_centers,
        "ch_index": ch_index
    }

# Find min, max, and median experiments
all_ch_indices = [(i, exp["ch_index"]) for i, exp in experiments.items()]
all_ch_indices.sort(key=lambda x: x[1])  
min_experiment = experiments[all_ch_indices[0][0]]
max_experiment = experiments[all_ch_indices[-1][0]]
median_index = len(all_ch_indices) // 2
median_experiment = experiments[all_ch_indices[median_index][0]]

# Results folder
results_folder = "algorithms/Lloyd_results"
os.makedirs(results_folder, exist_ok=True)

# Save results for each type of experiment
save_experiment_results(df, min_experiment, results_folder, "Worst experiment")
save_experiment_results(df, max_experiment, results_folder, "Best experiment")
save_experiment_results(df, median_experiment, results_folder, "Median experiment")

