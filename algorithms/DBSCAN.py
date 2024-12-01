import os
import pandas as pd
from DBSCAN_auxiliar_functions import *

# Load dataset
dataset_path = 'datasets/Dataset6.csv'
X = pd.read_csv(dataset_path, sep=';', header=None).values

# Define experiment parameters
results_folder = "algorithms/Results_DBSCAN"
os.makedirs(results_folder, exist_ok=True)
eps = 0.3
min_samples = 10

# Save DBSCAN results
save_dbscan_final_plot(X, eps, min_samples, results_folder, experiment_type="DBSCAN")
