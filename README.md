# Randomization-for-Clustering

This repository provides implementations of various clustering algorithms. The primary objective is to apply these 
algorithms to different datasets and analyze the resulting cluster configurations. Additionally, we define internal 
and external metrics to evaluate and compare the performance of the methods. We also include techniques to determine 
the optimal number of clusters for each method. To enhance understanding, the repository offers several plots that 
facilitate a more intuitive interpretation of the algorithms and their results.

The datasets folder contains custom datasets used across the different methods. Specifically, we include two distinct 
datasets to analyze different aspects of clustering performance.

The algorithms folder contains all the important code related to the different clustering algorithms studied. In order 
to run the algorithms, we need to execute the Lloyd.py, KMeans.py, DBSCAN.py, and GMM.py files. Once they are executed, 
the corresponding results for these methods will appear in the respective Results folder.

Apart from the above main files, there are other scripts ready to run. The KMeans_elbow_method.py file performs the 
elbow method for KMeans. The algorithm_comparison.py file compares the KMeans, DBSCAN, and Gaussian Mixture algorithms 
on three different datasets and saves the results in the algorithm_comparison.png file. Finally, the 
Rand_Score_Comparison.py file calculates the Rand Score Indexes for the KMeans and Gaussian Mixture Models based on a 
ground-truth dataset containing the correct labels for the data points.
