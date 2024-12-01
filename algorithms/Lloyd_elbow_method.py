import pandas as pd 
from sklearn import datasets
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# Load the IRIS dataset
iris = datasets.load_iris()
df = pd.read_csv('datasets/Dataset6.csv', sep=';', header=None).values 

# Instantiate the clustering model and visualizer
km = KMeans(random_state=42)
maximum_k = 10
visualizer = KElbowVisualizer(km, k=(2, maximum_k))
 
visualizer.fit(df)    
visualizer.show()
    