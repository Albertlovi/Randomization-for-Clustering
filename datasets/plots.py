import matplotlib.pyplot as plt
import pandas as pd


def plot_df(df: pd.DataFrame, sample_factor: float = 1):
    '''
    This funciton plots a given dataset of points in the plane.
    '''
    df = df.sample(int(len(df) * sample_factor))
    if df.shape[1] == 2:
        plt.figure(figsize=(8, 6))  
        plt.scatter(df[0], df[1], color='blue', alpha=0.7)

        plt.show()
    else:
        print("The dimension of the dataset is not 2, thus it can't be ploted in the plane.")


df = pd.read_csv('datasets/Dataset6.csv', sep=';', header=None)

plot_df(df, 0.01)