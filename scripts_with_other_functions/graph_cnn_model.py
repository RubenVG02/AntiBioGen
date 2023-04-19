from check_affinity import calculate_affinity
import matplotlib.pyplot as plt

import pandas as pd
import tensorflow as tf
import numpy as np


def create_graph(path_data_csv=r""):
    '''
    Function to create a graph with the predictions of the model and the real values of the data set

    Parameters:
        -path_data_csv: Path of the csv file with the data set
    
    '''
    real_data = pd.read_csv(
        path_data_csv, sep=",", header=0, names=["smiles", "sequence", "IC50"])

    
    fasta = np.array([fasta for fasta in real_data["sequence"]], dtype="S")
    ic50 = np.array([ic50 for ic50 in real_data["IC50"]], dtype="f")
    predicts = []
    for i in range(50):
        prediction = calculate_affinity(smile="CCN(CCO)CC(=O)N1CC[C@@H](C(=O)N[C@H]2C[C@@H](C)O[C@@H](C)C2)CC1", fasta=fasta[i])
        predicts.append(prediction)

    plt.scatter(predicts, ic50[0:50])
    plt.xlabel("Predictions")
    plt.ylabel("Real values")
    plt.show()


create_graph()
