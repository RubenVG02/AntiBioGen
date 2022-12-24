from check_affinitat import mesurador_afinitat
import matplotlib.pyplot as plt

import pandas as pd
import tensorflow as tf
import numpy as np


def crear_gràfiques(path_dades_bones=r"C:\Users\ASUS\Desktop\github22\dasdsd\CSV\500k_dades.csv"):

    dades_reals = pd.read_csv(
        r"C:\Users\ASUS\Desktop\github22\dasdsd\CSV\500k_dades.csv")

    smiles = [smile for smile in dades_reals["smiles"]]
    fasta = [fasta for fasta in dades_reals["sequence"]]
    ic50 = [ic50 for ic50 in dades_reals["IC50"]]
    predits = []
    for i in range(50):
        predicció = mesurador_afinitat(smile=smiles[i], fasta=fasta[i])
        predits.append(predicció)

    plt.scatter(predits, ic50[0:50])
    plt.xlabel("Valors Predits")
    plt.ylabel("Valors Reals")
    plt.show()


crear_gràfiques()
