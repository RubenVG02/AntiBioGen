from pretrained_rnn import generador
from check_affinitat import mesurador_afinitat

import tensorflow as tf
import pandas as pd
from keras.models import load_model
from keras.layers import CuDNNLSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
import tensorflow_datasets as tfds
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import numpy as np
import sys
import random
import time
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, Input, PReLU, Dropout, concatenate, BatchNormalization
import csv
from mega import Mega


target = "MAQTQGTKRKVCYYYDGDVGNYYYGQGHPMKPHRIRMTHNLLLNYGLYRKMEIYRPHKANAEEMTKYHSDDYIKFLRSIRPDNMSEYSKQMQRFNVGEDCPVFDGLFEFCQLSTGGSVASAVKLNKQQTDIAVNWAGGLHHAKKSEASGFCYVNDIVLAILELLKYHQRVLYIDIDIHHGDGVEEAFYTTDRVMTVSFHKYGEYFPGTGDLRDIGAGKGKYYAVNYPLRDGIDDESYEAIFKPVMSKVMEMFQPSAVVLQCGSDSLSGDRLGCFNLTIKGHAKCVEFVKSFNLPMLMLGGGGYTIRNVARCWTYETAVALDTEIPNELPYNDYFEYFGPDFKLHISPSNMTNQNTNEYLEKIKQRLFENLRMLPHAPGVQMQAIPEDAIPEESGDEDEEDPDKRISICSSDKRIACEEEFSDSDEEGEGGRKNSSNFKKAKRVKTEDEKEKDPEEKKEVTEEEKTKEEKPEAKGVKEEVKLA"


def crear_arxiu(nom_arxiu, headers=["smiles", "IC50"]):
    with open(f"{nom_arxiu}.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(headers)


def buscar_candidats(target=target, forma_guardat="csv", nom_arx="prueba_tio", pujar_a_mega=True):
    ic50 = []
    smiles = []
    ic50_menor = 100000
    mirar = []
    crear_arxiu(nom_arxiu=nom_arx)
    while not ic50_menor < 100:
        generats = generador(nombre_generats=10, img_druglike=False)
        smiles.extend(generats)
        for i in generats:
            i = i.replace("@", "").replace("/", "")
            try:
                predicció_ic50 = mesurador_afinitat(smile=i, fasta=target)
                ic50.append(predicció_ic50)
                mirar.append(predicció_ic50)
            except:
                ic50.append("error")
        ic50_menor = int(min(mirar))

        combinacio = list(zip(smiles, ic50))
        linies = open(f"{nom_arx}.csv", "r").read()
        with open(f"{nom_arx}.csv", "a", newline="") as file:
            for i in combinacio:
                if str(i[1]) not in linies:
                    file.write(f"{i[0]},{i[1]}\n")
    '''if pujar_a_mega == True:
        compte = Mega.login_anonymous()
        pujada = compte.upload(f"{nom_arx}.csv")
        compte.get_upload_like(pujada)'''


buscar_candidats()
