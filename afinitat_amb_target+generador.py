import os
import sys
import numpy as np
from rdkit.Chem import Descriptors, Lipinski
from rdkit import Chem
import tensorflow_datasets as tfds
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import CuDNNLSTM
from keras.models import load_model
import pandas as pd
import tensorflow as tf
from check_affinitat import mesurador_afinitat
from pretrained_rnn import generador
import random
import time
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, Input, PReLU, Dropout, concatenate, BatchNormalization
import csv
from mega import Mega
import sascorer
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))


target = "MAQTQGTKRKVCYYYDGDVGNYYYGQGHPMKPHRIRMTHNLLLNYGLYRKMEIYRPHKANAEEMTKYHSDDYIKFLRSIRPDNMSEYSKQMQRFNVGEDCPVFDGLFEFCQLSTGGSVASAVKLNKQQTDIAVNWAGGLHHAKKSEASGFCYVNDIVLAILELLKYHQRVLYIDIDIHHGDGVEEAFYTTDRVMTVSFHKYGEYFPGTGDLRDIGAGKGKYYAVNYPLRDGIDDESYEAIFKPVMSKVMEMFQPSAVVLQCGSDSLSGDRLGCFNLTIKGHAKCVEFVKSFNLPMLMLGGGGYTIRNVARCWTYETAVALDTEIPNELPYNDYFEYFGPDFKLHISPSNMTNQNTNEYLEKIKQRLFENLRMLPHAPGVQMQAIPEDAIPEESGDEDEEDPDKRISICSSDKRIACEEEFSDSDEEGEGGRKNSSNFKKAKRVKTEDEKEKDPEEKKEVTEEEKTKEEKPEAKGVKEEVKLA"


def crear_arxiu(nom_arxiu, headers=["smiles", "IC50", "score"]):
    with open(f"{nom_arxiu}.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(headers)


def buscar_candidats(target=target, forma_guardat="csv", nom_arx="prueba_tio", pujar_a_mega=True):
    ic50 = []
    smiles = []
    ic50_menor = 100000
    mirar = []
    score = []
    crear_arxiu(nom_arxiu=nom_arx)
    while not ic50_menor < 100:
        generats = generador(nombre_generats=10, img_druglike=False)
        smiles.extend(generats)
        for i in generats:
            score_molecula = sascorer.calculateScore(i)
            score.append(score_molecula)
            i = i.replace("@", "").replace("/", "")
            try:
                predicció_ic50 = mesurador_afinitat(smile=i, fasta=target)
                ic50.append(predicció_ic50)
                mirar.append(predicció_ic50)
            except:
                ic50.append("error")
        ic50_menor = int(min(mirar))

        combinacio = list(zip(smiles, ic50, score))
        linies = open(f"{nom_arx}.csv", "r").read()
        with open(f"{nom_arx}.csv", "a", newline="") as file:
            for i in combinacio:
                if str(i[1]) not in linies:
                    file.write(f"{i[0]},{i[1]},{i[2]}\n")
    '''if pujar_a_mega == True:
        compte = Mega.login_anonymous()
        pujada = compte.upload(f"{nom_arx}.csv")
        compte.get_upload_like(pujada)'''


buscar_candidats()
