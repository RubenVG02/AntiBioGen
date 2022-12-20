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


target = "PEEIRPKEVYLDRKLLTLEDKELGSGNFGTVKKGYYQMKKVVKTVAVKILKNEANDPALKDELLAEANVMQQLDNPYIVRMIGICEAESWMLVMEMAELGPLNKYLQQNRHVKDKNIIELVHQVSMGMKYLEESNFVHRDLAARNVLLVTQHYAKISDFGLSKALRADENYYKAQTHGKWPVKWYAPECINYYKFSSKSDVWSFGVLMWEAFSYGQKPYRGMKGSEVTAMLEKGERMGCPAGCPREMYDLMNLCWTYDVENRPGFAAVELRLRNYYYDVVN"


def buscar_candidats(target=target, forma_guardat="csv", nom_arx="resultats_pibe", pujar_a_mega=True):
    ic50 = []
    smiles = []
    ic50_menor = 100000
    mirar = []
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
        headers = ["smiles", "IC50"]
        ic50_menor = int(min(mirar))
        with open(f"{nom_arx}.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerows(zip(smiles, ic50))
    '''if pujar_a_mega == True:
        compte = Mega.login_anonymous()
        pujada = compte.upload(f"{nom_arx}.csv")
        compte.get_upload_like(pujada)'''


buscar_candidats()
