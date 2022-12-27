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

from rdkit.Chem import Draw


def generador(path_model=r"C:\Users\ASUS\Desktop\github22\modelo_rnn_smalls.hdf5", path_dades=r"C:\Users\ASUS\Desktop\github22\dasdsd\xab.txt",
              nombre_generats=100, img_druglike=True, path_desti_molecules=r"C:\Users\ASUS\Desktop\github22\dasdsd\moleculas_generadas//moleculas_nuevo_generador/moleculas_druglike.txt"):
    '''
        Paràmetres:
        -path_model: Path on es troba el model ja entrenat
        -path_dades: Path on es troben les dades utlitzades (path model i path dades han de tenir les mateixes dimensions/mateixa quantitat d'elements diferents)
        -nombre_generats: Nombre de molècules que es generen, per predeterminat 100
        -img_druglike: Per generar imatges .jpg de les molècules drug-like generades (estaran ordenades en funció de epoch) (Per default True)
        -Path_desti_molecules: Path de destí de les seqüencies SMILE generades
    '''
    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_idx = chunk[-1]
        target = tf.one_hot(target_idx, depth=48)
        target = tf.reshape(target, [-1])
        return input_text, target

    with open(f"{path_dades}") as f:
        dades = "\n".join(line.strip() for line in f)

    # per obtenir els elements unics de dades
    elements_smiles = {u: i for i, u in enumerate(sorted(set(dades)))}
    elements_smiles.update({-1: "\n"})

    # per passar els elements unics de dades a
    int_a_elements = {i: c for i, c in enumerate(elements_smiles)}
    int_a_elements.update({"\n": -1})

    mapa_int = len(elements_smiles)
    mapa_char = len(int_a_elements)

    max_smile = 137

    slices = np.array([[elements_smiles[c]] for c in dades])

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(slices)

    sequences = char_dataset.batch(137+1, drop_remainder=True)

    dataset = sequences.map(split_input_target)

    dataset = dataset.shuffle(20000).batch(256, drop_remainder=True)

    def crear_model():
        modelo = tf.keras.models.Sequential([CuDNNLSTM(128, input_shape=(137, 1), return_sequences=True),
                                            Dropout(0.1),
                                            CuDNNLSTM(
                                                256, return_sequences=True),
                                            BatchNormalization(),
                                            Dropout(0.1),
                                            CuDNNLSTM(
                                                512, return_sequences=True),
                                            BatchNormalization(),
                                            Dropout(0.1),
                                            CuDNNLSTM(
                                                256, return_sequences=True),
                                            BatchNormalization(),
                                            Dropout(0.1),
                                            CuDNNLSTM(128),
                                            Dropout(0.1),
                                            Dense(mapa_char+2, activation="softmax")])
        return modelo

    modelo = crear_model()
    modelo.load_weights(f"{path_model}")
    modelo.compile(loss='categorical_crossentropy', optimizer='adam')

    ### Generació de molecules###
    seq_length = 137
    dataX = []
    dataY = []
    for i in range(0, len(dades) - seq_length, 1):
        seq_in = dades[i:i + seq_length]
        seq_out = dades[i + seq_length]
        dataX.append([elements_smiles[char] for char in seq_in])
        dataY.append(elements_smiles[seq_out])
    pattern = dataX[np.random.randint(0, len(dataX)-1)]
    print("\"", ''.join([int_a_elements[value] for value in pattern]), "\"")
    final = ""
    total_smiles = []
    for i in range(nombre_generats):
        for i in range(random.randrange(100, 137)):
            x = np.reshape(pattern, (1, len(pattern), 1))
            prediction = modelo.predict(x, verbose=0)
            index = np.argmax(prediction)
            result = int_a_elements[index]
            seq_in = [int_a_elements[value] for value in pattern]
            sys.stdout.write(result)
            final += result
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
        final = final.split("\n")
        for i in final:
            mol1 = Chem.MolFromSmiles(i)
            if len(i) > 10:
                if mol1 == None:
                    print("error")
                elif not mol1 == None:
                    print(result)
                    print("Ha salido una buena, miro si es drug-like")
                    if Descriptors.ExactMolWt(mol1) < 500 and Descriptors.MolLogP(mol1) < 5 and Descriptors.NumHDonors(mol1) < 5 and Descriptors.NumHAcceptors(mol1) < 10:
                        if img_druglike == True:
                            Draw.MolToImageFile(
                                mol1, filename=fr"C:\Users\ASUS\Desktop\github22\dasdsd\moleculas_generadas\moleculas_nuevo_generador/molecula{int(time.time())}.jpg", size=(400, 300))
                        with open(f"{path_desti_molecules}", "a") as file:
                            with open(f"{path_desti_molecules}", "r") as f:
                                linies = [linea.rstrip() for linea in f]
                            if f"{i}" not in linies:
                                file.write(i + "\n")
                        total_smiles.append(i)
                        print("La molecula es drug-like")
            else:
                pass
        final = ""
    return total_smiles

generador()