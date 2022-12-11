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
import numpy as np


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_idx = chunk[-1]
    target = tf.one_hot(target_idx, depth=34)
    target = tf.reshape(target, [-1])
    return input_text, target


dades = open(r"C:\Users\ASUS\Desktop\github22\dasdsd\smiles.txt").read()

# per obtenir els elements unics de dades
elements_smiles = {u: i for i, u in enumerate(sorted(set(dades)))}


# per passar els elements unics de dades a
int_a_elements = dict((i, c) for i, c in enumerate(elements_smiles))


mapa_int = len(elements_smiles)
mapa_char = len(int_a_elements)

max_smile = 137


slices = np.array([[elements_smiles[c]] for c in dades])

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(slices)

sequences = char_dataset.batch(137+1, drop_remainder=True)

dataset = sequences.map(split_input_target)

dataset = dataset.shuffle(10000).batch(256, drop_remainder=True)


def crear_model():
    modelo = tf.keras.models.Sequential([CuDNNLSTM(128, input_shape=(137, 1), return_sequences=True),
                                         Dropout(0.1),
                                         CuDNNLSTM(256, return_sequences=True),
                                         BatchNormalization(),
                                         Dropout(0.1),
                                         CuDNNLSTM(512, return_sequences=True),
                                         BatchNormalization(),
                                         Dropout(0.1),
                                         CuDNNLSTM(256, return_sequences=True),
                                         BatchNormalization(),
                                         Dropout(0.1),
                                         CuDNNLSTM(128),
                                         Dropout(0.1),
                                         Dense(34, activation="softmax")])
    return modelo


modelo = crear_model()
modelo.load_weights(r"C:\Users\ASUS\Desktop\github22\dasdsd\model_rnn_2.hdf5")


### Generació de molecules###
for i in range(100):

    network_input = tfds.as_numpy(
        dataset.take(np.random.randint(0, 30000)))
    for i, x in enumerate(network_input):
        break
    llista_dades = dades.split("\n")
    seed = x[0][np.random.randint(0, 127)]
    molecula = ""
    for i in range(137):
        # predict et dona un float, amb max obtenim la millor predicció
        predicció = modelo.predict(np.reshape(seed, (1, len(seed), 1)))
        index = np.argmax(predicció)
        resultat = int_a_elements[index]
        molecula += resultat
        seed = np.append(seed, np.array([[index]]), axis=0)
        seed = seed[1:len(seed)]
    mol1 = Chem.MolFromSmiles(molecula)
    if mol1 == None:
        print("error")
    else:
        print(molecula)
        print("Ha salido una buena")
        break
