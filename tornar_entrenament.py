import tensorflow as tf
import pandas as pd
from rdkit import DataStructs, Chem
import numpy as np
from keras.layers import CuDNNLSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


dades = open(r"C:\Users\ASUS\Desktop\github22\smalls_prueba.txt").read()

# per obtenir els elements unics de dades a numeros enters mitjançant un diccionari
# així associem un valor numeric a cada lletra
elements_smiles = {u: i for i, u in enumerate(sorted(set(dades)))}
elements_smiles.update({-1: "\n"})

# per passar els elements numerics a elements dels smiles
int_a_elements = {i: u for i, u in enumerate(sorted(set(dades)))}
int_a_elements.update({"\n": -1})

mapa_int = len(elements_smiles)+3
mapa_char = len(int_a_elements)+3


def split_input_target(chunk, valors=mapa_char-1):
    input_text = chunk[:-1]
    target_idx = chunk[-1]
    target = tf.one_hot(target_idx, depth=valors)
    target = tf.reshape(target, [-1])
    return input_text, target


max_smile = 137

slices = np.array([[elements_smiles[c]] for c in dades])

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(slices)

sequences = char_dataset.batch(137+1, drop_remainder=True)

dataset = sequences.map(split_input_target)

dataset = dataset.shuffle(20000).batch(128, drop_remainder=True)

modelo = tf.keras.models.Sequential([CuDNNLSTM(128, input_shape=(137, 1), return_sequences=True),
                                     Dropout(0.15),
                                     CuDNNLSTM(256, return_sequences=True),
                                     BatchNormalization(),
                                     Dropout(0.15),
                                     CuDNNLSTM(512, return_sequences=True),
                                     BatchNormalization(),
                                     Dropout(0.15),
                                     CuDNNLSTM(256, return_sequences=True),
                                     BatchNormalization(),
                                     Dropout(0.15),
                                     CuDNNLSTM(128),
                                     Dropout(0.15),
                                     Dense(mapa_char-1, activation="softmax")])


modelo.load_weights(
    r"C:\Users\ASUS\Desktop\github22\modelo_rnn_smalls.hdf5")

modelo.compile(optimizer="adam",
               loss="categorical_crossentropy", metrics=["accuracy"])

filepath = "modelo_rnn_smalls.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True, mode='min')
callbacks_list = [checkpoint]

r = modelo.fit(dataset, epochs=150, callbacks=callbacks_list, batch_size=128)
