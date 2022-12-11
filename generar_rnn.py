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


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_idx = chunk[-1]
    target = tf.one_hot(target_idx, depth=34)
    target = tf.reshape(target, [-1])
    return input_text, target


dades = open(r"C:\Users\ASUS\Desktop\github22\dasdsd\smiles.txt").read()

# per obtenir els elements unics de dades
elements_smiles = {u: i for i, u in enumerate(sorted(set(dades)))}
elements_smiles.update({-1: "\n"})

# per passar els elements unics de dades a
int_a_elements = {i: u for i, u in enumerate(sorted(set(dades)))}
elements_smiles.update({-1: "\n"})

mapa_int = len(elements_smiles)
mapa_char = len(int_a_elements)

max_smile = 137

slices = np.array([[elements_smiles[c]] for c in dades])

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(slices)

sequences = char_dataset.batch(137+1, drop_remainder=True)

dataset = sequences.map(split_input_target)

dataset = dataset.shuffle(10000).batch(256, drop_remainder=True)

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

modelo.compile(optimizer="adam",
               loss="categorical_crossentropy", metrics=["accuracy"])

filepath = "model_rnn_3_128.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True, mode='min')
callbacks_list = [checkpoint]

r = modelo.fit(dataset, epochs=200, callbacks=callbacks_list)

plt.plot(r.history["accuracy"], label="accuracy")
plt.legend()
plt.show()
