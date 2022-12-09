import tensorflow as tf
import pandas as pd
from rdkit import DataStructs, Chem
import numpy as np
from keras.layers import CuDNNLSTM
from keras.callbacks import ModelCheckpoint


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_idx = chunk[-1]
    target = tf.one_hot(target_idx, depth=34)
    target = tf.reshape(target, [-1])
    return input_text, target


dades = open("smiles.txt").read()

elements_smiles = {valor: lletra for lletra, valor in enumerate(set(dades))}

max_smile = 137

slices = np.array([[elements_smiles[i] for i in dades]])

ds = tf.data.Dataset.from_tensor_slices(slices).batch(
    max_smile+1, drop_remainder=True).map(split_input_target)

ds = ds.shuffle(10000).batch(256, drop_remainder=True)


def model():
    modelo = tf.keras.models.Sequential()
    modelo.add(CuDNNLSTM(units=128, return_sequences=True, input_shape=(137, 1)))
    modelo.add(tf.keras.layers.Dropout(0.1))
    modelo.add(CuDNNLSTM(units=256, return_sequences=True))
    modelo.add(tf.keras.layers.Dropout(0.1))
    modelo.add(CuDNNLSTM(units=256, return_sequences=True))
    modelo.add(tf.keras.layers.Dropout(0.1))
    modelo.add(CuDNNLSTM(units=128, return_sequences=True))
    modelo.add(tf.keras.layers.Dropout(0.1))
    modelo.add(tf.keras.layers.Dense(34, activation="softmax"))

    modelo.compile(optimizer="adam",
                   loss="categorical_crossentropy", metrics=["accuracy"])

    save_model_path = "model_rnn.hdf5"
    checkpoint = ModelCheckpoint(save_model_path,
                                 monitor='accuracy',
                                 verbose=1,
                                 save_best_only=True)

    modelo.fit(ds, epochs=200, callbacks=[checkpoint])


model()
