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


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_idx = chunk[-1]
    target = tf.one_hot(target_idx, depth=51)
    target = tf.reshape(target, [-1])
    return input_text, target


with open(r"C:\Users\ASUS\Desktop\github22\dasdsd\xab.txt") as f:
    dades = "\n".join(line.strip() for line in f)


# per obtenir els elements unics de dades
elements_smiles = {u: i for i, u in enumerate(sorted(set(dades)))}
elements_smiles.update({-1: "\n"})


# per passar els elements unics de dades a
int_a_elements = dict((i, c) for i, c in enumerate(elements_smiles))
int_a_elements.update({"\n": -1})

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
                                         Dense(mapa_char-1, activation="softmax")])
    return modelo


modelo = crear_model()
modelo.load_weights(
    r"C:\Users\ASUS\Desktop\github22\dasdsd\modelo_prueba_rnn.hdf5")
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
for i in range(100):
    for i in range(random.randrange(50, 137)):
        x = np.reshape(pattern, (1, len(pattern), 1))
        prediction = modelo.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_a_elements[index]
        seq_in = [int_a_elements[value] for value in pattern]
        sys.stdout.write(result)
        final += result
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("\nDone.")
    mol1 = Chem.MolFromSmiles(final)
    print(mol1)
    if mol1 == None:
        print("error")
    elif not mol1 == None:
        print(result)
        print("Ha salido una buena")
        Draw.MolToImageFile(
            mol1, filename=f"moleculas_generadas/molecula{int(time.time())}.jpg", size=(400, 300))
        with open(r"C:\Users\ASUS\Desktop\github22\dasdsd\moleculas_generadas/moleculas.txt", "a") as file:
            file.write(final.replace("\n", "") + "\n" + "\n")
    final = ""


'''for i in range(100):


    network_input = tfds.as_numpy(
        dataset.take(np.random.randint(0, 30000)))
    for i, x in enumerate(network_input):
        break
    llista_dades = dades.split("\n")
    seed = x[0][np.random.randint(0, 127)]
    molecula = ""
    for i in range(137):
        # predict et dona un float, amb max obtenim la millor predicció
        predicció = modelo.predict(np.reshape(
            seed, (1, len(seed), 1)), verbose=0)
        index = np.argmax(predicció)
        resultat = int_a_elements[index]
        molecula += resultat
        seed = np.append(seed, np.array([[index]]), axis=0)
        seed = seed[1:len(seed)]
    mol1 = Chem.MolFromSmiles(molecula)
    print(mol1)
    if mol1 == None:
        print("error")
    else:
        print(molecula)
        print("Ha salido una buena")
        Draw.MolToImageFile(mol1, filename="molecula22.jpg",
                            size=(400, 300))
        break
'''
