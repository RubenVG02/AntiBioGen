from arreglar_dades import neteja_dades_afinitat
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

dades = neteja_dades_afinitat()

arx = pd.read_csv(dades, sep=",")


# valor maxim que vull que tingin les meves smiles, serviran per entrenar el model
maxim_smiles = 140
elements_smiles = ['6', '3', '=', 'H', 'C', 'O', 'c', '#', 'a', '[', 't', 'r', 'K', 'n', 'B', 'F', '4', '+', ']', '-', '1', 'P',
                   '0', 'L', '%', 'g', '9', 'Z', '(', 'N', '8', 'I', '7', '5', 'l', ')', 'A', 'e', 'o', 'V', 's', 'S', '2', 'M', 'T', 'u', 'i']
# elements_smiles fa referencia a els elements pels quals es poden formar els smiles

int_smiles = dict(zip(elements_smiles, range(1, len(elements_smiles)+1)))
# Per associar tots els elements amb un int determinat(range és 1, len+1 perquè es plenen amb zeros per arribar a maxim_smiles)

maxim_fasta = 5000
elements_fasta = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
                  'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']  # Format pels diferents aa que formen els fasta

int_fasta = dict(zip(elements_fasta, range(1, len(elements_fasta))))
# Per associar tots els elements amb un int determinat(range és 1, len+1 perquè es plenen amb zeros per arribar a maxim_fasta)


def convertir(arx=arx):
    '''
        Funció per convertir tots els elements (tant smiles, com fasta) en int, per tal de ser entrenats al model

    '''

    smiles_amb_numeros = []  # Smiles obtinguts amb int_smiles[1] i els smiles del df
    for i in arx.smiles:
        for elements in i:  # Elements fa referència a els elements que formen elements_smile
            smiles_amb_numeros.append(int_smiles[elements])
        while len(i) < maxim_smiles:
            smiles_amb_numeros.append(0)

    fasta_amb_numeros = []
    for i in arx.sequence:
        for elements in i:  # Elements fa referència a els elements que formen elements_smile
            fasta_amb_numeros.append(int_fasta[elements])
        while len(i) < maxim_smiles:
            fasta_amb_numeros.append(0)

    ic50_numeros = list(arx.IC50)

    return smiles_amb_numeros, fasta_amb_numeros, ic50_numeros


def model_cnn():
    ''' 
        Model per entrenar les dades.

    '''
    # model per smiles
    i = tf.keras.Input(shape=(maxim_smiles,))
    embed = tf.keras.layers.Embedding(input_dim=len(
        elements_smiles)+1, input_length=maxim_smiles, output_dim=128)(i)
    x = tf.keras.layers.Conv1D(
        filters=32, kernel_size=3, padding="SAME")(embed)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="SAME")(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv1D(
        filters=128, kernel_size=3, padding="SAME")(x)
    x = tf.keras.layers.PReLU()(x)
    pool = tf.keras.layers.GlobalMaxPooling1D()(
        x)  # maxpool per obtenir un vector de 1d

    # model per fastas
    i2 = tf.keras.Input(shape=(maxim_smiles,))
    embed2 = tf.keras.layers.Embedding(input_dim=len(
        elements_fasta)+1, input_length=maxim_fasta, output_dim=256)(i2)
    x2 = tf.keras.layers.Conv1D(
        filters=32, kernel_size=3, padding="SAME")(embed2)
    x2 = tf.keras.layers.PReLU()(x2)
    x2 = tf.keras.layers.Conv1D(
        filters=64, kernel_size=3, padding="SAME")(x2)
    x2 = tf.keras.layers.PReLU()(x2)
    x2 = tf.keras.layers.Conv1D(
        filters=128, kernel_size=3, padding="SAME")(x2)
    x2 = tf.keras.layers.PReLU()(x2)
    pool2 = tf.keras.layers.GlobalMaxPooling1D()(
        x2)  # maxpool per obtenir un vector de 1d

    junt = tf.keras.layers.concatenate(inputs=[pool, pool2])

    # dense

    de = tf.keras.layers.Dense(units=512, activation="relu",)(junt)
    dr = tf.keras.layers.Dropout(0.2)(de)
    de2 = tf.keras.layers.Dense(units=256, activation="relu")(dr)

    # output

    op = tf.keras.layers.Dense(1, activation="relu")(de2)

    modelo = tf.keras.models.Model(inputs=[i, i2], outputs=[op])

    modelo.compile(optimizer="adam", loss="mse",
                   metrics=tf.keras.metrics.MeanSquaredError())

    tamany_per_epoch = 5000  # Utilitzem un valor elevat per poder obtenir millors resultats
    # utilizarem el 80/20 per entrenar y fer test al nostre model
    training = len(arx)*0.8

    train = arx[:int(training)]
    test = arx[int(training):]
    final = tamany_per_epoch
    for i in range(20):  # utilitzarem 20 epochs
        while tamany_per_epoch < len(arx):
            X_smiles, X_fasta, T_IC50 = convertir(arx[0:final])
            X_smiles_test, X_fasta_test, T_IC50_test = convertir(test)

            r = modelo.fit({X_smiles, X_fasta}, {"output": np.array(T_IC50)},
                           validation_data=(X_smiles_test, X_fasta_test), epochs=1, batch_size=64)

        final += tamany_per_epoch

    plt.plot(r.history["loss"], label="loss")
    plt.plot(r.history["val_loss"], label="val_loss")


model_cnn()
