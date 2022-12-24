import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.losses import mean_squared_error
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, Input, PReLU, Dropout, concatenate, BatchNormalization
from keras.regularizers import l2

# dades = neteja_dades_afinitat()


def mesurador_afinitat(path_csv=r"C:\Users\ASUS\Desktop\github22\dasdsd\CSV\500k_dades.csv", smile="", fasta="", path_model=r"C:\Users\ASUS\Desktop\github22\model_prueba_cnn.hdf5"):
    arx = pd.read_csv(f"{path_csv}", sep=",")

    # valor maxim que vull que tingin les meves smiles, serviran per entrenar el model
    maxim_smiles = 100
    elements_smiles = ['6', '3', '=', 'H', 'C', 'O', 'c', '#', 'a', '[', 't', 'r', 'K', 'n', 'B', 'F', '4', '+', ']', '-', '1', 'P',
                       '0', 'L', '%', 'g', '9', 'Z', '(', 'N', '8', 'I', '7', '5', 'l', ')', 'A', 'e', 'o', 'V', 's', 'S', '2', 'M', 'T', 'u', 'i']
    # elements_smiles fa referencia a els elements pels quals es poden formar els smiles

    int_smiles = dict(zip(elements_smiles, range(1, len(elements_smiles)+1)))
    # Per associar tots els elements amb un int determinat

    maxim_fasta = 5000
    elements_fasta = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
                      'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']  # Format pels diferents aa que formen els fasta

    int_fasta = dict(zip(elements_fasta, range(1, len(elements_fasta)+1)))
    # Per associar tots els elements amb un int determinat(range és 1, len+1 perquè es plenen amb zeros per arribar a maxim_fasta)

    # regulador kernel
    regulador = l2(0.001)

    # model per a smiles
    smiles_input = tf.keras.Input(
        shape=(maxim_smiles,), dtype='int32', name='smiles_input')
    embed = Embedding(input_dim=len(
        elements_smiles)+1, input_length=maxim_smiles, output_dim=128)(smiles_input)
    x = Conv1D(
        filters=32, kernel_size=3, padding="SAME", input_shape=(4000, maxim_smiles))(embed)
    x = PReLU()(x)

    x = Conv1D(filters=64, kernel_size=3, padding="SAME")(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv1D(
        filters=128, kernel_size=3, padding="SAME")(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    pool = GlobalMaxPooling1D()(
        x)  # maxpool per obtenir un vector de 1d

    # model per fastas
    fasta_input = tf.keras.Input(shape=(maxim_fasta,), name='fasta_input')
    embed2 = Embedding(input_dim=len(
        elements_fasta)+1, input_length=maxim_fasta, output_dim=256)(fasta_input)
    x2 = Conv1D(
        filters=32, kernel_size=3, padding="SAME", input_shape=(4000, maxim_fasta))(embed2)
    x2 = PReLU()(embed2)

    x2 = Conv1D(
        filters=64, kernel_size=3, padding="SAME")(x2)
    x2 = BatchNormalization()(x2)
    x2 = PReLU()(x2)
    x2 = Conv1D(
        filters=128, kernel_size=3, padding="SAME")(x2)
    x2 = BatchNormalization()(x2)
    x2 = PReLU()(x2)
    pool2 = GlobalMaxPooling1D()(
        x2)  # maxpool per obtenir un vector de 1d

    junt = concatenate(inputs=[pool, pool2])

    # dense

    de = Dense(units=1024, activation="relu")(junt)
    dr = Dropout(0.3)(de)
    de = Dense(units=1024, activation="relu")(dr)
    dr = Dropout(0.3)(de)
    de2 = Dense(units=512, activation="relu")(dr)

    # output

    output = Dense(
        1, activation="relu", name="output", kernel_initializer="normal")(de2)

    modelo = tf.keras.models.Model(
        inputs=[smiles_input, fasta_input], outputs=[output])

    modelo.load_weights(f"{path_model}")
    smiles_in = []
    for element in smile:
        smiles_in.append(int_smiles[element])
    while (len(smiles_in) != maxim_smiles):
        smiles_in.append(0)

    fasta_in = []
    for amino in fasta:
        fasta_in.append(int_fasta[amino])
    while (len(fasta_in) != maxim_fasta):
        fasta_in.append(0)

    predecir = modelo.predict({'smiles_input': np.array(smiles_in).reshape(1, 100,),
                               'fasta_input': np.array(fasta_in).reshape(1, 5000,)})[0][0]
    print(predecir)

    return predecir


# mesurador_afinitat(smile="CSc1ccccc1-c1ccccc1-c1nnnn1-c1ccccc1F",fasta = "MGGDLVLGLGALRRRKRLLEQEKSLAGWALVLAGTGIGLMVLHAEMLWFGGCSWALYLFLVKCTISISTFLLLCLIVAFHAKEVQLFMTDNGLRDWRVALTGRQAAQIVLELVVCGLHPAPVRGPPCVQDLGAPLTSPQPWPGFLGQGEALLSLAMLLRLYLVPRAVLLRSGVLLNASYRSIGALNQVRFRHWFVAKLYMNTHPGRLLLGLTLGLWLTTAWVLSVAERQAVNATGHLSDTLWLIPITFLTIGYGDVVPGTMWGKIVCLCTGVMGVCCTALLVAVVARKLEFNKAEKHVHNFMMDIQYTKEMKESAARVLQEAWMFYKHTRRKESHAARRHQRKLLAAINAFRQVRLKHRKLREQVNSMVDISKMHMILYDLQQNLSSSHRALEKQIDTLAGKLDALTELLSTALGPRQLPEPSQQSK")
