
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, Input, PReLU, Dropout, concatenate, BatchNormalization
from keras.regularizers import l2



arx = pd.read_csv(
    r"C:\Users\ASUS\Desktop\github22\dasdsd\CSV\500k_dades.csv", sep=",")


# valor maxim que vull que tingin les meves smiles, serviran per entrenar el model
maxim_smiles = 130
elements_smiles = ['N', '2', 'H', 'K', 'Z', 'O', 'M', ']', 'c', 'l', '=', '6', ')', 'F', 'o', 'r', '7', 'P','g', '5', 't', '8', '9', '1', '0', 'I', '4', '[', 'i', 'a', 'C', '-', 'n', '#', 'L', '(', 'S', 'B', 'A', 'T', 's', '3', '+', 'e']
# elements_smiles fa referencia a els elements pels quals es poden formar els smiles
# si es vol utilitzar totes les del teu arx: arx.smiles.apply(list).sum(), tot i que trigará molt en arx molt grans

int_smiles = dict(zip(elements_smiles, range(1, len(elements_smiles)+1)))
# Per associar tots els elements amb un int determinat (range és 1, len+1 perquè es plenen amb zeros per arribar a maxim_smiles)

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
        smiles_llista1 = []
        for elements in i:  # Elements fa referència a els elements que formen elements_smile
            try:
                smiles_llista1.append(int_smiles[elements])
            except:
                pass
        while (len(smiles_llista1) != maxim_smiles):
            smiles_llista1.append(0)
        smiles_amb_numeros.append(smiles_llista1)

    fasta_amb_numeros = []
    for i in arx.sequence:
        fasta_lista1 = []
        for elements in i:  # Elements fa referència a els elements que formen elements_smile
            try:
                fasta_lista1.append(int_fasta[elements])
            except:
                pass
        while (len(fasta_lista1) != maxim_fasta):
            fasta_lista1.append(0)
        fasta_amb_numeros.append(fasta_lista1)

    ic50_numeros = list(arx.IC50)

    return smiles_amb_numeros, fasta_amb_numeros, ic50_numeros


X_test_smile, X_test_fasta, T_test_IC50 = convertir(arx[350000:])


def model_cnn():
    ''' 
        Model per entrenar les dades.
    '''
    # regulador kernel
    regulador = l2(0.001)

    # model per a smiles
    smiles_input = Input(
        shape=(maxim_smiles,), dtype='int32', name='smiles_input')
    embed = Embedding(input_dim=len(
        elements_smiles)+1, input_length=maxim_smiles, output_dim=128)(smiles_input)
    x = Conv1D(
        filters=32, kernel_size=3, padding="SAME", input_shape=(50700, maxim_smiles))(embed)
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
    fasta_input = Input(shape=(maxim_fasta,), name='fasta_input')
    embed2 = Embedding(input_dim=len(
        elements_fasta)+1, input_length=maxim_fasta, output_dim=256)(fasta_input)
    x2 = Conv1D(
        filters=32, kernel_size=3, padding="SAME", input_shape=(50700, maxim_fasta))(embed2)
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

 
    # funció per mirar la precisió del model (serà la nostra metric)
    def r2_score(y_true, y_pred):
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1-SS_res/(SS_tot)+K.epsilon())

    modelo.load_weights(
        r"C:\Users\ASUS\Desktop\github22\dasdsd\models_definitius\model_prueba_cnn.hdf5")
    modelo.compile(optimizer="adam",
                   # categorical_crossentropy/mean_squared_logarithmic_error/ tf.keras.losses.mean_squared_logarithmic_error
                   loss={'output': "mean_squared_logarithmic_error"},
                   metrics={'output': r2_score})
    # s
    save_model_path = "model_prueba_cnn.hdf5"
    checkpoint = ModelCheckpoint(save_model_path,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True)

    # Utilitzem un valor elevat per poder obtenir millors resultats
    tamany_per_epoch = 50700

    train = arx[:355000]
    loss = []
    loss_validades = []
    epochs = 50

    for epoch in range(epochs):  # Quantitat d'epochs que vols utilitzar
        inici = 0
        final = tamany_per_epoch
        print(f"Començant el epoch {epoch+1}")

        while final < 355000:
            X_smiles, X_fasta, y_train = convertir(train[inici:final])

            r = modelo.fit({'smiles_input': np.array(X_smiles),
                            'fasta_input': np.array(X_fasta)}, {'output': np.array(y_train)},
                           validation_data=({'smiles_input': np.array(X_test_smile),
                                             'fasta_input': np.array(X_test_fasta)}, {'output': np.array(T_test_IC50)}),  callbacks=[checkpoint], epochs=1, batch_size=64, shuffle=True)

            inici += tamany_per_epoch
            final += tamany_per_epoch

        loss.append(r.history["loss"])
        loss_validades.append(r.history["val_loss"])

    plt.plot(range(epochs), loss, label="loss")
    plt.plot(range(epochs), loss_validades, label="val_loss")
    plt.legend()
    plt.show()


model_cnn()
