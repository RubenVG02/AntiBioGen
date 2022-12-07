from arreglar_dades import neteja_dades_afinitat
import pandas as pd
from tensorflow import keras
import re

dades=neteja_dades_afinitat()

arx=pd.read_csv(dades, sep=",")
re.sub("nan","",arx["IC50"])
datfr = arx[arx.IC50.notna()]
datfr = datfr[arx.smiles.notna()]
# print(datfr)
df_no_dup = datfr.drop_duplicates(['smiles'])
df_no_dup["IC50"].astype(float)
re.sub(">","",df_no_dup["IC50"])
df2 = df_no_dup[df_no_dup["IC50"] <= 1000000]

df2 = df2[df2["IC50"] > 0]
df2 = df2.reset_index(drop=True)
df2.to_csv(f"prueba.csv", index=False, sep=";")

print(arx.head())
print(len(arx))
print(len(arx["IC50"]>1000000))
'''maxim_smiles=140  #valor maxim que vull que tingin les meves smiles, serviran per entrenar el model
elements_smiles=['6', '3', '=', 'H', 'C', 'O', 'c', '#', 'a', '[', 't', 'r', 'K', 'n', 'B', 'F', '4', '+', ']', '-', '1', 'P', '0', 'L', '%', 'g', '9', 'Z', '(', 'N', '8', 'I', '7', '5', 'l', ')', 'A', 'e', 'o', 'V', 's', 'S', '2', 'M', 'T', 'u', 'i']
#elements_smiles fa referencia a els elements pels quals es poden formar els smiles

maxim_fasta=5000
elements_fasta=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L','M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

def model_cnn():

    #model per smiles
    i=keras.keras.Input(shape=(maxim_smiles,))
    embed=keras.keras.layers.Embedding(input_dim=len(elements_smiles),input_length=maxim_smiles, output_dim=128)(i)
    x=keras.keras.layers.Conv1D(filters=32, kernel_size=3, padding="SAME")(embed)
    x=keras.keras.layers.PReLU()(x)
    x=keras.keras.layers.Conv1D(filters=64, kernel_size=3, padding="SAME")(x)
    x=keras.keras.layers.PReLU()(x)
    x=keras.keras.layers.Conv1D(filters=128, kernel_size=3, padding="SAME")(x)
    x=keras.keras.layers.PReLU()(x)
    pool=keras.keras.layers.GlobalAvgPool1D()(x) #maxpool per obtenir un vector de 1d

    #model per fastas
    i2=keras.keras.Input(shape=(maxim_smiles,))
    embed2=keras.keras.layers.Embedding(input_dim=len(elements_fasta+1),input_length=maxim_fasta, output_dim=256)(i2)
    x2=keras.keras.layers.Conv1D(filters=32, kernel_size=3, padding="SAME")(embed2)
    x2=keras.keras.layers.PReLU()(x2)
    x2=keras.keras.layers.Conv1D(filters=64, kernel_size=3, padding="SAME")(x2)
    x2=keras.keras.layers.PReLU()(x2)
    x2=keras.keras.layers.Conv1D(filters=128, kernel_size=3, padding="SAME")(x2)
    x2=keras.keras.layers.PReLU()(x2)
    pool2=keras.keras.layers.GlobalAvgPool1D()(x2) #maxpool per obtenir un vector de 1d


    junt=keras.keras.layers.concatenate(pool, pool2)

    #dense

    de=keras.keras.layers.Dense(units=512, activation="relu",)(junt)
    dr=keras.keras.layers.Dropout(0.2)(de)
    de2=keras.keras.layers.Dense(units=256,activation="relu")(dr)

    #output

    op=keras.keras.layers.Dense(1,activation="relu")(de2)

    modelo=keras.keras.models.Model(inputs=[i,i2], output=op)

    modelo.compile(optimizer="adam", loss="mse")
 
    modelo.fit()
'''