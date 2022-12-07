from arreglar_dades import neteja_dades_afinitat
import pandas as pd
import re


dades = neteja_dades_afinitat()

arx = pd.read_csv(dades, sep=",")
arx["IC50"].str.strip()
arx[arx["IC50"].str.contains("nan") == False]
datfr = arx[arx.smiles.notna()]
# print(datfr)
df_no_dup = datfr.drop_duplicates(['smiles'])
df_no_dup["IC50"] = df_no_dup["IC50"].str.replace(r"[<>]", "", regex=True)
df_no_dup["IC50"] = df_no_dup["IC50"].astype(float)
'''df2 = df_no_dup[df_no_dup["IC50"] <= 1000000]'''
print("Hasta aqui va bien")
'''df2 = df2[df2["IC50"] > 0]
df2 = df2.reset_index(drop=True)'''
df_no_dup.to_csv(f"prueba.csv", index=False, sep=";")

print(arx.head())
print(len(arx))
print(len(arx["IC50"] > 1000000))
