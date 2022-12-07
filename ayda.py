from arreglar_dades import neteja_dades_afinitat
import pandas as pd
import re


dades = neteja_dades_afinitat()


def neteka_dades_cnn(arx_final="CNN_Net"):
    arx = pd.read_csv(dades, sep=",")
    arx["IC50"].str.strip()
    arx = arx.dropna(how="any").reset_index(drop=True)
    # print(datfr)
    df_no_dup = arx.drop_duplicates(['smiles'])
    df_no_dup["IC50"] = df_no_dup["IC50"].str.replace(r"[<>]", "", regex=True)
    df_no_dup["IC50"] = df_no_dup["IC50"].astype(float)
    #df_no_dup["IC50"] = df_no_dup[df_no_dup["IC50"] < 1000000]
    df_no_dup.to_csv(f"{arx_final}.csv", index=False, sep=";")

    print(arx.head())
    print(len(arx))
