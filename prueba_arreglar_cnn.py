import pandas as pd

import csv

with open ("inh.tsv","r") as file:
    df=pd.read_csv(file, sep="\t", on_bad_lines="skip",low_memory=False)
    columna_cadena=df["BindingDB Target Chain  Sequence"]
    columna_50=df["IC50 (nM)"]
    columna_smiles=df["Ligand SMILES"]

    cadena=[]
    for i in columna_cadena:
        cadena.append(i)
    ic50=[]
    for i in columna_50:
        ic50.append(i)
    smiles=[]
    for i in columna_smiles:
        smiles.append(i)
    headers=["smiles", "IC50","sequence"]
    listas=[smiles, ic50, cadena]
    
with open("probando.csv","w") as archivo:
    write=csv.writer(archivo)
    write.writerow(headers)
    write.writerows(zip(*listas))

        

