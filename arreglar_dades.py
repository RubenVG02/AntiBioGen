import pandas as pd
from chembl_webresource_client.new_client import new_client
import csv


def seleccio_mol(id, nom_arx="dades", nom_df="df"):
    '''
    Parametres:
        -ID de CHEMBL 
        -Nom de l'arxiu (per determinat, "dades1")
        -Nom df (per determinat, "df")
    '''
    act = new_client.activity
    # IC50: cantidad de subs inhibidora que necesitas para inhibir al 50%
    res = act.filter(target_chembl_id=id).filter(standard_type="IC50")
    nom_df = pd.DataFrame.from_dict(res)
    nom_df.to_csv(f"{nom_arx}.csv", index_label=False)
    return nom_df


def neteja_dades_rnn(nom_arx="drugs.csv", nom_arx2="dades_netes"):
    ''' 
        Paràmetres:
        -nom_arx: nom arxiu input
        -nom_arx2: nom de l'arxiu net
    '''
    arx = pd.read_csv(nom_arx, sep=";", index_col=False)
    datfr = arx[arx.Standard_Value.notna()]
    datfr = datfr[arx.Smiles.notna()]
    # print(datfr)
    df_no_dup = datfr.drop_duplicates(['Smiles'])
    selecc = ['Molecule_ChEMBLID', 'Smiles', 'Standard_Value']
    df2 = df_no_dup[selecc]
    print(df2.value_counts())

    # SV de 1 m, gairebé inactiu, aixo elimina els elements >=1000000
    df2 = df2[df2["Standard_Value"] <= 1000000]
    df2 = df2[df2["Standard_Value"] > 0]
    df2 = df2.reset_index(drop=True)
    df2.to_csv(f"{nom_arx2}.csv", index=False, sep=";")

    # Quants elements utilitzes per entrenar el model
    print(df2.value_counts())


def neteja_dades_afinitat(nom_arx="inh", nom_desti="cnn_arreglat"):
    ''' 
        Paràmetres:
         -nom_arx: Nom de l'arxiu d'origen a modificar, en format tsv
         -nom_destí: Nom de l'arxiu de destí que es crearà

    '''
    with open(f"{nom_arx}.tsv", "r", encoding="utf8") as file:
        df = pd.read_csv(file, sep="\t", on_bad_lines="skip", low_memory=False)
        columna_cadena = df["BindingDB Target Chain  Sequence"]
        columna_50 = df["IC50 (nM)"]
        columna_smiles = df["Ligand SMILES"]

    cadena = []
    for i in columna_cadena:
        cadena.append(i)
    ic50 = []
    for i in columna_50:
        ic50.append(i)
    smiles = []
    for i in columna_smiles:
        smiles.append(i)

    headers = ["smiles", "IC50", "sequence"]
    listas = [smiles, ic50, cadena]
    with open(f"{nom_desti}.csv", "w", encoding="utf8") as archivo:
        write = csv.writer(archivo)
        write.writerow(headers)
        write.writerows(zip(*listas))

    return f"{nom_desti}.csv"

#selecció = seleccio_mol("CHEMBL6137", nom_df="datafr")


neteja_dades_afinitat()
