import pandas as pd
from chembl_webresource_client.new_client import new_client


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


def neteja_dades(nom_arx="drugs.csv", nom_arx2="dades_netes"):
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


#selecció = seleccio_mol("CHEMBL6137", nom_df="datafr")
selecció = neteja_dades()
