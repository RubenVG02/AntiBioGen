
from mega import Mega
import csv

from pretrained_rnn import generador
from check_affinitat import mesurador_afinitat

import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw
import os
import sys
import base64

# Errors sacore
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

target = "MTSVMSHEFQLATAETWPNPWPMYRALRDHDPVHHVVPPQRPEYDYYVLSRHADVWSAARDHQTFSSAQGLTVNYGELEMIGLHDTPPMVMQDPPVHTEFRKLVSRGFTPRQVETVEPTVRKFVVERLEKLRANGGGDIVTELFKPLPSMVVAHYLGVPEEDWTQFDGWTQAIVAANAVDGATTGALDAVGSMMAYFTGLIERRRTEPADDAISHLVAAGVGADGDTAGTLSILAFTFTMVTGGNDTVTGMLGGSMPLLHRRPDQRRLLLDDPEGIPDAVEELLRLTSPVQGLARTTTRDVTIGDTTIPAGRRVLLLYGSANRDERQYGPDAAELDVTRCPRNILTFSHGAHHCLGAAAARMQCRVALTELLARCPDFEVAESRIVWSGGSYVRRPLSVPFRVTS"


def crear_arxiu(nom_arxiu, headers=["smiles", "IC50", "score"]):
    '''
    Funció per crear el arxiu .csv al qual se li afegiran les dades obtingudes
    
    Paràmetres:
    -headers: Noms de les columnes que volem utilizar 
    '''
    with open(f"{nom_arxiu}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

def pujar_mega(nom_arx):
    '''
    Funció per pujar l'arxiu csv obtingut a mega, i obtenir un enllaç de descàrrega
    '''
    mail="joneltmp+dilzy@gmail.com"
    contra=base64.b64decode("J2NudncnZDkwY253cTljcG53cW5lamR3cHFjbm1qZXcnYzlu")
    contra=contra.decode("UTF-8")
    mega=Mega()
    mega._login_user(email=mail, password=contra)
    pujada = mega.upload(f"{nom_arx}.csv")
    link=mega.get_upload_link(pujada)
    print(link)
    
def dibuix_millor(ic50_menor, ic50, smiles, nom_arx):
    '''
    Funció per dibuixar la millor molècula obtinguda en funció del valor de IC50
    '''
    index=ic50.index(ic50_menor)
    millor=smiles[index]
    molecula=Chem.MolFromSmiles(millor)
    Draw.MolToImageFile(molecula, filename=f"millor_molecula_{nom_arx}.jpg",
            size=(400, 300))

def buscar_candidats(target=target, nom_arx="Alfa_Pol3 (B.Subtilis)", pujar_a_mega=True, dibuixar_menor=True, maxim_molecules=5, db_smiles=True, arx_db=r"C:\Users\ASUS\Desktop\github22\dasdsd\moleculas_generadas\moleculas_nuevo_generador\moleculas_druglike2.txt", valor_acceptat=100):
    '''
    Funció per generar molécules utilitzant un model RNN, i comparar la seva afinitat amb un target específic, a més d'obtenir un score representatiu a la 
    complexitat de la seva síntesi
    
    Paràmetres:
    -target: Seqüència target la qual utilizarem per mirar l'afinitat
    -nom_arx: nom de l'arxiu csv on es guardaran els resultats obtinguts (Columnes: smiles, IC50, score)
    -pujar_a_mega: pujar el csv generat a Mega.nz i obtenir un link de descàrrega per poder descarregar posteriorment l'arxiu. Per default, True
    -dibuixar_menor: Obtenir un arxiu .jpg de la molècula smile amb millor afinitat. Per default, True
    
    
    Returns:
    -Arxiu .csv amb els resultats obtinguts amb el teu target d'interés.
    -Imatge amb format .jpg de la millor molècula obtinguda per el teu target, si dibuixar_menor=True
    -Enllaç a Mega del teu .csv si pujar_a_mega=True
    '''
    ic50 = []
    smiles = []
    mirar = []
    score = []
    crear_arxiu(nom_arxiu=nom_arx)
    valor=0
    while not valor==maxim_molecules:
        if not db_smiles:
            generats = generador(nombre_generats=10, img_druglike=False)
            smiles.extend(generats)
        else:
            with open(arx_db, "r") as file:
                generats=[line.rstrip() for line in file]
        for i in generats:
            molecula = Chem.MolFromSmiles(i)
            sascore = sascorer.calculateScore(molecula)
            score.append(sascore)
            if db_smiles:
                smiles.append(i)
            i = i.replace("@", "").replace("/", "")
            try:
                predicció_ic50 = mesurador_afinitat(smile=i, fasta=target)
                if predicció_ic50<valor_acceptat:
                    valor+=1
                ic50.append(float(predicció_ic50))
                mirar.append(float(predicció_ic50))
            except:
                ic50.append(999999999999999)
            if valor==maxim_molecules:
                break
    ic50_menor = min(ic50)
    combinacio = list(zip(smiles, ic50, score))
    linies = open(f"{nom_arx}.csv", "r").read()
    with open(f"{nom_arx}.csv", "a", newline="") as file:
        for i in combinacio:
            if str(i[1]) not in linies and float(i[1])<100:
                file.write(f"{i[0]},{i[1]},{i[2]}\n")
                
    if pujar_a_mega == True:
        pujar_mega(nom_arx)
        
    
    '''FQx1aKXvDO4jabS4siLmxw'''
    if dibuixar_menor==True:
        dibuix_millor(ic50_menor,ic50,smiles, nom_arx)
            

buscar_candidats(nom_arx="Citocrom_P450 (M.Tuberculosis)")



