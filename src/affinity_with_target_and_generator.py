
from mega import Mega
import csv

from pretrained_rnn import generator
from check_affinity import calculate_affinity

import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw
import os
import sys
import base64
import qrcode

#To import sascore to use the Accessibility Score
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

target = "MTSVMSHEFQLATAETWPNPWPMYRALRDHDPVHHVVPPQRPEYDYYVLSRHADVWSAARDHQTFSSAQGLTVNYGELEMIGLHDTPPMVMQDPPVHTEFRKLVSRGFTPRQVETVEPTVRKFVVERLEKLRANGGGDIVTELFKPLPSMVVAHYLGVPEEDWTQFDGWTQAIVAANAVDGATTGALDAVGSMMAYFTGLIERRRTEPADDAISHLVAAGVGADGDTAGTLSILAFTFTMVTGGNDTVTGMLGGSMPLLHRRPDQRRLLLDDPEGIPDAVEELLRLTSPVQGLARTTTRDVTIGDTTIPAGRRVLLLYGSANRDERQYGPDAAELDVTRCPRNILTFSHGAHHCLGAAAARMQCRVALTELLARCPDFEVAESRIVWSGGSYVRRPLSVPFRVTS"


def create_file(name_file, headers=["smiles", "IC50", "score"]):
    '''
    Function to create the .csv file to which the obtained data will be added
    
    Parameters:
    -name_file: Name of the file to be created
    -headers: Names of the columns we want to use
    '''
    with open(f"{name_file}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

def upload_mega(name_file, mail="", passw=""):
    '''
    Function to upload the file to mega and obtain the link to download it

    Parameters:
    -name_file: Name of the file to be uploaded
    -mail: Mail of the mega account
    -passw: Password of the mega account

    
    '''
    mega=Mega()
    mega._login_user(email=mail, password=passw)
    upload = mega.upload(f"{name_file}.csv")
    link=mega.get_upload_link(upload)
    print(link)
    return link
    
def draw_best(ic50_minor, ic50, smiles, name_file):
    '''
    Function to draw the best molecule obtained
    
    Parameters:
    -ic50_minor: Value of the best affinity obtained
    -ic50: List of all the affinities obtained
    -smiles: List of all the smiles obtained
    -name_file: Name of the file where the image will be saved
    
    '''

    index=ic50.index(ic50_minor)
    best=smiles[index]
    molecule=Chem.MolFromSmiles(best)
    Draw.MolToImageFile(molecule, filename=fr"examples\results_examples\mols_obtained/best_molecule_{name_file}.jpg",
            size=(400, 300))

def find_candidates(target=target, name_file_destination="Alfa_Pol3 (B.Subtilis)", upload_to_mega=True, draw_minor=True, max_molecules=5, db_smiles=True, arx_db=r"examples\generated_molecules\molecules_new_generator\moleculas_druglike2.txt", accepted_value=100, generate_qr=True):
    '''
    Function to generate molecules using an RNN model, and compare their affinity with a specific target, in addition to obtaining a representative score in the
    complexity of its synthesis
    
    Parameters:
    -target: Target sequence that we will use to look at the affinity
    -name_file_destination: name of the csv file where the obtained results will be saved (Columns: smiles, IC50, score)
    -upload_to_mega: upload the generated csv to Mega.nz and obtain a download link to be able to download the file later. By default, True
    -draw_minor: Get a .jpg file of the smile molecule with better affinity. By default, True
    -db_smiles: Analyze affinity from a .txt file with SMILES. Default, True
    -arx_db: Link of the db archive with SMILES. Requires db_smiles=True
    -accepted_value: Value from which we can consider a molecule as valid expressed in nM. Default value, 100.
    -max_molecules: Maximum amount of molecules you want in your destination file, based on the accepted_value parameter. By default, 5.
    -generate_qr: To generate a qr file of your mega link. Requires upload_to_mega=True. The QR will be saved as qr_{arx_name}.
    
    Returns:
    -File .csv with the results obtained with your target of interest.
    -Image in .jpg format of the best molecule obtained by your target, if draw_minor=True
    -Link to Mega from your .csv if upload_to_mega=True
    '''
    ic50 = []
    smiles = []
    score = []
    create_file(name_file=name_file_destination)
    valor=0
    while not valor==max_molecules:
        if not db_smiles:
            generated = generator(nombre_generats=10, img_druglike=False)
            smiles.extend(generated)
        else:
            with open(arx_db, "r") as file:
                generated=[line.rstrip() for line in file]
        for i in generated:
            molecule = Chem.MolFromSmiles(i)
            sascore = sascorer.calculateScore(molecule)
            score.append(sascore)
            if db_smiles:
                smiles.append(i)
            i = i.replace("@", "").replace("/", "")
            try:
                ic50_prediction = calculate_affinity(smile=i, fasta=target)
                if ic50_prediction<accepted_value:
                    valor+=1
                ic50.append(float(ic50_prediction))
            except:
                ic50.append(999999999999999)
            if valor==max_molecules:
                break
    ic50_menor = min(ic50)
    combination = list(zip(smiles, ic50, score))
    lines = open(f"{name_file_destination}.csv", "r").read()
    with open(f"{name_file_destination}.csv", "a", newline="") as file:
        for i in combination:
            if str(i[1]) not in lines and float(i[1])<100:
                file.write(f"{i[0]},{i[1]},{i[2]}\n")
                
    if upload_to_mega == True:
        enllaç=upload_mega(name_file=name_file_destination)
        if generate_qr:
            qr_generat=qrcode.make(enllaç)
            qr_generat.save(fr"examples\results_examples\qr_codes_generated/qr_{name_file_destination}.png")
        
    
    '''FQx1aKXvDO4jabS4siLmxw'''
    if draw_minor==True:
        draw_best(ic50_menor,ic50,smiles, name_file_destination)
            



find_candidates()


