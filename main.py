import pandas as pd
import requests
import csv

from DeepPurpose import utils, dataset
from DeepPurpose import DTI as modelos

cols = ["CMPD CHEMBLID ", "CANONICAL SMILES"]
df = pd.read_csv("antibiotics.csv", on_bad_lines='skip',
                 usecols=cols, sep=";").to_csv("name + smile.csv", index=False, sep=";")


CHEMBL_ID_K12="CHEMBL2540"

#mol, mol_nombre, mol_id = dataset.load_antiviral_drugs()
