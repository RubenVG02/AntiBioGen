from rdkit import Chem
from rdkit.Chem import Draw



smile = "OC(=O)[C@@H](C)[NH2+][C@@H](Cc1c[nH]c2ccccc12)C(=O)NCC(C)C"

molecula = Chem.MolFromSmiles(smile)
Draw.MolToImageFile(molecula, filename="molecula22.jpg",
                    size=(400, 300))

