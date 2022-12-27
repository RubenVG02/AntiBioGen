from rdkit import Chem
from rdkit.Chem import Draw



smile = "CCC(=O)N1CCc2ccc(S(=O)(=O)NCC[NH+](C)Cc3ccccc3)cc21"

molecula = Chem.MolFromSmiles(smile)
Draw.MolToImageFile(molecula, filename="molecula22.jpg",
                    size=(400, 300))
