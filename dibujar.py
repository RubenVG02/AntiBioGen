from rdkit import Chem
from rdkit.Chem import Draw



smile = "Cc1nccn1C1CCN(C(=O)c2ccc(NC(=O)c3ccccc3)cc2)CC1"

molecula = Chem.MolFromSmiles(smile)
Draw.MolToImageFile(molecula, filename="molecula22.jpg",
                    size=(400, 300))

