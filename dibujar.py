import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from IPython.display import SVG


smile = '''c(cc12)C1=CCNCC1
O=C1N[CH](CN2C(=O)C(=OC(=O)C(=OC(=O)C2=O)c1cccc(c1)C#N)C1CC1
OC(=O)c1ccc(cc1)S(=O)(=O)c1ccc(Oc2ccc(F)cc2F)c(=O)c1C
Cc1cc'''

molecula = Chem.MolFromSmiles(smile)
Draw.MolToImageFile(molecula, filename="molecula22.jpg",
                    size=(400, 300))
