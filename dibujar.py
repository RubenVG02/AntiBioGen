import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from IPython.display import SVG


smile = '''(=O)NC1CCCCCC1
CC[C@H](C(=O)N[C@@H](C)c1ccc(-c2ccccc2)n1)c1ccccc1
CCn1cnnn1CNC(=O)c1ccc(-n2nc3sc3c(c2)OCO3)SC@H](O)C1
O=C(Nc1nc2ccc(Cl)c'''

molecula = Chem.MolFromSmiles(smile)
Draw.MolToImageFile(molecula, filename="molecula22.jpg",
                    size=(400, 300))
