import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from IPython.display import SVG


smile = "CCc1nc(NC(=O)N2CCN(C(=O)[C@@H]3CCC[C@@H]3C)CC2)ccc1OC"

molecula = Chem.MolFromSmiles(smile)
Draw.MolToImageFile(molecula, filename="molecula22.jpg",
                    size=(400, 300))
