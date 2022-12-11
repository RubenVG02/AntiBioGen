import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from IPython.display import SVG


smile = "COCCOc1cc2ncc(C#N)c(Nc3cc(OC)c(OC)c(OC)c3)c2cc1OCCOC"

molecula = Chem.MolFromSmiles(smile)
Draw.MolToImageFile(molecula, filename="molecula22.jpg",
                    size=(400, 300))
