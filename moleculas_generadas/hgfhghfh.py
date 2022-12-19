from rdkit import Chem
from rdkit.Chem import Draw


final = "CC[C@H]1CCC[C@@H](C)[C@H]1NC(=O)N(C)C(=O)N1CCCC[C@@H]1C"
asa = final.replace(" ", '''\\n''')
print(asa)
a = Chem.MolFromSmiles(asa)
Draw.MolToImageFile(
    a, filename="moleculas_generadas/molecula9999999999.jpg", size=(400, 300))
print(a)
