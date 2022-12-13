from rdkit import Chem
import rdkit.Chem.rdmolfiles as rdmol
from rdkit.Chem import Draw


import rdkit
from rdkit import Chem


def mol2seq(m):
    aa_smiles = {'ALA': 'C[C@H](N)C=O', 'CYS': 'N[C@H](C=O)CS', 'ASP': 'N[C@H](C=O)CC(=O)O', 'GLU': 'N[C@H](C=O)CCC(=O)O', 'PHE': 'N[C@H](C=O)Cc1ccccc1', 'GLY': 'NCC=O', 'HIS': 'N[C@H](C=O)Cc1c[nH]cn1', 'ILE': 'CC[C@H](C)[C@H](N)C=O', 'LYS': 'NCCCC[C@H](N)C=O', 'LEU': 'CC(C)C[C@H](N)C=O',
                 'MET': 'CSCC[C@H](N)C=O', 'ASN': 'NC(=O)C[C@H](N)C=O', 'PRO': 'O=C[C@@H]1CCCN1', 'GLN': 'NC(=O)CC[C@H](N)C=O', 'ARG': 'N=C(N)NCCC[C@H](N)C=O', 'SER': 'N[C@H](C=O)CO', 'THR': 'C[C@@H](O)[C@H](N)C=O', 'VAL': 'CC(C)[C@H](N)C=O', 'TRP': 'N[C@H](C=O)Cc1c[nH]c2ccccc12', 'TYR': 'N[C@H](C=O)Cc1ccc(O)cc1'}
    aas = ['GLY', 'ALA', 'VAL', 'CYS', 'ASP', 'GLU', 'PHE', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO',
           'GLN', 'ARG', 'SER', 'THR', 'TRP', 'TYR']  # order important because gly is substructure of other aas
    # detect the atoms of the backbone and assign them with info
    CAatoms = m.GetSubstructMatches(
        Chem.MolFromSmarts("[C:0](=[O:1])[C:2][N:3]"))
    for atoms in CAatoms:
        a = m.GetAtomWithIdx(atoms[2])
        info = Chem.AtomPDBResidueInfo()
        info.SetName(" CA ")  # spaces are important
        a.SetMonomerInfo(info)
    # detect the presence of residues and set residue name for CA atoms only
    for curr_aa in aas:
        matches = m.GetSubstructMatches(Chem.MolFromSmiles(aa_smiles[curr_aa]))
        for atoms in matches:
            for atom in atoms:
                a = m.GetAtomWithIdx(atom)
                info = Chem.AtomPDBResidueInfo()
                if a.GetMonomerInfo() != None:
                    if a.GetMonomerInfo().GetName() == " CA ":
                        info.SetName(" CA ")
                        info.SetResidueName(curr_aa)
                        a.SetMonomerInfo(info)
    # renumber the backbone atoms so the sequence order is correct:
    # generate backbone SMILES
    bbsmiles = "O" + \
        "C(=O)CN" * \
        len(m.GetSubstructMatches(Chem.MolFromSmiles(aa_smiles["GLY"])))
    backbone = m.GetSubstructMatches(Chem.MolFromSmiles(bbsmiles))[0]
    id_list = list(backbone)
    id_list.reverse()
    for idx in [a.GetIdx() for a in m.GetAtoms()]:
        if idx not in id_list:
            id_list.append(idx)
    m_renum = Chem.RenumberAtoms(m, newOrder=id_list)
    return Chem.MolToSequence(m_renum)


smile = "N(CCC)NC(=O)NC(C)(C)CCOc1ccccc1CC(=O)N1CCN(C(=O)c2ccccc2)CC1CCN(CCO)C(=O)Nc1cnn(C2CC2)c1CCOC(=O)C1CC[NH+](CCC(=O)Nc2ccccc2)"
m = Chem.MolFromSmiles(smile)


Draw.MolToMPL(m).savefig("molecula.jpg", bbox_inches="tight")

print(mol2seq(m))
