from rdkit import Chem
from rdkit.Chem import Draw


smile = "OC(=O)[C@@H](C)[NH2+][C@@H](Cc1c[nH]c2ccccc12)C(=O)NCC(C)C"


def draw_smile(smile, filename):
    '''
    Function to draw a molecule from a smile sequence

    Parameters:
        -smile: SMILE sequence of the molecule
        -filename: Name of the file where the image will be saved
    
    '''
    molecule = Chem.MolFromSmiles(smile)
    Draw.MolToImageFile(molecule, filename=filename,
                        size=(400, 300))

    
