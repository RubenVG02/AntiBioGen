
def comparar(dades1=r"C:\Users\ASUS\Desktop\github22\dasdsd\arxius_txt\smalls_prueba.txt", dades2=r"C:\Users\ASUS\Desktop\github22\dasdsd\exemples_de_smiles_per_entrenar\xaa.txt"):
    '''
    Funció per comparar la quantitat d'elements que tenen dos db d'SMILES, així pots fer-les servir per un model determinat
    
    Paràmetres:
        -dades1: Path del primer arxiu amb Smiles
        -dades2: Path del segon arxiu amb Smiles
        
    Return:
        -list amb els dos arxius
    '''


    dades1 = open(dades1).read()
    dades2 = open(dades2).read()
    
    diferència=list(set(dades1)-set(dades2))
    return diferència

comparar()