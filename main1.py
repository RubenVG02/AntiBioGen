dades = open(r"C:\Users\ASUS\Desktop\github22\dasdsd\xab.txt").read()
dades2 = open(
    r"C:\Users\ASUS\Desktop\github22\dasdsd/prueba/xaa.txt").read()

print(list(set(dades)-set(dades2)))

'''# per obtenir els elements unics de dades
elements_smiles = {u: i for i, u in enumerate(sorted(set(dades)))}
elements_smiles.update({-1: "\n"})

# per passar els elements unics de dades a
int_a_elements = {i: u for i, u in enumerate(sorted(set(dades)))}
int_a_elements.update({"\n": -1})

elements_smiles2 = {u: i for i, u in enumerate(sorted(set(dades2)))}
elements_smiles2.update({-1: "\n"})

# per passar els elements unics de dades a
int_a_elements2 = {i: u for i, u in enumerate(sorted(set(dades2)))}
int_a_elements2.update({"\n": -1})

mapa_int = len(elements_smiles)
mapa_char = len(int_a_elements)

restantes = list(set)

print(mapa_char-1)
print(mapa_int)'''
