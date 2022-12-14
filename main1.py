dades = open(r"C:\Users\ASUS\Desktop\github22\dasdsd\parte1_txt.txt").read()

# per obtenir els elements unics de dades
elements_smiles = {u: i for i, u in enumerate(sorted(set(dades)))}
elements_smiles.update({-1: "\n"})

# per passar els elements unics de dades a
int_a_elements = {i: u for i, u in enumerate(sorted(set(dades)))}
int_a_elements.update({"\n": -1})

mapa_int = len(elements_smiles)
mapa_char = len(int_a_elements)

print(mapa_char-1)
print(mapa_int)
