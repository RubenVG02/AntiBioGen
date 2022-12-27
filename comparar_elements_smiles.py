dades = open(r"C:\Users\ASUS\Desktop\github22\smalls_prueba.txt").read()
dades2 = open(
    r"C:\Users\ASUS\Desktop\github22\dasdsd/prueba/xaa.txt").read()

print(len(set(dades)))
print(list(set(dades)-set(dades2)))
