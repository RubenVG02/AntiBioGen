dades = open(r"C:\Users\ASUS\Desktop\github22\dasdsd/prueba/xab.txt").read()
dades2 = open(
    r"C:\Users\ASUS\Desktop\github22\dasdsd/prueba/xaa.txt").read()

print(list(set(dades2)-set(dades)))
