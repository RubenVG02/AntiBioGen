lista1 = [1, 2, 3, 4]
lista2 = [5, 7, 3, 5]

a = zip(lista1, lista2)

a = list(a)

for i in a:
    print(f"{i[0]},{i[1]}")
