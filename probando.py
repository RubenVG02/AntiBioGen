lista1 = ["hola", "ffdfds", "fdsfdsf", "adios"]
lista2 = [423243, 7, 3, 5]

a = zip(lista1, lista2)

a = list(a)
with open("prueba.csv", "w") as file:
    for i in a:
        file.write(f"{i[0]},{i[1]}\n")
        print(f"{i[0]},{i[1]}")
