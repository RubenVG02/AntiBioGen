from sklearn.datasets import load_breast_cancer
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # Para estandarizar tus valores

import matplotlib.pyplot as plt
import numpy as np

# %%
datos = load_breast_cancer()
# print(datos.data.shape)  # Para obtener las dimensiones

# print(datos.target_names)  # PAra saber en funcion de que esta tu df
# print(datos.target)  # Binario de lo que quieres buscar en tu df``

X_train, X_test, Y_train, Y_test = train_test_split(
    datos.data, datos.target, test_size=0.2)

N, D = X_train.shape
print(X_train)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Esto es para escalar tus valores

modelo = tf.keras.models.Sequential([tf.keras.layers.Input(
    shape=(D,)), tf.keras.layers.Dense(1, activation="sigmoid")])
modelo.compile(optimizer="adam", loss="binary_crossentropy",
               metrics=["accuracy"])

r = modelo.fit(X_train, Y_train, validation_data=(
    X_test, Y_test), epochs=200)

print(modelo.evaluate(X_train, Y_train))

prediccion = modelo.predict(X_test)
# Flatten te lo convierte en una array (,1), y round para decidir binario
prediccion = np.round(prediccion).flatten()

print(prediccion)
# recuerda que en np, el == te da bools, y aqui haces las medias de los booleanos
print((np.mean(prediccion == Y_test)))
# Esto en si es lo mismo que model.evaluate()

'''
plt.plot(r.history["loss"], label="perdida"),
plt.plot(r.history["val_loss"], label="val_perdida"),
plt.legend()
plt.show()'''
