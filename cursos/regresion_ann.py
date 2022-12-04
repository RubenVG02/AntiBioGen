import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

X = np.random.random((1000, 2)) * 6-3  # matrix de 1000x2

Y = np.cos(2*X[:, 0]) + np.cos(3*X[:, 1])
# el [:,0] te da todos los elementos de la columna 0 [ first_row:last_row , column]

# scatter es para meter los puntos en un mapa de dispersion
plt.figure().add_subplot(111, projection="3d").scatter(X[:, 0], X[:, 1], Y)
# plt.show()


modelo = tf.keras.models.Sequential()
modelo.add(tf.keras.layers.Dense(128, input_shape=(2,), activation="relu"))
modelo.add(tf.keras.layers.Dense(1))

adam_dif = tf.keras.optimizers.Adam(0.01)
modelo.compile(optimizer=adam_dif, loss="mse")
r = modelo.fit(X, Y, epochs=100)


# aqui como no usamos test, no hay val loss, solo loss normal
plt.plot(r.history["loss"], label="loss")
plt.legend()
plt.show()
