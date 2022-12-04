import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wget


datos = pd.read_csv("moore.csv", header=None, sep=",").values
# Con esto, conviertes el df en filas desconocidas (-1) y solo 1 columna (1)
X = datos[:, 0].reshape(-1, 1)
Y = datos[:, 1]

#plt.scatter(X, Y)
# plt.show()

Y = np.log(Y)
'''plt.scatter(X, Y)
plt.show()'''

X = X-X.mean()

modelo = tf.keras.models.Sequential(
    [tf.keras.layers.Input(shape=(1,)), tf.keras.layers.Dense(1)])

modelo.compile(optimizer=tf.keras.optimizers.SGD(
    learning_rate=0.001, momentum=0.9), loss="mse")


def schedule(epoch, lr):  # Esto te sirve para ir cambiando el learning rate en funcion de los epoch
    if epoch >= 50:
        return 0.0001
    return 0.001


scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

r = modelo.fit(X, Y, epochs=150, callbacks=[scheduler])

plt.plot(r.history["loss"], label="loss")
plt.show()


prediccion = modelo.predict(X).flatten()
plt.scatter(X, Y)
plt.plot(X, prediccion)
plt.show()

modelo.save("modelo.h5")
