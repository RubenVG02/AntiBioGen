import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

# esto es para escalar de 0-1, recuerda que los x son 28x28xN
X_train, X_test = X_train/255, X_test/255

modelo = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(
    128, "relu"), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(10, activation="softmax")])  # Como es multiclass, usas softmax para clasficar, no puedes usar relu pq es para bool
# El 10 es pq hay 10 digitos

# El dropout te sirve para establecer nodos como 0 aleatoriamente

modelo.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy", metrics=["accuracy"])
r = modelo.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10)

plt.plot(r.history["loss"], label="loss")
plt.plot(r.history["val_loss"], label="val_loss")
plt.plot(r.history["accuracy"], label="acc")
plt.plot(r.history["val_accuracy"], label="val_acc")
plt.legend()
plt.show()
