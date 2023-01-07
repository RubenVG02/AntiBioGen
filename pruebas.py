import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import random

path_dades=r"C:\Users\ASUS\Desktop\github22\dasdsd\xab.txt"
def split_input_target(valors):
    input_text = valors[:-1]
    target_idx = valors[-1]
    target = tf.one_hot(target_idx, depth=mapa_char-2)  #depth ha de ser igual al numero d'outputs diferents del model pre-entrenat
    target = tf.reshape(target, [-1])
    return input_text, target

with open(f"{path_dades}") as f:
    dades = "\n".join(line.strip() for line in f)

# per obtenir els elements unics de dades
elements_smiles = {u: i for i, u in enumerate(sorted(set(dades)))}
elements_smiles.update({-1: "\n"})

# per passar els elements unics de dades a
int_a_elements = {i: c for i, c in enumerate(elements_smiles)}
int_a_elements.update({"\n": -1})

mapa_int = len(elements_smiles)
mapa_char = len(int_a_elements)



slices = np.array([[elements_smiles[c]] for c in dades])

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(slices)

sequences = char_dataset.batch(137+1, drop_remainder=True)

dataset = sequences.map(split_input_target)

dataset = dataset.shuffle(20000).batch(256, drop_remainder=True)

generador_seeds=tfds.as_numpy(dataset.take(random.randint(0, len(dades))).take(random.randint(0,2)))
for x,i in enumerate(generador_seeds):
    print(x,i)

aaa=i[0][np.random.randint(0, 127)]

