import tensorflow as tf
import tensorflow_datasets as tfds

from keras.layers import CuDNNLSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import BatchNormalization

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw

import numpy as np
import sys
import random
import time




def generador(path_model=r"C:\Users\ASUS\Desktop\github22\dasdsd\nuevos_modelos\modelo_prueba_rnn_aversiva.hdf5", path_dades=r"C:\Users\ASUS\Desktop\github22\dasdsd\xab.txt",
              nombre_generats=200, img_druglike=True, path_desti_molecules=r"C:\Users\ASUS\Desktop\github22\dasdsd\moleculas_generadas//moleculas_nuevo_generador/moleculas_druglike2.txt"):
    '''
        Paràmetres:
        -path_model: Path on es troba el model ja entrenat
        -path_dades: Path on es troben les dades utlitzades (path model i path dades han de tenir les mateixes dimensions/mateixa quantitat d'elements diferents)
        -nombre_generats: Nombre de molècules que es generen, per predeterminat 100
        -img_druglike: Per generar imatges .jpg de les molècules drug-like generades (estaran ordenades en funció de epoch) (Per default True)
        -Path_desti_molecules: Path de destí de les seqüencies SMILE generades
    '''
    def split_input_target(valors):
        input_text = valors[:-1]
        target_idx = valors[-1]
        target = tf.one_hot(target_idx, depth=mapa_char-2)  #depth ha de ser igual al numero d'outputs diferents del model pre-entrenat
        target = tf.reshape(target, [-1])
        return input_text, target
    
    def crear_seeds(maxim_molecules=137):
        '''
        Funció que agafa el teu ds i permet obtenir una seed d'aquest per generar molecules
        
        Paràmetres:
            -maxim_molecules: Longitut maxima que vulguis que tingui el teu pattern/seed, per default, 137
        
        '''
        generador_seeds=tfds.as_numpy(dataset.take(random.randint(0, len(dades))).take(1))
        for a, b in enumerate(generador_seeds):
            break
        pattern=b[0][np.random.randint(0,maxim_molecules)]
        return pattern

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
    

    

    def crear_model():
        modelo = tf.keras.models.Sequential([CuDNNLSTM(128, input_shape=(137, 1), return_sequences=True),
                                            Dropout(0.1),
                                            CuDNNLSTM(
                                                256, return_sequences=True),
                                            BatchNormalization(),
                                            Dropout(0.1),
                                            CuDNNLSTM(
                                                512, return_sequences=True),
                                            BatchNormalization(),
                                            Dropout(0.1),
                                            CuDNNLSTM(
                                                256, return_sequences=True),
                                            BatchNormalization(),
                                            Dropout(0.1),
                                            CuDNNLSTM(128),
                                            Dropout(0.1),
                                            Dense(mapa_char-2, activation="softmax")])
        return modelo

    modelo = crear_model()
    modelo.load_weights(f"{path_model}")
    modelo.compile(loss='categorical_crossentropy', optimizer='adam')

    ###GENERACIÓ DE MOLÈCULES###
    seq_length = 137
    pattern=crear_seeds(maxim_molecules=seq_length)
    print("\"", ''.join([int_a_elements[value[0]] for value in pattern]), "\"")
    final = ""
    total_smiles = []
    for i in range(nombre_generats):
        for i in range(random.randrange(50,137)):
            x = np.reshape(pattern, (1, len(pattern)))
            predicció = modelo.predict(x, verbose=0)
            index = np.argmax(predicció)  #Agafar el valor màxim de l'array de predicció
            resultat = int_a_elements[index]
            print(resultat, end="")
            final += resultat
            pattern=np.append(pattern, index)
            pattern = pattern[1:len(pattern)]
        final = final.split("\n")
        for i in final:
            mol1 = Chem.MolFromSmiles(i)
            if len(i) > 20:
                if mol1 == None:
                    print("error")
                elif not mol1 == None:
                    print(resultat)
                    print("Ha sortit una molècula químicament possible, miro si és drug-like")
                    if Descriptors.ExactMolWt(mol1) < 500 and Descriptors.MolLogP(mol1) < 5 and Descriptors.NumHDonors(mol1) < 5 and Descriptors.NumHAcceptors(mol1) < 10:  
                        with open(f"{path_desti_molecules}", "a") as file:
                            with open(f"{path_desti_molecules}", "r") as f:
                                linies = [linea.rstrip() for linea in f]
                            if f"{i}" not in linies:
                                file.write(i + "\n")
                                if img_druglike == True:
                                    Draw.MolToImageFile(
                                mol1, filename=fr"C:\Users\ASUS\Desktop\github22\dasdsd\moleculas_generadas\moleculas_nuevo_generador/molecula{int(time.time())}.jpg", size=(400, 300))
                        total_smiles.append(i)
                        print("La molècula és drug-like")
            else:
                pass
        final = ""
    return total_smiles

generador()