# Antibiotics Discovery


## Usage

First of all, to make a good prediction, the target proteins must be in the FASTA (amino acids) sequence.

Then, you need 2 neural networks, one for toxicity prediction and another to generate antibiotic compounds. You can train them yourself using `cnn_affinity.py` for toxicity prediction and 'generate_rnn.py' for antibiotic generation. You can also use the ones I have trained, which are in the `definitive_models` folder.

To train your models you need data, which can be obtained from different databases such as Chembl, Pubchem, etc.

### CNN USAGE (Affinity) ###
To use the CNN, use `check_affinity.py`. You need to specify the path to the model, the path to the data, and the path to the target protein. The program will return the toxicity of the designed bioinsecticides using the `calculate_affinity` function.

### RNN USAGE (Generation) ###

To use the RNN, use `pretrained_rnn.py`. You need to specify the path to the model, the path to the data, and the path to the target protein. The program will return the designed bioinsecticides using the generate function.

### COMBINATION ###

To use the combination of both models, use `affinity_with_target_and_generator.py`. You need to specify the path to the model, the path to the data, and the path to the target protein. The program will return the designed bioinsecticides using the generate function. You can also specify the toxicity limit of the designed bioinsecticides using the `calculate_affinity` function. The program will return the designed bioinsecticides with a lower toxicity than the limit. You can also specify a path of generated molecules to check.



## Installation

To use this project, you need to have Python 3.7 or higher installed. Then, you need to install the following libraries:
- Keras
- Tensorflow
- Numpy
- Pandas
- Matplotlib

To install the required libraries, use:

```bash
pip install requirements.txt
```


## Authors

- [@RubenVG02](https://www.github.com/RubenVG02)

## Features

- Design of new antibiotic compounds based on the target protein
- Predicting the toxicity of the designed antibiotics
- Obtaining CSV files and screenshots of the results
- Fast and easy to use

## Future Improvements

- Add more databases to the CNN
- Add more databases to the RNN
- Use GA to improve the efficiency of the generation
- Directly obtain the 3D structure of the designed compounds


## License

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgements

- [Keras](https://keras.io/)
- [Tensorflow](https://www.tensorflow.org/)
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)


