# Antibiotics Discovery

## Overview

This project focuses on discovering and designing new antibiotic compounds based on target proteins. It includes tools for predicting toxicity, generating antibiotic compounds, and obtaining 3D structures of designed molecules. The project utilizes neural networks for toxicity prediction and antibiotic generation, alongside genetic algorithms to enhance the design process.

## Usage

### Preparing Data

1. **FASTA Sequence**: Ensure your target protein is in FASTA format (amino acid sequence).

2. **Neural Networks**: You need two neural networks:
   - **Toxicity Prediction**: Use `cnn_affinity.py` to train or utilize the pre-trained model.
   - **Antibiotic Generation**: Use `generate_rnn.py` to train or utilize the pre-trained model.

   Alternatively, use the pre-trained models located in the `definitive_models` folder.

3. **Data**: Use data from databases such as Chembl, PubChem, or other relevant sources.

### CNN Usage (Affinity Prediction)

To predict toxicity using the CNN model, use `check_affinity.py`. You need to specify the path to the model, the path to the data, and the path to the target protein. The program will return the toxicity of the designed bioinsecticides using the `calculate_affinity` function.

### RNN USAGE (Generation) ###

To generate antibiotic compounds using the RNN model, use `pretrained_rnn.py`. You need to specify the path to the model, the path to the data, and the path to the target protein. The program will return the designed bioinsecticides using the generate function.

### Combination of Models ###

For combining both models (generation and toxicity prediction), use `affinity_with_target_and_generator.py`. You need to specify the path to the model, the path to the data, and the path to the target protein. The program will generate antibiotic compounds and filter out those exceeding the specified toxicity limit. You can also specify a path to check generated molecules.



## Installation

Via Git Clone:
```bash

git clone https://github.com/RubenVG02/AntibioticsDiscovery.git

```

Via Lastest Release:

```bash

https://github.com/RubenVG02/AntibioticsDiscovery/releases

```

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




