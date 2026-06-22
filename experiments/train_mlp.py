""" [...] """
from pathlib import Path
from makemore import data, neural, neural_mlp

import numpy as np

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
NAMES_PATH = PROJECT_ROOT / "data" / "names.txt"

SEED = 2147483647

# ``CONTEXT_SIZE`` is an hyperparameter that tells  
# how many previous characters the model is allowed to look at. 
CONTEXT_SIZE = 3

# ``N_EMB`` is an hyperparameter that tells how many dimension
# has each embedding.
N_EMB = 2

HIDDEN_LAYER_OUTPUTS = 100

# The ``LEARNING_RATE`` hyperparameter is roughly choosen;
# there could be used the same "optimization" approach used
# by Karpathy in its video (makemore part2), but this i
# temporarily ``out of scopes``.
LEARNING_RATE = 0.1
EPOCHS = 500


def main():

    # === Build the Dataset ===
    raw_dataset = data.read_dataset(NAMES_PATH)
    stoi, itos = data.build_vocab(raw_dataset)
    X, Y = data.build_dataset(raw_dataset, stoi, CONTEXT_SIZE)

    alphabet_len = len(stoi)    # Vocabulary size: V

    # === Parameters Creation ===
    rng = np.random.default_rng(SEED)
    parameters = neural_mlp.init_params(alphabet_len, N_EMB, CONTEXT_SIZE, HIDDEN_LAYER_OUTPUTS, rng)

    # === TRAINING LOOP ===
    neural_mlp.train(EPOCHS, LEARNING_RATE, alphabet_len, CONTEXT_SIZE, N_EMB,
                     X, Y, parameters, 10)

    
if __name__ == "__main__":
    main()