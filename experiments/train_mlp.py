""" [...] """
from pathlib import Path
from makemore import data, neural

import numpy as np

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
NAMES_PATH = PROJECT_ROOT / "data" / "names.txt"

SEED = 2147483647

# ``BLOCK_SIZE`` is an hyperparameter that tells  
# how many previous characters the model is allowed to look at. 
BLOCK_SIZE = 3

# ``N_EMB`` is an hyperparameter that tells how many dimension
# has each embedding.
N_EMB = 2


def main():

    # --- Dataset ---
    # Load the raw words, build the vocabulary, then build the (X, Y) dataset
    # with a context window of BLOCK_SIZE characters:
    #   X: (N, BLOCK_SIZE) int  -- each row is one context (BLOCK_SIZE indices)
    #   Y: (N,)            int  -- each entry is the next-character index to predict
    raw_dataset = data.read_dataset(NAMES_PATH)
    stoi, itos = data.build_vocab(raw_dataset)
    X, Y = data.build_dataset(raw_dataset, stoi, BLOCK_SIZE)

    # Vocabulary size V
    alphabet_len = len(stoi)

    # --- Optional Logs ---
    # Read the first examples back as (context -> target) to verify the window:
    # for i in range(5):
    #     print(f"{''.join(itos[x] for x in X[i])} -> {itos[Y[i]]}")
    #
    # print(X[4, 2])   # expected: 1

    
    # --- Embedding lookup ---
    # Embedding forward pass: create the look-up table C, 
    # the first (linear) layer of the MLP.
    # Then map every index in X to its embedding row.
    rng = np.random.default_rng(SEED)

    # C is a learnable parameter, exactly like W in the bigram.
    C = neural.build_layer(alphabet_len, N_EMB, rng)
    embeddings = neural.embed(X, C)

    # Expected shape: (N, BLOCK_SIZE, N_EMB) = (N, 3, 2).
    #
    # It is NOT (N, 27, 2): 27 is the number of characters, so the numberw of
    # orws in C. The lookup process seleects ``BLOCK_SIZE`` of those rows 
    # per example.
    print(embeddings.shape)

    
if __name__ == "__main__":
    main()