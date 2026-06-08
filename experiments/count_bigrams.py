"""Count bigrams in the ``data/names.txt`` dataset and plot the corresponding
matrix in ``output/bigrams_matrix.png`` file.
"""
from pathlib import Path
from makemore import counting, data

import numpy as np

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
NAMES_PATH = PROJECT_ROOT / "data" / "names.txt"
BIGRAM_COUNTS_PATH = PROJECT_ROOT / "output" / "bigram_counts_matrix.png"
BIGRAM_PROBS_PATH = PROJECT_ROOT / "output" / "bigram_probs_matrix.png"

def main():    
      
    # 1. Read the dataset
    dataset = data.read_dataset(NAMES_PATH)
    
    # 2. Create the alphabet
    stoi, itos = data.build_vocab(dataset)

    # 3. Create the bigrams counts matrix
    N = counting.count_bigrams(dataset, stoi)

    # 4. Transform it to a probabilities matrix 
    P = counting.to_probability_matrix(N)

    # 5. Save both matrices
    counting.save_bigrams_matrix_plot(N, itos, BIGRAM_COUNTS_PATH)
    counting.save_bigrams_matrix_plot(P, itos, BIGRAM_PROBS_PATH)


if __name__ == "__main__":
    main()