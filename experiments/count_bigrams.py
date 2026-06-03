"""Count bigrams in the ``data/names.txt`` dataset and plot the corresponding
matrix in ``output/bigrams_matrix.png`` file.
"""
from pathlib import Path
from makemore import bigram
import numpy as np

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "names.txt"
OUTPUT_PATH = PROJECT_ROOT / "output" / "bigrams_matrix.png"

def main():    
    # Let's first read the dataset.
    dataset = bigram.read_dataset(DATA_PATH)
    
    # Create converters
    stoi, itos = bigram.build_vocab(dataset)

    # Prepare the empty matrix
    N = np.zeros( (len(stoi), len(stoi)), dtype=np.int32 )

    # For each word in the dataset...
    for word in dataset:
        # ... create the token
        bounded_word = bigram.BOUNDARY_TOKEN + word + bigram.BOUNDARY_TOKEN
        # And for each bigram...
        for ch1, ch2 in zip(bounded_word, bounded_word[1:]):
            # ... convert each char into the corresponding int value.
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]

            N[ix1, ix2] += 1

    bigram.pretty_format_bigrams_matrix(N, itos, OUTPUT_PATH)

    
if __name__ == "__main__":
    main()