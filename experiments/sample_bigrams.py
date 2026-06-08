"""Sample names from the bigram probability matrix learned on names.txt."""
from pathlib import Path
from makemore import counting, data

import numpy as np

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
NAMES_PATH = PROJECT_ROOT / "data" / "names.txt"

NUM_SAMPLES = 10
SEED = 2147483647

def main():

    # 1. Read the dataset
    dataset = data.read_dataset(NAMES_PATH)
    
    # 2. Create the alphabet
    stoi, itos = data.build_vocab(dataset)

    # 3. Create the bigrams counts matrix
    N = counting.count_bigrams(dataset, stoi)

    # 4. Transform it to a probabilities matrix 
    P = counting.to_probability_matrix(N)

    # 5. Sample names from P, using a single rng seeded once for reproducibility.
    rng = np.random.default_rng(SEED)
    for _ in range(NUM_SAMPLES):
        print(counting.sample(P, itos, rng))


if __name__ == "__main__":
    main()