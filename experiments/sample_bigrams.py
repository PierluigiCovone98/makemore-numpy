"""Sample names from the bigram probability matrix learned on names.txt."""
from pathlib import Path
from makemore import bigram
import numpy as np

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
NAMES_PATH = PROJECT_ROOT / "data" / "names.txt"
NUM_SAMPLES = 20
SEED = 42

def main():

    # 1. Read the dataset
    dataset = bigram.read_dataset(NAMES_PATH)
    
    # 2. Create the alphabet
    stoi, itos = bigram.build_vocab(dataset)

    # 3. Create the bigrams counts matrix
    N = bigram.count_bigrams(dataset, stoi)

    # 4. Transform it to a probabilities matrix 
    P = bigram.to_probability_matrix(N)

    # 5. Sample names from P, using a single rng seeded once for reproducibility.
    rng = np.random.default_rng(SEED)
    for _ in range(NUM_SAMPLES):
        print(bigram.sample(P, itos, rng))


if __name__ == "__main__":
    main()