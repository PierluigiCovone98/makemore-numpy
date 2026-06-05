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
    # Build the bigram probability matrix.
    dataset = bigram.read_dataset(NAMES_PATH)
    stoi, itos = bigram.build_vocab(dataset)

    N = np.zeros((len(stoi), len(stoi)), dtype=np.int32)
    for word in dataset:
        bounded_word = bigram.BOUNDARY_TOKEN + word + bigram.BOUNDARY_TOKEN
        for ch1, ch2 in zip(bounded_word, bounded_word[1:]):
            N[stoi[ch1], stoi[ch2]] += 1

    P = bigram.to_probability_matrix(N)

    # Sample names from P, using a single rng seeded once for reproducibility.
    rng = np.random.default_rng(SEED)
    for _ in range(NUM_SAMPLES):
        print(bigram.sample(P, itos, rng))


if __name__ == "__main__":
    main()