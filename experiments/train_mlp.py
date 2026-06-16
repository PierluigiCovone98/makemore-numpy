""" [...] """
from pathlib import Path
from makemore import data

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
NAMES_PATH = PROJECT_ROOT / "data" / "names.txt"

# ``BLOCK_SIZE`` is the single new hyperparameter of the MLP: 
# how many previous characters the model is allowed to look at. 
BLOCK_SIZE = 3


def main():

    # Build the dataset with a context window of BLOCK_SIZE characters.
    dataset = data.read_dataset(NAMES_PATH)
    stoi, itos = data.build_vocab(dataset)
    X, Y = data.build_dataset(dataset, stoi, BLOCK_SIZE)

    # Log
    # for i in range(32):
    #     print(f"{''.join(itos[x] for x in X[i])} -> {itos[Y[i]]}")

if __name__ == "__main__":
    main()