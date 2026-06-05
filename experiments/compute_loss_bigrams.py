"""Compute the loss value for the bigram language model.
The average negative log likelihood is our loss function.
"""
from pathlib import Path
from makemore import bigram

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
NAMES_PATH = PROJECT_ROOT / "data" / "names.txt"

def main():

    # 1. Read the dataset
    dataset = bigram.read_dataset(NAMES_PATH)
    
    # 2. Create the alphabet
    stoi, itos = bigram.build_vocab(dataset)

    # 3. Create the bigrams counts matrix
    N = bigram.count_bigrams(dataset, stoi)

    # 4. Transform it to a probabilities matrix 
    P = bigram.to_probability_matrix(N)

    # Invoke the loss function computation
    print( bigram.mean_nll(dataset, stoi, P) )


if __name__ == "__main__":
    main()


