"""Character-Level language model that process and generate text one character at time (autoregressive).
Text files are assumed to be "one word per line".
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# The "TXT" file become a list of strings.
type Stoi =  dict[str, int]
type Itos =  dict[int, str]


# Constants
BOUNDARY_TOKEN: str = '.'


def read_dataset( dataset_path: str ) -> list[str]:
    """Read the dataset transforming it in a list of strings."""

    # "Path" is more useful than "OS" (for me)
    path = Path(dataset_path)

    return path.read_text().splitlines()


def build_vocab( dataset: list[str] ) -> tuple[Stoi, Itos]:
    """Takes in input the read dataset (list of strings) and returns
    dictionaries 'stoi' and 'itos', that are required to encode a character to its integer
    and an integer to the corresponding character, respectively.
    
    In addition to the alphabet retrieved from the dataset, a special token '.' is inserted
    to mark the start and the end of a word; the corresponding integer to it is: 0.

    The first element that is returned is "stoi";
    the second one is "itos".
    """

    sorted_alphabet = sorted( set(''.join(dataset)) )

    stoi : Stoi = {}
    itos : Itos = {}

    # I do not expect "len_alphabet" to be high.
    for i, s in enumerate(sorted_alphabet, start=1):
        stoi[s] = i
        itos[i] = s

    # We force the '.' special token to be mapped to '0' (and vice-versa)
    # such that, when there will be used more characters that comes before 
    # in the ASCII ordering, our mapping criteria does not work anymore.
    stoi[BOUNDARY_TOKEN] = 0
    itos[0] = BOUNDARY_TOKEN

    return (stoi, itos)


def pretty_format_bigrams_matrix(N: np.ndarray, itos: Itos) -> None:
    """Render the bigram count matrix as a heatmap. From Karpathy's makemore 1."""
    plt.figure(figsize=(16, 16))
    plt.imshow(N, cmap='Blues')

    n = N.shape[0]
    for i in range(n):
        for j in range(n):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")  # <-- top

    plt.axis("off")

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / "bigrams_matrix.png")


def main():
    # Let's reproduce what Karpathy does here.
    
    # 1. Let's first read the dataset.
    dataset = read_dataset("data/names.txt")
    
    # Create converters
    stoi, itos = build_vocab(dataset)

    # Prepare the empty matrix
    N = np.zeros( (len(stoi), len(stoi)), dtype=np.int32 )

    # For each word in the dataset...
    for word in dataset:
        # ... create the token
        bounded_word = BOUNDARY_TOKEN + word + BOUNDARY_TOKEN
        # And for each bigram...
        for ch1, ch2 in zip(bounded_word, bounded_word[1:]):
            # ... convert each char into the corresponding int value.
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]

            N[ix1, ix2] += 1

    pretty_format_bigrams_matrix(N, itos)

    
if __name__ == "__main__":
    main()