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


def read_dataset( dataset_path: Path ) -> list[str]:
    """Read the dataset transforming it in a list of strings."""

    return dataset_path.read_text().splitlines()


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


def pretty_format_bigrams_matrix(N: np.ndarray, itos: Itos, output_path: Path) -> None:
    """Render the bigram count matrix as a heatmap. From Karpathy's makemore 1."""
    plt.figure(figsize=(16, 16))
    plt.imshow(N, cmap='Blues')

    n = N.shape[0]
    for i in range(n):
        for j in range(n):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")

    plt.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)