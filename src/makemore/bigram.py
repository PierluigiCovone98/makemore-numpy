"""Character-Level language model that process and generate text one character at time (autoregressive).
Text files are assumed to be "one word per line".
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# The "TXT" file become a list of strings.
type StrToInt =  dict[str, int]
type IntToStr =  dict[int, str]


# Constants
BOUNDARY_TOKEN: str = '.'


def read_dataset( dataset_path: Path ) -> list[str]:
    """Read the dataset transforming it in a list of strings."""

    return dataset_path.read_text().splitlines()


def build_vocab( dataset: list[str] ) -> tuple[StrToInt, IntToStr]:
    """Takes in input the read dataset (list of strings) and returns
    dictionaries 'StrToInt' and 'IntToStr', that are required to encode a character to its integer
    and an integer to the corresponding character, respectively.
    
    In addition to the alphabet retrieved from the dataset, a special token '.' is inserted
    to mark the start and the end of a word; the corresponding integer to it is: 0.

    The first element that is returned is "stoi";
    the second one is "itos".
    """

    sorted_alphabet = sorted( set(''.join(dataset)) )

    stoi : StrToInt= {}
    itos : IntToStr = {}

    for i, s in enumerate(sorted_alphabet, start=1):
        stoi[s] = i
        itos[i] = s

    # We force the '.' special token to be mapped to '0' (and vice-versa) because
    # without this, if a character appearing before '.' in ASCII ordering were added, 
    # our mapping would break.
    stoi[BOUNDARY_TOKEN] = 0
    itos[0] = BOUNDARY_TOKEN

    return (stoi, itos)


def save_bigrams_matrix_plot( N: np.ndarray, itos: IntToStr, output_path: Path ) -> None:
    """Render the bigram count matrix as a heatmap. From Karpathy's makemore 1."""
    
    plt.figure(figsize=(16, 16))
    plt.imshow(N, cmap='Blues')

    # Different formats for different types of matrices.
    fmt = "d" if np.issubdtype(N.dtype, np.integer) else ".2f"

    n = N.shape[0]
    for i in range(n):
        for j in range(n):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
            plt.text(j, i, f"{N[i, j].item():{fmt}}", ha="center", va="top", color="gray")

    plt.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)


def to_probability_matrix( counts_matrix: np.ndarray ) -> np.ndarray:
    """Assumes a non-negative count matrix.
    From that, computes and returns the relative matrix of probabilities.
    """

    # Sum the "count_matrix" over axis one means: 
    #   "I want that for each column j, you sum all elements of the i-th row".
    # The "keepdims=True" avoids that the resulting vector has one dimension
    # less (this is important when broadcasting is performed).
    return counts_matrix / counts_matrix.sum( axis=1, keepdims=True )