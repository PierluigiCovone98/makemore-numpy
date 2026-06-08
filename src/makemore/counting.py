"""Counting-based (frequentist) bigram model: estimate probabilities directly from observed bigram counts.
Text files are assumed to be "one word per line".
"""
import numpy as np
import matplotlib.pyplot as plt

from makemore import data
from pathlib import Path


# Constants
SMOOTHING: int = 1


def count_bigrams( dataset: list[str], stoi: data.StrToInt ) -> np.ndarray:
    """For a given dataset, returns the bigrams counts matrix."""
    N = np.zeros( (len(stoi), len(stoi)), dtype=np.int32 )
    
    for ix1, ix2 in data.iter_bigrams(dataset, stoi):
        N[ ix1, ix2 ] += 1

    return N


def to_probability_matrix( counts_matrix: np.ndarray ) -> np.ndarray:
    """Assumes a non-negative count matrix.
    computes and returns the relative matrix of probabilities.
    """
    # Sum the "count_matrix" over "axis one" means: 
    #   "I want that, for each column j, you sum all elements of the i-th row".
    #
    # The "keepdims=True" avoids that the resulting vector has one dimension
    # less (this is important when broadcasting is performed).
    # It's a simple normalization over rows.
    #
    # Notice that it's also perfomed the "smoothing" of the bigrams counts matrix 
    # such that no more counts equals to zero are present in the matrix (and so: 
    # there are no zero probabilities). 
    smoothed = counts_matrix + SMOOTHING 
    return smoothed / smoothed.sum( axis=1, keepdims=True )


def sample( P: np.ndarray, itos: data.IntToStr, rng: np.random.Generator ) -> str:
    """Sample a new word from the bigrams probabilities matrix.
    
    Inputs
    ------
        P : The bigrams probabilities matrix. We use it by rows. 

        itos : The map between integers and corresponding strings.

    Returns
    -------
        The sample retrieved from the sampling operation.
    """
    out : list[str] = [] 

    i : int = 0
    while True:
        
        # Using "multinomial" because we want to get a character following 
        # the probability distribution of the i-th row.
        # 
        # Notice that the "rng.multinomial()" function returns an array
        # that has the same dimension of the given "pvals", and each dimension
        # has the same size of the original one.
        #
        # What "multinomial" does is:
        #   Ok, you specify how many time you want me to perform the experiment
        #   (exactly n-times) and I will tell you how frequent each result occurs.
        # How frequent results occur shuld roughly match the given distribution of
        # probabilities. 
        #
        # Because we're interested in one repetition of the experiment (n=1),
        # we need a way to "extract" the only outcome that occurred. 
        # To do that, we use "np.argmax()".
        j = np.argmax( rng.multinomial(n=1, pvals=P[i]) ).item() 
        
        # If the next character is the boundary token, then return (without add it)
        if j == 0:
            return ''.join(out)

        out.append( itos[j] )
        i = j

    
def mean_nll( dataset: list[str], stoi: data.StrToInt, P: np.ndarray ) -> float:
    """Compute the mean negative log likelihood of the bigram language model."""

    nll: float = 0.0
    n_bigrams: int = 0

    for ix1, ix2 in data.iter_bigrams(dataset, stoi):
        nll += -np.log( P[ix1, ix2] )
        n_bigrams += 1

    return nll/n_bigrams


# === RENDERING ===
def save_bigrams_matrix_plot( N: np.ndarray, itos: data.IntToStr, output_path: Path ) -> None:
    """Render the bigram count matrix as a heatmap."""
    
    plt.figure(figsize=(20, 20))
    plt.imshow(N, cmap='Blues')

    # Different formats for different types of matrices.
    fmt = "d" if np.issubdtype(N.dtype, np.integer) else ".4f"

    n = N.shape[0]
    for i in range(n):
        for j in range(n):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
            plt.text(j, i, f"{N[i, j].item():{fmt}}", ha="center", va="top", color="gray")

    plt.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)