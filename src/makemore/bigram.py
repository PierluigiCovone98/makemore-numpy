"""Character-Level language model that process and generate text one character at time (autoregressive).
Text files are assumed to be "one word per line".
"""
from pathlib import Path
from collections.abc import Iterator

import numpy as np
import matplotlib.pyplot as plt


# The "TXT" file become a list of strings.
type StrToInt =  dict[str, int]
type IntToStr =  dict[int, str]


# Constants
BOUNDARY_TOKEN: str = '.'
# To avoid the presence of bigrams probabilities values equals to zero,
# we use a value ``SMOOTHING`` (that become a hyperparameter of the model) 
# such that no bigrams have the count at zero.
SMOOTHING: int = 1


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


def iter_bigrams( dataset: list[str], stoi: StrToInt) -> Iterator[ tuple[int, int] ]:
    """Yield (ix1, ix2) for every bigram in the boundary-padded dataset."""
    
    for word in dataset:
        bounded_word = BOUNDARY_TOKEN + word + BOUNDARY_TOKEN
        for ch1, ch2 in zip(bounded_word, bounded_word[1:]):
            yield ( stoi[ch1], stoi[ch2] )


def count_bigrams( dataset: list[str], stoi: StrToInt ) -> np.ndarray:
    """For a given dataset, returns the bigrams counts matrix."""

    # Prepare the empty matrix
    N = np.zeros( (len(stoi), len(stoi)), dtype=np.int32 )
    
    for ix1, ix2 in iter_bigrams(dataset, stoi):
        N[ ix1, ix2 ] += 1

    return N


def to_probability_matrix( counts_matrix: np.ndarray ) -> np.ndarray:
    """Assumes a non-negative count matrix.
    From that, computes and returns the relative matrix of probabilities.
    """

    # Sum the "count_matrix" over "axis one" means: 
    #   "I want that, for each column j, you sum all elements of the i-th row".
    #
    # The "keepdims=True" avoids that the resulting vector has one dimension
    # less (this is important when broadcasting is performed).
    #
    # It's perfomed the "smoothing" of the bigrams counts matrix such that 
    # no more counts equals to zero, are present (and so: there are no zero
    # probabilities). 

    smoothed = counts_matrix + SMOOTHING 
    return smoothed / smoothed.sum( axis=1, keepdims=True )


def sample( P: np.ndarray, itos: IntToStr, rng: np.random.Generator ) -> str:
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
        # Notice that the "np.random.multinomial()" function returns an array
        # that has the same dimension of the given "pvals", and each dimension
        # has the same size of the original one.
        #
        # What "multinomial" does is:
        #   Ok, you specify how many time you want me to perform the experiment
        #   (exactly n-times) and I will tell you how frequent each result occurs.
        # How frequent results occur shuld roughly match the given distribution of
        # probabilities. 
        #
        # Because we're interested in one repetition of the experiment, we need a way
        # to "extract" the only outcome that occurred. To do that, we use "np.argmax()".
        j = np.argmax( rng.multinomial(n=1, pvals=P[i]) ).item() 
        
        # If the next character is the boundary token, then return (without add it)
        if j == 0:
            return ''.join(out)


        out.append( itos[j] )
        
        i = j

    
def mean_nll( dataset: list[str], stoi: StrToInt, P: np.ndarray ) -> float:
    """Compute the mean negative log likelihood of the bigram language model."""

    nll: float = 0.0
    n_bigrams: int = 0

    for ix1, ix2 in iter_bigrams(dataset, stoi):
        nll += -np.log( P[ix1, ix2] )
        n_bigrams += 1

    return nll/n_bigrams


def build_training_set(dataset: list[str], stoi: StrToInt) -> tuple[np.ndarray, np.ndarray]:
    """Build (xs, ys) parallel arrays of bigram indices from the dataset."""

    xs, ys = [], []

    for ix1, ix2 in iter_bigrams(dataset, stoi):
        xs.append(ix1)
        ys.append(ix2)

    return ( np.array(xs, dtype=np.int32), np.array(ys,  dtype=np.int32) )


def one_hot( indices: np.ndarray, num_classes: int ) -> np.ndarray:
    """Convert a vector of indices (input vector) into a matrix where each row 
    is an input encoded as one-hot vector.
    """
     
    # It is more efficient to allocate the final numpy array once
    output = np.zeros( (indices.size, num_classes), dtype=np.float32 )
    
    for i, ix in enumerate(indices):
        output[i, ix] = 1

    return output


# === RENDERING ===
def save_bigrams_matrix_plot( N: np.ndarray, itos: IntToStr, output_path: Path ) -> None:
    """Render the bigram count matrix as a heatmap. From Karpathy's makemore 1."""
    
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