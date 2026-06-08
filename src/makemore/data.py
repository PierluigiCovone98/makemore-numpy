"""Collection of functions for ``data preparation`` for the problem of 
predicting the next character in a sequence (bigram problem). 
In particular:
    dataset loading, vocabulary building, bigram extraxtion and one-hot encoding.

Functions implemented here are used by more than one single approach (by the counting 
and the neural models).
"""
from pathlib import Path
from collections.abc import Iterator

import numpy as np


# Character-level vocabulary types.
type StrToInt = dict[str, int]
type IntToStr = dict[int, str]


# Constants
BOUNDARY_TOKEN: str = '.'


def read_dataset( dataset_path: Path ) -> list[str]:
    """Read the dataset from a file, transforming it in a list of strings."""
    return dataset_path.read_text().splitlines()


def build_vocab( dataset: list[str] ) -> tuple[StrToInt, IntToStr]:
    """Build vocabulary mappings (the vocab) from the given dataset.

    In addition to the alphabet of the dataset, it is inserted into the vocab
    a special token ``.`` too, that bounds a word into a pair of dots.

    The used convention is to manually map the ``boundary toke`` such that
    it is always mapped to the ``0`` integer.

    The first element of the returned tuple is the String-To-Integer mapping;
    the second one is the Integer-To-String mappping.
    """
    alphabet = sorted( set( ''.join(dataset) ) )

    stoi: StrToInt = {} 
    itos: IntToStr = {}

    for i, ch in enumerate(alphabet, start=1):
        stoi[ch] = i
        itos[i] = ch

    # We force the special token to be mapped to '0' by convention;
    # without this, if a character appearing before our special token
    # is added to the vocab, because of the ASCII ordering, our 
    # mapping would break.
    stoi[BOUNDARY_TOKEN] = 0
    itos[0] = BOUNDARY_TOKEN
    
    return (stoi, itos)


def iter_bigrams( dataset: list[str], stoi: StrToInt ) -> Iterator[ tuple[int, int] ]:
    """Yield (ix1, ix2) for every bigram in the boundary-padded dataset, where:
        
        - ix1   is the integer corresponding to the first character of the bigram,
                according to the vocab.
        - ix2   is the integer corresponding to the second character of the bigram,
                according to the vocab.       
    """
    for word in dataset:
        bounded_word = BOUNDARY_TOKEN + word + BOUNDARY_TOKEN
        for ch1, ch2 in zip(bounded_word, bounded_word[1:]):
            yield ( stoi[ch1], stoi[ch2] )


def build_training_set( dataset: list[str], stoi: StrToInt ) -> tuple[np.ndarray, np.ndarray]:
    """For a given dataset, returns the training set for the "next character prediction problem.
    
    The training set is composed by the ``inputs vector`` and the ``targets vector``:
    both are vectors of integers representing the index of a specific character, 
    according to the defined vocab.
    """
    xs, ys = [], []

    for ix1, ix2 in iter_bigrams(dataset, stoi):
        xs.append(ix1)
        ys.append(ix2)

    return ( np.array(xs, dtype=np.int32), np.array(ys,  dtype=np.int32) )


def one_hot( indices: np.ndarray, num_classes: int ) -> np.ndarray:
    """Convert a vector of indices into a matrix where each row 
    is an input encoded as ``one-hot vector``.
    """
     
    # Efficiency reasons motivate the pre-allocation of the entire econding.
    output = np.zeros( (indices.size, num_classes), dtype=np.float32 )
    
    for i, ix in enumerate(indices):
        output[i, ix] = 1

    return output