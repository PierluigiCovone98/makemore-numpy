"""Collection of functions for ``data preparation`` for the problem of 
predicting the next character in a sequence (bigram problem). 
In particular:
    dataset loading, vocabulary building, bigram extraction and one-hot encoding.

Functions implemented here are used by more than one single approach (by the counting 
and the neural models).
"""
from pathlib import Path
from dataclasses import dataclass
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

    The used convention is to manually map the ``boundary token`` such that
    it is always mapped to the ``0`` integer.

    The first element of the returned tuple is the String-To-Integer mapping;
    the second one is the Integer-To-String mapping.
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

@dataclass
class SplitRawDataset:
    # Optimize parameters of the model (using gradient descent).
    raw_train_set: list[str]
    
    # Train hyperparameters of the model.
    raw_dev_set: list[str]
    
    # Evaluate the performance of the model at the end (only few times).
    raw_test_set: list[str]


def split_raw_dataset( raw_dataset: list[str],
                       train_frac: float, dev_frac: float,
                       rng: np.random.Generator) -> SplitRawDataset:
    """Partition the words dataset into train / dev / test sets.

    The words are shuffled first -- so the three sets share the same
    distribution regardless of any order in the source -- then cut at cumulative
    boundaries: ``train_frac`` of the words go to train, ``dev_frac`` to dev, and
    the remainder to test (so the test fraction is whatever the first two leave).

    The split is on *words*, before ``build_dataset``, so every example of a
    given word stays in a single set (no leakage across the split). ``rng`` makes
    the shuffle reproducible.
    """
    if not (0 < train_frac and 0 < dev_frac and train_frac + dev_frac < 1.0):
        raise ValueError(
            f"train_frac and dev_frac must be positive and sum to less than 1.0 "
            f"(the rest is the test fraction); got train_frac={train_frac}, dev_frac={dev_frac}."
        )

    raw_dataset_copy = raw_dataset[:]
    rng.shuffle(raw_dataset_copy)

    # Partitioning
    len_raw_dataset = len(raw_dataset_copy)

    upper_idx_tr = int(train_frac*len_raw_dataset)
    upper_idx_dev = upper_idx_tr + int(dev_frac*len_raw_dataset)

    raw_train_set = raw_dataset[:upper_idx_tr]
    raw_dev_set = raw_dataset[upper_idx_tr:upper_idx_dev]
    raw_test_set = raw_dataset[upper_idx_dev:]
    
    return SplitRawDataset(raw_train_set= raw_train_set,
                           raw_dev_set= raw_dev_set,
                            raw_test_set= raw_test_set)


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


def build_dataset_bigram( dataset: list[str], stoi: StrToInt ) -> tuple[np.ndarray, np.ndarray]:
    """For a given dataset, returns the dataset for the "next character prediction problem“.
    
    The dataset is composed by the ``inputs vector`` and the ``targets vector``:
    both are vectors of integers representing the index of a specific character, 
    according to the defined vocab.

    Notice that, because the "neural bigram" model uses all the example of the dataset
    as training elements, the dataset and the ``taining set`` are the same entity.

    Notice also that this method exists for retro-compatibility with already tested code, 
    but in the form it's equivalent to invoke the ``data.build_dataset`` function with
    "context_size=1"; the main difference is that here, the ``inputs vector X`` has shape
    (N,), while in the the ``data.build_dataset`` result it has shape (N, 1).
    """
    xs, ys = [], []

    for ix1, ix2 in iter_bigrams(dataset, stoi):
        xs.append(ix1)
        ys.append(ix2)

    return ( np.array(xs, dtype=np.int32), np.array(ys,  dtype=np.int32) )


def build_dataset( raw_dataset: list[str], stoi: StrToInt, context_size: int) -> tuple[np.ndarray, np.ndarray]:
    """For a given raw dataset, returns the corresponding dataset for the "next character prediction problem“.
    
    The dataset is composed by the ``inputs vector, X`` and the ``targets vector, Y``, where
    ``X`` is a vector of vectors, each of which have ``context_size`` integers that represent
    characters in the context, while ``Y`` is the vector of integers representing the index 
    of the ``actual character`` of the corresponding context.
    Notice that another term to name ``targets`` is: ``labels``.
    """
    X, Y = [], []
    
    for word in raw_dataset:

        # Initially a list of all 0s.
        context: list[int] = [0] * context_size

        right_bounded_word = word + BOUNDARY_TOKEN

        for ch in right_bounded_word:
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)

            # Update: crop and append.
            # Notice that, both the slice and the "+" operation
            # creates a new list (does not modify the existing one).
            # This was on purpose, avoiding the risk of modify
            # the already placed context. 
            context = context[1:] + [ix] 

    return  ( np.array(X, dtype=np.int32), np.array(Y,  dtype=np.int32) )


def one_hot( indices: np.ndarray, num_classes: int ) -> np.ndarray: 
    """Convert a vector of indices into a matrix where each row 
    is an input encoded as ``one-hot vector``.
    """
     
    # Efficiency reasons motivate the pre-allocation of the entire econding.
    output = np.zeros( (indices.size, num_classes), dtype=np.float32 )
    
    for i, ix in enumerate(indices):
        output[i, ix] = 1

    return output


def sample( probs: np.ndarray, itos: IntToStr, rng: np.random.Generator ) -> str:
    """Generate one string by autoregressive sampling from a bigram model.

    ``probs[i]`` is the distribution over the next character given that the
    previous character is ``i``; ``probs`` has shape (V, V) where V is the
    vocabulary size. Sampling starts from the boundary token (index 0) and
    stops as soon as the boundary token is drawn again.

    Agnostic to where ``probs`` came from: works the same on a counting
    model (normalized frequencies) and on a neural model (``softmax(W)``).
    """
    out: list[str] = []

    i: int = 0
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
        j = np.argmax( rng.multinomial(n=1, pvals=probs[i]) ).item() 
        
        # If the next character is the boundary token, then return (without add it)
        if j == 0:
            return ''.join(out)

        out.append( itos[j] )
        i = j