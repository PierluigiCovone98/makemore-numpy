"""Neural bigram model for next-character prediction: a single linear layer
trained by gradient descent, with a hand-written backward pass (no autograd).

Same task as the counting model, learned instead of counted.
"""
import numpy as np


def build_layer( n_inputs: int, n_outs: int, rng: np.random.Generator) -> np.ndarray:
    """Build a neural net linear layer with ``n_outs`` neurons where each neuron has ``n_inputs`` weights.
    Weights are initialized with single precision float values, from a "standard normal" distribution. 
    """
    # Check dimensions integrity 
    if n_inputs <= 0 or n_outs <= 0:
        raise ValueError("Dimensions must be positive numbers")

    return rng.standard_normal( (n_inputs, n_outs), dtype= np.float32) 


def linear_forward( xenc: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Compute the matrix multiplication between the ``inputs vector`` and a nerual net layer ``W``.

    Assumes that neurons in the layer ``W`` are composed only by weights.  
    """
    return xenc @ W


def softmax( logits: np.ndarray ) -> np.ndarray:
    """Turn each row of unconstrained logits into a probability distribution.

    Logits are real numbers: they can be negative and need not sum to 1, so they
    cannot be read as probabilities directly. Softmax fixes both:
        -   exp() makes every value positive, 
        -   and dividing by the row sum normalizes each row to sum to 1.
    Applied row-wise, so each row becomes the next-character distribution for one input.
    
    For tecnichal constraints we from each logit the max value of the row in which it 
    is located. This ensures that the exponentiation does not overflow.
    """
    logits = logits - logits.max(axis=1, keepdims=True)
    counts = np.exp(logits)
    return counts / np.sum(counts, axis=1, keepdims=True)


def mean_nll( probs: np.ndarray, ys: np.ndarray ) -> float:
    """Compute the mean negative log likelihood."""
    return - np.mean( np.log( probs[ np.arange(ys.size), ys ] ) ).item()


def backward( probs: np.ndarray, ys: np.ndarray ) -> np.ndarray:
    """ [...] """

    # --- d(loss) / d(probs) ---
    dprobs = np.zeros( shape=probs.shape, dtype=np.float32 )
    
    ys_len = len(ys)
    for k, yk in enumerate(ys):
        dprobs[k, yk] = - (1 / ys_len) * (1 / probs[k,yk])

    return dprobs