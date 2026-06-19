"""Shared neural-network primitives for next-character prediction.

Architecture-agnostic building blocks, used by more than one model: 
layer and bias initialization, the embedding lookup, the linear forward, 
softmax, mean NLL, and the generic gradient functions.
"""
import numpy as np

from makemore import data


def build_layer( n_inputs: int, n_outs: int, rng: np.random.Generator) -> np.ndarray:
    """Build a neural net linear layer with ``n_outs`` neurons where each neuron has ``n_inputs`` weights.
    Weights are initialized with single precision float values, from a "standard normal" distribution. 
    """
    # Check dimensions integrity 
    if n_inputs <= 0 or n_outs <= 0:
        raise ValueError("Dimensions must be positive numbers")

    return rng.standard_normal( (n_inputs, n_outs), dtype= np.float32) 


def create_biases( n_outs: int, rng: np.random.Generator) -> np.ndarray: 
    """Create a vector of ``n_outs`` random biases."""

    return rng.standard_normal( n_outs, dtype= np.float32) 


def embed( X: np.ndarray, C: np.ndarray ) -> np.ndarray:
    """Map each character index in ``X`` to its embedding vector via lookup in ``C``.

    ``C`` is the look-up table of shape ``(V, n_emb)``, the first (linear)
    layer of the network, learned like any other weight matrix, where ``n_emb``
    is an hyperparameter that tells how many dimension each embedding have.
    Looking up an index is exactly ``one_hot(idx) @ C``, just done by indexing 
    instead of a matrix multiplication.

    Indexing preserves the shape of ``X`` and appends the embedding axis:
    ``X`` of shape ``(N, block_size)`` yields ``(N, block_size, n_emb)``.
    """
    return C[X]


def concatenate_embs( embeddings: np.ndarray ) -> np.ndarray:
    """Performs the concatenation of embeddings.
    
    It assumes that the input matrix has a shape equal to (N, context_size, n_emb),
    where:
        -   ``N`` is the number of ``observed examples``;
        -   ``context_size`` is the number of spots in the context window;
        -   ``n_emb`` are dimensions of each array.
    
    Returns the concatenation of the input matrix, that is a matrix of shape
    equal to: 
        (N, context_size * n_emb).
    """
    return embeddings.reshape( embeddings.shape[0], embeddings.shape[1] * embeddings.shape[2] )


def linear_forward( x: np.ndarray, W: np.ndarray ) -> np.ndarray:
    """Linear layer forward pass: matrix-multiply the input ``x`` by the weights ``W``.

    Generic over layers. ``x`` is whatever enters the layer in the forward pass; 
    each column of ``W`` is one neuron, so ``x @ W`` gives one output per neuron. 
    
    Bias, if any, is added by the caller.
    """
    return x @ W


def softmax( logits: np.ndarray ) -> np.ndarray:
    """Turn each row of unconstrained logits into a probability distribution.

    Logits are real numbers: they can be negative and need not sum to 1, so they
    cannot be read as probabilities directly. Softmax fixes both:
        -   exp() makes every value positive, 
        -   and dividing by the row sum normalizes each row to sum to 1.
    Applied row-wise, so each row becomes the next-character distribution for one input.
    
    For numerical stability we subtract from each logit the max value of its row: 
    this keeps the exponentiation from overflowing, and leaves the result unchanged 
    (softmax is shift-invariant).
    """
    logits = logits - logits.max(axis=1, keepdims=True)
    counts = np.exp(logits)
    return counts / np.sum(counts, axis=1, keepdims=True)


def mean_nll( probs: np.ndarray, ys: np.ndarray ) -> float:
    """Compute the mean negative log likelihood."""
    return - np.mean( np.log( probs[ np.arange(ys.size), ys ] ) ).item()


# === Derivatives methods ===

def d_loss_d_logits( probs: np.ndarray, ys: np.ndarray, alphabet_len: int) -> np.ndarray:
    """Compute d(loss)/d(logits) as (probs - onehot(ys)) / N.

    The compact form left after composing d(loss)/d(probs) with the
    softmax Jacobian: "predicted - correct", scaled by the mean.
    """
    yenc = data.one_hot(ys, alphabet_len)
    N = len(ys)
    dlogits = (probs - yenc) / N

    return dlogits


def d_loss_d_w( layer_input: np.ndarray, d_out: np.ndarray ) -> np.ndarray:
    """Gradient of the loss, generic over layers, w.r.t. a linear layer's weights.

    ``layer_input`` is what entered the layer in the forward pass 
    (e.g. the activations feeding the output layer, or the
    concatenated embeddings feeding the hidden layer); 
    
    ``d_out`` is the gradient already backpropagated to 
    that layer's output .
    """
    return layer_input.T @ d_out


def d_loss_d_x( d_out: np.ndarray, W: np.ndarray) -> np.ndarray :
    """Gradient of the loss w.r.t. a linear layer's input ``x``.
     
    ``d_out`` is the gradient already backpropagated to 
    that layer's pre-activation output.

    ``W`` are the linear layer's weights. 
    """
    return d_out @ W.T


def d_loss_d_b( d_out: np.ndarray ) -> np.ndarray:
    """[...]"""
    return d_out.sum(axis=0)  