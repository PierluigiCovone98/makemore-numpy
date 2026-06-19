"""Backward pass for the MLP next-character model: the architecture-specific
assembly of the hand-written gradients.

The forward pass lives in the experiment (``experiments/train_mlp.py``); what
lives here is the *wiring* of its backward. ``backward`` walks the pipeline in
reverse -- output layer, tanh, hidden layer, concatenation, embedding lookup --
chaining the generic gradient primitives from ``neural`` and returning the five
parameter gradients.

Mirrors ``neural_bigram``: the per-link mathematics is generic and tested in
``neural``; what is specific to this model is the order of the links and which
forward quantity feeds which gradient.
"""
import numpy as np

from makemore import neural


def backward(alphabet_len: int, block_size: int, n_emb: int, X: np.ndarray, Y: np.ndarray,
            concat_embeddings: np.ndarray, W1: np.ndarray, W2: np.ndarray,  activations: np.ndarray,
            probs: np.ndarray,) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the full MLP backward pass; return the gradients ``(dC, dW1, db1, dW2, db2)``.

    Walks the forward pipeline in reverse, one link at a time, chaining the
    generic gradient primitives from ``neural``. Receives only what the chain
    consumes -- the biases and ``C`` are absent, since a gradient never needs
    the value of the parameter it targets.
    """
    # 1. We have already seen (with the neural bigram model)
    #    that d(loss)/d(probs) is not used by the subsequent
    #    step, that is d(loss)/d(logits).
    #    So: let's directly compute dlogits here.
    dlogits = neural.d_loss_d_logits(probs, Y, alphabet_len) 
    
    # 2. Let's now deal layer 2 of the MLP
    dW2 = neural.d_loss_d_w(activations, dlogits)
    dactivations = neural.d_loss_d_x(dlogits, W2)
    db2 = neural.d_loss_d_b(dlogits)

    # 3. Proceeding back to the hidden layer
    dpreactivations = neural.d_loss_d_preactivations(activations, dactivations)
    dW1 = neural.d_loss_d_w(concat_embeddings, dpreactivations)
    dconcat = neural.d_loss_d_x(dpreactivations, W1)
    db1 = neural.d_loss_d_b(dpreactivations)

    # 4. First layer
    dembeddings = neural.unconcatenate_embs(dconcat, block_size, n_emb)
    dC = neural.d_loss_d_C( dembeddings, X, alphabet_len, n_emb )

    return (dC, dW1, db1, dW2, db2)