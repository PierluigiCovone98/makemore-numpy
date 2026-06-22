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

from dataclasses import dataclass
from makemore import neural


@dataclass
class MLPParams:
    C: np.ndarray; W1: np.ndarray; b1: np.ndarray; W2: np.ndarray; b2: np.ndarray
    

def init_params(alphabet_len: int,  n_emb: int, context_size: int, hidden_layer_outputs: int, rng: np.random.Generator) -> MLPParams:
    """Build a fresh ``MLPParams`` with random weights.

    The factory for the dataclass: maps the architecture's dimensions to the
    five parameter arrays with the shapes each layer requires, keeping
    ``MLPParams`` itself pure data.
    """
    C = neural.build_layer(alphabet_len, n_emb, rng)
    W1 = neural.build_layer( context_size * n_emb, hidden_layer_outputs, rng)
    b1 = neural.create_biases( hidden_layer_outputs, rng )
    W2 = neural.build_layer(hidden_layer_outputs, alphabet_len, rng)
    b2 = neural.create_biases( alphabet_len, rng )
    
    return MLPParams(C=C, W1=W1, b1=b1, W2=W2, b2=b2)

    
def forward(X: np.ndarray, params: MLPParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the MLP forward pass.

    Returns ``probs`` plus the two intermediates the backward pass consumes. 
    The loss is left to the caller, so this function can also be used by
    evaluation and sampling, which have no targets ``Y``.
    """
    embeddings = neural.embed(X, params.C)
    # Embeddings are concatenated because linear layers are defined on vectors (not matrices)
    concat_embeddings = neural.concatenate_embs(embeddings)
    preactivations = neural.linear_forward(concat_embeddings, params.W1) + params.b1
    activations = np.tanh( preactivations )
    logits = neural.linear_forward(activations, params.W2) + params.b2
    probs = neural.softmax(logits)

    return (probs, activations, concat_embeddings)


def backward(alphabet_len: int, context_size: int, n_emb: int, X: np.ndarray, Y: np.ndarray,
            concat_embeddings: np.ndarray, W1: np.ndarray, W2: np.ndarray,  activations: np.ndarray,
            probs: np.ndarray,) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the full MLP backward pass; return the gradients ``(dC, dW1, db1, dW2, db2)``.

    Walks the forward pipeline in reverse, one link at a time, chaining the
    generic gradient primitives from ``neural``. Receives only what the chain
    consumes -- the biases and ``C`` are absent, since a gradient never needs
    the value of the parameter it targets.
    """
    # Compute ``dlogits`` without compute ``dprobs`` first, 
    # as we've alredy seen for the nerual bigram model. 
    dlogits = neural.d_loss_d_logits(probs, Y, alphabet_len) 
    
    # Backward through the MLP layer2. 
    dW2 = neural.d_loss_d_w(activations, dlogits)
    dactivations = neural.d_loss_d_x(dlogits, W2)
    db2 = neural.d_loss_d_b(dlogits)

    # Backward through the activation layer.
    dpreactivations = neural.d_loss_d_preactivations(activations, dactivations)
    dW1 = neural.d_loss_d_w(concat_embeddings, dpreactivations)
    dconcat = neural.d_loss_d_x(dpreactivations, W1)
    db1 = neural.d_loss_d_b(dpreactivations)

    # Backward through the MLP first layer.
    dembeddings = neural.unconcatenate_embs(dconcat, context_size, n_emb)
    dC = neural.d_loss_d_C( dembeddings, X, alphabet_len, n_emb )

    return (dC, dW1, db1, dW2, db2)


def train( steps: int, lr: float, alphabet_len: int, context_size: int, n_emb: int, batch_size: int,
          Xtr: np.ndarray, Ytr: np.ndarray, 
          params: MLPParams,rng: np.random.Generator,
          log_every: int = 0) -> None:
    """Train ``params`` in place by mini-batch gradient descent: each step takes 
    ``batch size`` random examples from the training inputs vector.

    Each step runs forward, backward, and an in-place parameter update
    No gradient reset between steps: ``backward`` recomputes them from scratch,
    with no graph to clear. 
    
    If ``log_every > 0``, prints the loss every ``log_every`` steps.
    Notice that the printed loss is the one of the batch:
    this means that lossess between one step and another could have noise.
    """
    if steps <= 0:
        raise ValueError("Steps must be at least 1.")

    len_Xtr = len(Xtr)

    for step in range(steps):
        # No "zerograd()" required because there's no computational graph under the hood.

        # Choose ``batch_size`` number of indices between ``0`` and 
        # the ``len of the training inputs vector``.
        idx = rng.integers(0, len_Xtr, size=batch_size)
        
        # X batch , Y batch       
        Xb, Yb = Xtr[idx], Ytr[idx]

        # === FORWARD ===
        probs, activations, concat_embeddings = forward(Xb, params)
        
        loss = neural.mean_nll(probs, Yb)

        # Logs
        if log_every > 0 and (
            (step % log_every == 0) or (step == steps-1) ):
                print(f"step {step:>5d} / {steps:<5d}: loss={loss:.8f}")

        # === Backward ===
        dC, dW1, db1, dW2, db2 = backward(alphabet_len, context_size, n_emb, Xb, Yb,
                            concat_embeddings, params.W1, params.W2, activations, probs)

    
        # === Update ===
        params.C -= lr * dC
        params.W1 -= lr * dW1
        params.b1 -= lr * db1
        params.W2 -= lr * dW2
        params.b2 -= lr * db2


def evaluate(X: np.ndarray, Y: np.ndarray, params: MLPParams) -> float:
    """Mean NLL loss of ``params`` on a dataset, without training.

    One forward pass plus the loss, with no backward and no update,
    so the parameters are left untouched.
    """
    probs, _, _ = forward(X, params)
    return neural.mean_nll(probs, Y)
