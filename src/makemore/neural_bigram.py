"""Neural bigram model for next-character prediction: a single linear layer
trained by gradient descent, with a hand-written backward pass (no autograd).

Same task as the counting model, learned instead of counted.
"""
import numpy as np

from makemore import neural

def backward( probs: np.ndarray, ys: np.ndarray, alphabet_len: int, xenc: np.ndarray) -> np.ndarray:
    """Run the full backward pass and return dW, the gradient of the loss
    with respect to the weights.

    Walks the forward pipeline in reverse, one link at a time; each step
    consumes the previous one. See ``docs/backpropagation_by_hand.pdf`` for
    the full derivation.
    """

    # --- d(loss) / d(probs) ---
    # Not computed: it cancels into ``dlogits`` during the derivation
    # (see docs/backpropagation_by_hand.pdf). d_loss_d_probs is kept
    # below only as a documented step, exercised by its own test.
    # 
    # dprobs = d_loss_d_probs(probs, ys)

    # --- d(loss) / d(logits) ---
    dlogits = neural.d_loss_d_logits(probs, ys, alphabet_len)

    # --- d(loss) / d(W)
    dW = neural.d_loss_d_w(xenc, dlogits)

    return dW


# === Track the process ===

def d_loss_d_probs( probs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Compute d(loss)/d(probs).

    The loss reads a single entry per row of ``probs`` (the target's one),
    so the gradient is zero everywhere except at ``[k, ys[k]]``, which
    holds -1/(N * probs[k, ys[k]]).
    """
    dprobs = np.zeros( shape=probs.shape )
    
    # k     :=  refers to the k-th experiment (observed bigram) 
    # yk    :=  refers to the target of the k-th experiment
    for k, yk in enumerate(ys):
        # N :=  len(ys)     (Number of experiments)
        dprobs[k, yk] = - (1 / len(ys)) * (1 / probs[k,yk])

    return dprobs


def train( W: np.ndarray, epochs: int, lr: float, xenc: np.ndarray, ys: np.ndarray, alphabet_len: int, log_every: int = 0) -> None:
    """Train the network in place by full-batch gradient descent.

    At each step: forward pass, backward pass (returns dW), parameter
    update ``W -= lr * dW``. No gradient state to reset between steps —
    ``backward`` is a pure function that recomputes dW from scratch each
    time (unlike autograd engines, which accumulate gradients on
    parameters and require an explicit zero-grad before each step).

    Full batch: every step uses the whole dataset, so one step is one
    epoch. Mini-batching can be added later as an opt-in argument.

    If ``log_every > 0``, prints the loss every ``log_every`` steps;
    if ``0`` (default), trains silently.
    """
    if epochs <= 0:
        raise ValueError("Epochs must be at least 1.")
    
    for step in range(epochs):
        # === No "zerograd()" required

        # === Forward pass 
        logits = neural.linear_forward(xenc, W)
        probs = neural.softmax(logits)
        loss = neural.mean_nll(probs, ys)

        # Logs
        if log_every > 0 and (
            (step % log_every == 0) or (step == epochs-1) ):
            print(f"step {step:>5d} / {epochs:<5d}: loss={loss:.8f}")

        # === Backward pass
        dW = backward(probs, ys, alphabet_len, xenc)
            
        # === Update
        W -= lr * dW