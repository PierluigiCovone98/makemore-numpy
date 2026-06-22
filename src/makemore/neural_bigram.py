"""Neural bigram model for next-character prediction: a single linear layer
trained by gradient descent, with a hand-written backward pass (no autograd).

Same task as the counting model, learned instead of counted.
"""
import numpy as np

from makemore import neural, data

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


def sample( probs: np.ndarray, itos: data.IntToStr, rng: np.random.Generator ) -> str:
    """Generate one string by autoregressive sampling from the neural bigram model.

    ``probs[i]`` is the distribution over the next character given the previous
    character ``i``; ``probs`` has shape (V, V), here from ``softmax(W)``. Sampling
    starts from the boundary token (index 0) and stops as soon as it is drawn again.

    Intentionally duplicated in ``counting.sample``: same (V, V) shape, but the two
    models are independent and may diverge.
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