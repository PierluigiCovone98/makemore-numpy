"""Neural bigram model for next-character prediction: a single linear layer
trained by gradient descent, with a hand-written backward pass (no autograd).

Same task as the counting model, learned instead of counted.
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


def linear_forward( xenc: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Compute the matrix multiplication between the ``inputs vector`` and a neural net layer ``W``.

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
    dlogits = d_loss_d_logits(probs, ys, alphabet_len)

    # --- d(loss) / d(W)
    dW = d_loss_d_w(xenc, dlogits)

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


def d_loss_d_logits( probs: np.ndarray, ys: np.ndarray, alphabet_len: int) -> np.ndarray:
    """Compute d(loss)/d(logits) as (probs - onehot(ys)) / N.

    The compact form left after composing d(loss)/d(probs) with the
    softmax Jacobian: "predicted - correct", scaled by the mean.
    """
    yenc = data.one_hot(ys, alphabet_len)
    N = len(ys)
    dlogits = (probs - yenc) / N

    return dlogits


def d_loss_d_w( xenc: np.ndarray, dlogits: np.ndarray, ) -> np.ndarray:
    """Compute d(loss)/d(W) as xenc.T @ dlogits."""
    return xenc.T @ dlogits


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
        logits = linear_forward(xenc, W)
        probs = softmax(logits)
        loss = mean_nll(probs, ys)

        # Logs
        if log_every > 0 and (
            (step % log_every == 0) or (step == epochs-1) ):
            print(f"step {step:>5d} / {epochs:<5d}: loss={loss:.8f}")

        # === Backward pass
        dW = backward(probs, ys, alphabet_len, xenc)
            
        # === Update
        W -= lr * dW


def sample( probs: np.ndarray, itos: data.IntToStr, rng: np.random.Generator ) -> str:
    """ [...] 
    
    ``probs`` is the result of applying the ``softmax`` operation to the ``logits`` matrix.
    """
    out: list[str] = []

    i: int = 0
    while True:
        j = np.argmax( rng.multinomial(n=1, pvals=probs[i]) ).item() 
        
        # If the next character is the boundary token, then return (without add it)
        if j == 0:
            return ''.join(out)

        out.append( itos[j] )
        i = j