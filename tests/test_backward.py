"""Gradient checks for the hand-written backward pass.

Each test compares an analytic gradient (from neural.backward) against a
numerical one, obtained by perturbing each entry by ±epsilon and measuring
the change in loss via central differences. Agreement within tolerance is
evidence the analytic derivation is correct.

Uses a tiny hand-checkable setup (small vocab, few examples) and float64
throughout, since finite differences are too noisy in float32.
"""
import pytest
import numpy as np

from makemore import data, neural


# Constants
SEED = 2147483647
EPSILON = 1e-5


@pytest.fixture
def forward_setup() -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Run one full forward pass on a tiny fixed dataset, shared by the gradient checks.

    Returns (probs, ys, alphabet_len, logits), where ``probs`` and ``logits`` are in float64 
    so that finite-difference perturbations aren't swamped by float32 rounding noise. 
    pytest re-runs this per test, so each check starts from a clean, unperturbed copy.
    """
    # --- Arrange: prepare the situation ---

    # 1. Prepare the ``training set``.
    dataset = ["ba"]
    stoi, itos = data.build_vocab(dataset)
    xs, ys = data.build_training_set(dataset, stoi)
    
    # 2. Initialize the ``network``.
    alphabet_len = len(stoi)
    xenc = data.one_hot(xs, num_classes=alphabet_len)
    rng = np.random.default_rng(SEED)
    W = neural.build_layer(alphabet_len, alphabet_len, rng)

    # 3. Forward Pass.
    logits = neural.linear_forward(xenc, W).astype( np.float64 )
    probs = neural.softmax(logits).astype( np.float64 )
    loss = neural.mean_nll(probs, ys)

    return probs, ys, alphabet_len, logits


def test_dprobs_matches_numerical_gradient(forward_setup):
    """Check dL/dprobs against central finite differences.

    Perturbs each probs[i,j] by +-epsilon, recomputes the loss, 
    and compares the numerical slope with the analytic gradient from 
    d_loss_d_probs function.
    """
    probs, ys, _, _ = forward_setup 

    # --- Act: exectute what you wanna test ---
    dprobs_analytic = neural.d_loss_d_probs(probs, ys).astype( np.float64 )

    # --- Assert: verify if works ---
    dprobs_numeric = np.zeros( probs.shape, dtype=np.float64 )
    for i in range( probs.shape[0]):
        for j in range(probs.shape[1]):
            original_prob = probs[i,j]

            probs[i,j] = original_prob + EPSILON
            loss_plus = neural.mean_nll(probs, ys)

            probs[i,j] = original_prob - EPSILON
            loss_minus = neural.mean_nll(probs, ys)

            probs[i,j] = original_prob

            dprobs_numeric[i,j] = ( loss_plus - loss_minus ) / (2 * EPSILON)

    # Use the "-s" option to print
    print("[Test 1] dprobs")
    print("- dprobs_analytic:")
    print(dprobs_analytic)
    print("- dprobs_numeric:")
    print(dprobs_numeric)
    print()

    assert np.allclose(dprobs_analytic, dprobs_numeric, atol=1e-6)


def test_dlogits_matches_numerical_gradient(forward_setup):
    """Check dL/dlogits against central finite differences.

    Perturbs each logit by +-epsilon and re-runs softmax + nll 
    (a logit shift moves the whole row's probabilities), 
    comparing the numerical slope with the analytic (probs - onehot)/N,
    from d_loss_d_logits function.
    """
    probs, ys, alphabet_len, logits = forward_setup 

    # --- Act: exectute what you wanna test ---
    dlogits_analytic = neural.d_loss_d_logits(probs, ys, alphabet_len).astype( np.float64 )

    # --- Assert: verify if works ---
    dlogits_numeric = np.zeros( probs.shape, dtype=np.float64 )
    for i in range( logits.shape[0]):
        for j in range(logits.shape[1]):
            
            original_logit = logits[i, j]

            logits[i, j] = original_logit + EPSILON
            probs_plus = neural.softmax(logits)
            loss_plus = neural.mean_nll(probs_plus, ys)

            logits[i, j] = original_logit - EPSILON
            probs_minus = neural.softmax(logits)
            loss_minus = neural.mean_nll(probs_minus, ys)

    
            logits[i, j] = original_logit

            dlogits_numeric[i, j] = ( loss_plus - loss_minus ) / (2 * EPSILON)
    
    # Use the "-s" option to print
    print("[Test 1] dlogits")
    print("- dlogits_analytic:")
    print(dlogits_analytic)
    print("- dlogits_numeric:")
    print(dlogits_numeric)
    print()

    assert np.allclose(dlogits_analytic, dlogits_numeric, atol=1e-6)