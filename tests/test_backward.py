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
def forward_setup() -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
    """Run one full forward pass on a tiny fixed dataset, shared by the gradient checks.

    Returns (probs, ys, alphabet_len, logits, xenc, W). W is cast to float64
    right after initialization, so every downstream quantity (logits, probs)
    inherits the precision: finite-difference perturbations would be swamped
    by float32 rounding noise. pytest re-runs this per test, so each check
    starts from a clean, unperturbed copy.
    """
    # --- Arrange: prepare the situation ---

    # 1. Prepare the ``training set``.
    dataset = ["ba"]
    stoi, itos = data.build_vocab(dataset)
    xs, ys = data.build_dataset_bigram(dataset, stoi)
    
    # 2. Initialize the ``network``.
    alphabet_len = len(stoi)
    xenc = data.one_hot(xs, num_classes=alphabet_len)
    rng = np.random.default_rng(SEED)
    W = neural.build_layer(alphabet_len, alphabet_len, rng).astype( np.float64 )

    # 3. Forward Pass.
    logits = neural.linear_forward(xenc, W).astype( np.float64 )
    probs = neural.softmax(logits).astype( np.float64 )
    loss = neural.mean_nll(probs, ys)

    return probs, ys, alphabet_len, logits, xenc, W


def test_dprobs_matches_numerical_gradient(forward_setup):
    """Check dL/dprobs against central finite differences.

    Perturbs each probs[i,j] by +-epsilon, recomputes the loss, 
    and compares the numerical slope with the analytic gradient from 
    d_loss_d_probs function.
    """
    probs, ys, _, _, _, _= forward_setup 

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
    probs, ys, alphabet_len, logits, _, _ = forward_setup 

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
    print("[Test 2] dlogits")
    print("- dlogits_analytic:")
    print(dlogits_analytic)
    print("- dlogits_numeric:")
    print(dlogits_numeric)
    print()

    assert np.allclose(dlogits_analytic, dlogits_numeric, atol=1e-6)


def test_dW_matches_numerical_gradient(forward_setup):
    """Check dL/dW against central finite differences.

    Perturbs each W[i,j] by +-epsilon and re-runs the whole forward pass
    (linear forward, softmax, nll), since a weight change propagates through
    every downstream quantity. Compares the numerical slope with the analytic
    xenc.T @ dlogits, from the d_loss_d_w function.
    """
    probs, ys, alphabet_len, logits, xenc, W = forward_setup 

    # --- Act: exectute what you wanna test ---
    dlogits = neural.d_loss_d_logits(probs, ys, alphabet_len).astype( np.float64 )
    dW_analytic = neural.d_loss_d_w(xenc, dlogits).astype( np.float64 )

    # --- Assert: verify if works ---
    dW_numeric = np.zeros( W.shape, dtype=np.float64 )
    for i in range( W.shape[0] ):
        for j in range( W.shape[1] ):
            
            original_weight = W[i, j]

            W[i, j] = original_weight + EPSILON
            logits_plus = neural.linear_forward(xenc, W)
            probs_plus = neural.softmax(logits_plus)
            loss_plus = neural.mean_nll(probs_plus, ys)

            W[i, j] = original_weight - EPSILON
            logits_minus = neural.linear_forward(xenc, W)
            probs_minus = neural.softmax(logits_minus)
            loss_minus = neural.mean_nll(probs_minus, ys)

    
            W[i, j] = original_weight
            
            dW_numeric[i, j] = ( loss_plus - loss_minus ) / (2 * EPSILON)
    
    # Use the "-s" option to print
    print("[Test 3] dW")
    print("- dW_analytic:")
    print(dW_analytic)
    print("- dW_numeric:")
    print(dW_numeric)
    print()

    assert np.allclose(dW_analytic, dW_numeric, atol=1e-6)


def test_full_backward(forward_setup):
    """Check the full backward() pipeline against central finite differences.

    The mathematics is already validated piece by piece by the previous
    tests; what is under test here is the *wiring*: that ``backward()``
    orchestrates the individual gradient functions correctly (right
    arguments, right order, right return value). Today the function is a
    thin wrapper, but as the network grows, this test guards the assembly,
    not the math.
    """
    probs, ys, alphabet_len, logits, xenc, W = forward_setup 

    # --- Act: exectute what you wanna test ---
    backward_analytic = neural.backward(probs, ys, alphabet_len, xenc)

    # --- Assert: verify if works ---
    backward_numeric = np.zeros( W.shape, dtype=np.float64 )
    for i in range( W.shape[0] ):
        for j in range( W.shape[1] ):
            
            original_weight = W[i, j]

            W[i, j] = original_weight + EPSILON
            logits_plus = neural.linear_forward(xenc, W)
            probs_plus = neural.softmax(logits_plus)
            loss_plus = neural.mean_nll(probs_plus, ys)

            W[i, j] = original_weight - EPSILON
            logits_minus = neural.linear_forward(xenc, W)
            probs_minus = neural.softmax(logits_minus)
            loss_minus = neural.mean_nll(probs_minus, ys)

    
            W[i, j] = original_weight
            
            backward_numeric[i, j] = ( loss_plus - loss_minus ) / (2 * EPSILON)
    
    # Use the "-s" option to print
    print("[Test 4] Backward")
    print("- backward_analytic:")
    print(backward_analytic)
    print("- backward_numeric:")
    print(backward_numeric)
    print()

    assert np.allclose(backward_analytic, backward_numeric, atol=1e-6)