"""Gradient checks for the hand-written backward pass.

Each test compares an analytic gradient (from neural.backward) against a
numerical one, obtained by perturbing each entry by ±epsilon and measuring
the change in loss via central differences. Agreement within tolerance is
evidence the analytic derivation is correct.

Uses a tiny hand-checkable setup (small vocab, few examples) and float64
throughout, since finite differences are too noisy in float32.
"""
import numpy as np

from makemore import data, neural


# Constants
SEED = 2147483647


def test_dprobs_matches_numerical_gradient():

    # --- Arrange: prepare the situation ---
    # 1. Preparint the ``Training Set``
    dataset = ["ba"]
    stoi, itos = data.build_vocab(dataset)
    xs, ys = data.build_training_set(dataset, stoi)

    # 2. Initialize the 'network'
    alphabet_len = len(stoi)
    xenc = data.one_hot(xs, num_classes=alphabet_len)
    rng = np.random.default_rng(SEED)       # np.random.Generator
    W = neural.build_layer(alphabet_len, alphabet_len, rng)
    
    # 3. Forward Pass
    logits = neural.linear_forward(xenc, W)
    probs = neural.softmax(logits).astype( np.float64 )
    loss = neural.mean_nll(probs, ys)

    # --- Act: exectute what you wanna test ---
    dprobs_analytic = neural.backward(probs, ys).astype( np.float64 )


    # --- Assert: verify if works ---
    dprobs_numeric = np.zeros( probs.shape, dtype=np.float64 )
    epsilon = 1e-5
    for i in range( probs.shape[0]):
        for j in range(probs.shape[1]):
            original_prob = probs[i,j]

            probs[i,j] = original_prob + epsilon
            loss_plus = neural.mean_nll(probs, ys)

            probs[i,j] = original_prob - epsilon
            loss_minus = neural.mean_nll(probs, ys)

            probs[i,j] = original_prob

            dprobs_numeric[i,j] = ( loss_plus - loss_minus ) / (2 * epsilon)

    print()
    print(dprobs_analytic)
    print(dprobs_numeric)
    assert np.allclose(dprobs_analytic, dprobs_numeric, atol=1e-6)