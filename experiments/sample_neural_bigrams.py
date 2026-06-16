"""Generate names by sampling from the trained neural bigram model."""
from pathlib import Path
from makemore import data, neural

import numpy as np

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
NAMES_PATH = PROJECT_ROOT / "data" / "names.txt"

SEED = 2147483647
EPOCHS = 1200
LEARNING_RATE = 50.0
N_SAMPLES = 10

def main():
    # === Train ===
    # === Build the training set
    dataset = data.read_dataset(NAMES_PATH)
    stoi, itos = data.build_vocab(dataset)
    xs, ys = data.build_dataset_bigram(dataset, stoi)
    alphabet_len = len(stoi)
    xenc = data.one_hot(xs, num_classes=alphabet_len)

    # === Actual training
    rng = np.random.default_rng(SEED)
    W = neural.build_layer(alphabet_len, alphabet_len, rng)
    neural.train(W, EPOCHS, LEARNING_RATE, xenc, ys, alphabet_len)

    # === Sample ===
    # === Build the (V, V) transition matrix
    #
    # Conceptually, we need one next-character distribution for each of the
    # V possible previous characters. The clean way to write it would be:
    #
    #     xenc_alphabet = data.one_hot(np.arange(alphabet_len), alphabet_len)
    #     P = neural.softmax(neural.linear_forward(xenc_alphabet, W))
    #
    # Here ``xenc_alphabet`` is the (V, V) identity matrix built by
    # one-hot-encoding every character of the alphabet (row i = one-hot of
    # i): one row per possible input, no example seen twice.
    #
    # But ``xenc_alphabet`` is precisely the identity, and I @ W = W, so the
    # linear forward is the identity operation and the whole thing collapses
    # to a single softmax on W. The linear forward "happens" only in our
    # ``mental model``.
    P = neural.softmax(W)

    for _ in range(N_SAMPLES):
        print(data.sample(P, itos, rng))

if __name__ == "__main__":
    main()