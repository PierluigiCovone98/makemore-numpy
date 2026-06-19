""" [...] """
from pathlib import Path
from makemore import data, neural

import numpy as np

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
NAMES_PATH = PROJECT_ROOT / "data" / "names.txt"

SEED = 2147483647

# ``BLOCK_SIZE`` is an hyperparameter that tells  
# how many previous characters the model is allowed to look at. 
BLOCK_SIZE = 3

# ``N_EMB`` is an hyperparameter that tells how many dimension
# has each embedding.
N_EMB = 2

HIDDEN_LAYER_OUTPUTS = 100


def main():

    # --- Dataset ---
    # Load the raw words, build the vocabulary, then build the (X, Y) dataset
    # with a context window of BLOCK_SIZE characters:
    #   X: (N, BLOCK_SIZE) int  -- each row is one context (BLOCK_SIZE indices)
    #   Y: (N,)            int  -- each entry is the next-character index to predict
    raw_dataset = data.read_dataset(NAMES_PATH)
    stoi, itos = data.build_vocab(raw_dataset)
    X, Y = data.build_dataset(raw_dataset[:32], stoi, BLOCK_SIZE)

    # Vocabulary size V
    alphabet_len = len(stoi)

    # --- Optional Logs ---
    # Read the first examples back as (context -> target) to verify the window:
    # for i in range(5):
    #     print(f"{''.join(itos[x] for x in X[i])} -> {itos[Y[i]]}")
    #
    # print(X[4, 2])   # expected: 1

    
    # --- Embedding lookup ---
    # Embedding forward pass: create the look-up table C, 
    # the first (linear) layer of the MLP.
    # Then map every index in X to its embedding row.
    rng = np.random.default_rng(SEED)


    # --- Forward pass ---
    # --- First layer.
    C = neural.build_layer(alphabet_len, N_EMB, rng)
    embeddings = neural.embed(X, C)
    
    # Linear layers are defined on vectors, not on matrices. 
    # This is why the output of the "first linear layer" must be a vector
    # (and so: why we concatenate).
    concat_embeddings = neural.concatenate_embs(embeddings)
    
    # --- Hidden layer.
    W1 = neural.build_layer( concat_embeddings.shape[1], HIDDEN_LAYER_OUTPUTS, rng)
    # Notice that in a (linear) layer there's one bias-per-neuron.
    b1 = neural.create_biases( HIDDEN_LAYER_OUTPUTS, rng )
    activations = np.tanh( neural.linear_forward(concat_embeddings, W1) + b1 )

    # --- Last layer.
    W2 = neural.build_layer(HIDDEN_LAYER_OUTPUTS, alphabet_len, rng)
    b2 = neural.create_biases( alphabet_len, rng )
    logits = neural.linear_forward(activations, W2) + b2

    # --- Softmax.
    probs = neural.softmax(logits)

    # --- Loss.
    loss = neural.mean_nll(probs, Y)
    print(loss)

    # --- Define parameters
    parameters = [C, W1, b1, W2, b2]



    # --- Backward pass ---
    #
    # 1. We have already seen (with the neural bigram model)
    #    that d(loss)/d(probs) is not used by the subsequent
    #    step, that is d(loss)/d(logits).
    #    So: let's directly compute dlogits here.
    dlogits = neural.d_loss_d_logits(probs, Y, alphabet_len) 
    
    # 2. Let's now create dW2 and dactivations
    dW2 = neural.d_loss_d_w(activations, dlogits)
    dactivations = neural.d_loss_d_x(dlogits, W2)
    db2 = neural.d_loss_d_b(dlogits)









    
if __name__ == "__main__":
    main()