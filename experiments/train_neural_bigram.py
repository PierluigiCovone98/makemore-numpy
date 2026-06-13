"""Solve the "next character prediction problem" with a "neural" approach."""
from pathlib import Path
from makemore import data, neural

import numpy as np

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
NAMES_PATH = PROJECT_ROOT / "data" / "names.txt"

SEED = 2147483647

EPOCHS = 1000
LEARNING_RATE = 50

def main():
    
    # The first thing we are going to do is: create the 
    # ``training set`` (inputs, xs, and targets, ys).
    dataset = data.read_dataset(NAMES_PATH)
    stoi, itos = data.build_vocab(dataset)
    xs, ys = data.build_training_set(dataset, stoi)
    # print( [(i.item(),t.item()) for i,t in zip(xs, ys)]  )

    alphabet_len = len(stoi)


    # === Initialize the 'network' ===
    # We want to encode our "inputs vector" using the "one-hot enconding"
    # formula (indices are not good inputs because the net would treat 
    # them as quantitative, but the information is qualitative).
    xenc = data.one_hot(xs, num_classes=alphabet_len)
    # print(xenc[:10])

    # Then we want to create our neural net (one single layer)
    #
    # Notice that in our example the neural net is composed by 
    # a single linear layer where the number of weights correspond
    # to the number of the outputs of the layer;
    # there's a specific reason for that:
    #   In general we know that each neuron in a layer 
    #   takes a number of inputs that depends from 
    #   the number of outputs of the previous layer 
    #   in the neural net, or from the input format 
    #   if the layer is the first layer in the neural net
    #   (this is the case).
    #   And because inputs in the "inputs vector" are 
    #   "one-hot encoding vectors", with 27 columns,
    #   each neuron in the first layer (our "W") takes 
    #   27 inputs.
    #
    #   The number of neurons in a layer (so, the number of)
    #   outputs of the layer) depends on the architecture with
    #   which the neural net is designed; is arbitrary.
    #   But the last layer of the nerual net is a special case
    #   (and our W is both the first and the last layer of the net):
    #   the number of outputs of the last layer of the neural net
    #   depends from the task that the neural net is achieving. 
    #   In our case, the task is "having a probability distribution"
    #   for all the characters in the alphabet (that is 27 elements
    #   long). Here we are.
    rng = np.random.default_rng(SEED)       # np.random.Generator
    W = neural.build_layer(alphabet_len, alphabet_len, rng)
    # print(W)


    # === Training Loop: Gradient Descent ===
    # By default we use "full batch" such that the number of
    # epochs corresponds to the number of iterations (or steps);
    # this means that we do not introduce other hyperparameters 
    # (the number of steps itself) and we do not have to manage
    # stuffs like "shuffle batches" at each epoch.
    for epoch in range(EPOCHS):
        
        # === No "zerograd()" required

        # === Forward pass 
        logits = neural.linear_forward(xenc, W)
        probs = neural.softmax(logits)
        loss = neural.mean_nll(probs, ys)
        
        print(loss)

        # === Backward pass
        dW = neural.backward(probs, ys, alphabet_len, xenc)

        # === Update
        W -= LEARNING_RATE * dW 



if __name__ == "__main__":
    main()