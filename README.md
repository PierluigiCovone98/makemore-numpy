# makemore-numpy

Character-level language models built from scratch in **pure NumPy**, with
a hand-written backward pass — no autograd engine underneath, just the
chain rule applied step by step. Loosely inspired by the first part of
Karpathy's *Neural Networks: Zero to Hero* series, with a different
target and a different setup.

## Why this project

This is a learning project. The aim is to understand how a neural
language model works — not by reproducing it through a framework, but
by deriving every gradient on paper, translating it into code, and
verifying it against numerical differentiation. The point is to
understand what's under the hood in modern frameworks, using toy
examples.

## Two approaches, same task

The same "next-character prediction problem" is solved twice, with two approaches:

- **Counting model** (`src/makemore/counting.py`): builds the bigram
  transition matrix directly from frequencies,
  then samples from it. No learning involved.
- **Neural model** (`src/makemore/neural_bigram.py`): a single linear layer
  followed by softmax and mean negative log-likelihood. The weights are
  initialized randomly and trained by full-batch gradient descent. The
  dataset is small enough that the network is re-trained from scratch
  every time the experiment is run (no checkpointing).

Both converge to a loss of about 2.45 on the same dataset, and produce
the same kind of nonsense outputs. They are the same model, reached two
different ways: one written down by hand, one learned. The neural
approach adds no power on this problem — but it opens a door the
counting model can't walk through, because learning scales to deeper networks. 
Those are the ones explored in the next steps of the roadmap

## Code organization
 
The library separates **shared primitives** from **model-specific
wiring**, so each model reuses the same building blocks without
inheriting another model's assumptions:
 
- `src/makemore/data.py` — dataset loading, vocabulary, dataset
  construction, and one-hot encoding. Shared by every approach.
- `src/makemore/neural.py` — architecture-agnostic neural primitives:
  layer and bias initialization, the embedding lookup, the linear
  forward, softmax, mean NLL, and the generic gradient pieces
  (`d_loss_d_logits`, `d_loss_d_w`). Used by more than one model.
- `src/makemore/neural_bigram.py` — the neural bigram model: its
  backward pass and training loop, built on the primitives above.
- `src/makemore/counting.py` — the counting model.
Keeping the primitives separate is what lets the MLP (next on the
roadmap) reuse the forward and gradient building blocks while writing
its own, separate backward assembly.

## Documentation

The full derivation of the backward pass for the **neural bigram** model —
from the loss down to the weights — is in
[`docs/backpropagation_by_hand.pdf`](docs/backpropagation_by_hand.pdf)

## Setup

The project is installed locally, together with the dependencies needed
to run the experiments and the tests. Open a terminal at the project
root and run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[experiments,dev]"
```

Requires Python 3.12+.

## Running things

Each experiment is a standalone script under `experiments/`. From the
project root:

```bash
python experiments/count_bigrams.py             # build the counting model
python experiments/sample_bigrams.py            # sample from counting
python experiments/compute_loss_bigrams.py      # reference loss (~2.4546)
python experiments/train_neural_bigram.py       # train the neural model
python experiments/sample_neural_bigrams.py     # train then sample from neural
```

The full test suite (gradient checks for the hand-written backward) runs
with:

```bash
pytest
```

## Roadmap

1. **Bigram** ✓ — counting and neural approaches both implemented.
2. **MLP** — one hidden layer with `tanh`, embeddings.
3. **RNN** — recurrence and backpropagation through time.