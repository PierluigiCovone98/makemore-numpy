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

## The bigram, two ways

The simplest version of the task — predict the next character from the
single previous one — is solved twice, with two approaches:

- **Counting model** (`src/makemore/counting.py`): builds the bigram
  transition matrix directly from frequencies, then samples from it. No
  learning involved.
- **Neural bigram** (`src/makemore/neural_bigram.py`): a single linear
  layer followed by softmax and mean negative log-likelihood. The weights
  are initialized randomly and trained by full-batch gradient descent. The
  dataset is small enough that the network is re-trained from scratch every
  time the experiment is run (no checkpointing).

Both converge to a loss of about **2.45** on the same dataset, and produce the
same kind of nonsense output. They are the same model reached two different
ways: one written down by hand, one learned. The neural version adds no
power on this problem — but it opens a door the counting model can't walk
through, because learning is what scales to deeper networks.

## The MLP

Instead of looking at a single previous
character, it conditions on a window of `context_size` of them; instead of
a one-hot encoding, it learns a low-dimensional **embedding** for each
character; and between input and output it adds a hidden layer with a
`tanh` non-linearity.

Its forward and backward passes are written by hand.
It is trained by mini-batch gradient descent over a proper train (80%) / dev (10%) / test (10%) split, and then it is measured with a separate evaluation pass.

On the names dataset it reaches about **2.36** train / **2.59** dev loss,
and its samples (`ken`, `man`, `dari`, `myn`, ...) are visibly more
name-like than the bigram's.

## Code organization

The library separates **shared primitives** from **model-specific wiring**,
so each model reuses the same building blocks without inheriting another
model's assumptions. The general-purpose modules are:

- `src/makemore/data.py` — dataset loading, vocabulary, dataset
  construction, the train / dev / test split, and one-hot encoding. Shared
  by every model.
- `src/makemore/neural.py` — architecture-agnostic neural primitives: layer
  and bias initialization, the embedding lookup, embedding concatenation,
  the linear forward, softmax, mean NLL, and the generic gradient pieces
  (`d_loss_d_logits`, `d_loss_d_w`, `d_loss_d_x`, `d_loss_d_b`, the `tanh`
  backward, the embedding-table backward). Used by more than one model.

Each model then lives in its own module and supplies only what is specific
to it.

## Documentation

The full derivation of the backward pass for the **neural bigram** model —
from the loss down to the weights — is in
[`docs/backpropagation_by_hand.pdf`](docs/backpropagation_by_hand.pdf).

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
python experiments/train_mlp.py                 # train, evaluate, and sample the MLP
```

The full test suite (gradient checks for the hand-written backward) runs
with:

```bash
pytest
```

## Roadmap

1. **Bigram** ✓ — counting and neural approaches both implemented.
2. **MLP** ✓ — embeddings, one hidden layer with `tanh`, multi-character context.