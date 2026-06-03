# makemore-numpy

Character-level language models in **pure NumPy**.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[experiments]"
```

## Roadmap

1. **Step 1 — Bigram** (`src/rnnumpy/bigram.py`): predicts the next character from the previous one.
2. **Step 2 — MLP**: adds a hidden layer with `tanh`.
3. **Step 3 — RNN**: adds recurrence (BPTT).