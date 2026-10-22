# Lesson 6 — Layers, softmax, minibatches

**Read [THEORY.md](THEORY.md) first.**

## What you build

A small neural-network library: `Dense`, `ReLU` and `Sigmoid` layers that all
obey one interface, a `Sequential` container, softmax with cross-entropy, and a
minibatch training loop. Then you train it to read handwritten digits.

## Setup

```bash
pytest lesson06_layers -x
python lesson06_layers/layers.py       # after task 5: train on digits, watch accuracy climb
```

To look at the data: `python -c "from nnlib.data import load_digits, show_digit; d = load_digits(); print(d.y_train[0]); print(show_digit(d.X_train[0]))"`

## Tasks

| # | What | Theory |
|---|---|---|
| 1 | `Dense.forward`, `Dense.backward` | §2 |
| 2 | `ReLU`, `Sigmoid` | §3 |
| 3 | `softmax`, `cross_entropy`, `cross_entropy_backward` | §4 |
| 4 | `Sequential.forward`, `Sequential.backward`, `sgd_step` | §5 |
| 5 | `minibatches`, `train_epoch`, `accuracy` | §6 |

Look closely at `test_every_gradient_in_a_deep_stack_is_correct`: three `Dense`
layers and two kinds of activation, gradient-checked, and you wrote no new
calculus. That is the whole point of the brick interface.

## Experiments

1. **Depth by editing a list.** Add a second hidden layer:
   `[Dense(64, 32), ReLU(), Dense(32, 32), ReLU(), Dense(32, 10)]`.
2. **Activations.** Replace each `ReLU()` with `Sigmoid()`. Compare validation
   accuracy after 5 epochs. Then try a 5-hidden-layer network with each.
3. **Batch size.** Try `batch_size=1` and `batch_size=1437` (the whole training
   set). Compare time per epoch and accuracy after 5 epochs.
4. **Mistakes.** After training, find validation images the model gets wrong and
   print them with `show_digit`. Would you have got them right?

## The limit

Two limits, one for each of the next two lessons:

1. **You don't yet know whether the network is any good.** Training accuracy is
   98.7% and validation 97.2% after 30 epochs. Is that gap a problem? What if the
   learning rate were wrong, or the initialisation? Lesson 7 is about diagnosing
   and fixing training.
2. **Every new kind of brick needs a hand-written backward.** Real networks use
   convolutions, attention, normalisation, and dozens more. Notice that every
   backward you wrote was the chain rule applied to a few additions and
   multiplications. If the numbers themselves recorded the arithmetic, the
   computer could write every backward for you. Lesson 8.

<details>
<summary>Hints (open only after trying)</summary>

- **Dense.backward:** `self.dW[...] = self.X.T @ dY`, the same for `db`, then return `dY @ self.W.T`.
- **ReLU.forward:** `self.mask = X > 0`, then `return X * self.mask`.
- **softmax:** `logits.max(axis=1, keepdims=True)` keeps shape `(n, 1)`. Same for the sum.
- **cross_entropy:** `P[np.arange(n), labels]` picks one probability per row.
- **cross_entropy_backward:** `dZ = P.copy()`, subtract 1 at `[np.arange(n), labels]`, divide by `n`.
- **minibatches:** `order = rng.permutation(len(X))`, then slice `order[start:start + batch_size]`.
</details>
