# Lesson 7 — Generalisation, initialisation, optimisers

**Read [THEORY.md](THEORY.md) first.**

## What you build

The tools for training well rather than just training: a train/validation split,
He initialisation, three optimisers (SGD, Momentum, Adam), early stopping, a
training loop that records everything, and `diagnose`, which looks at a training
run and says whether it diverged, underfit, overfit, or was healthy.

The payoff test runs four real training runs with deliberately different
problems and checks your `diagnose` names each one correctly.

## Setup

```bash
pytest lesson07_generalisation -x
python lesson07_generalisation/generalisation.py   # initialisation scales and an overfitting run
```

Lesson 6's layers (`Dense`, `ReLU`, `Sequential`, `softmax`, …) are imported,
finished, from `nnlib.layers`.

## Tasks

| # | What | Theory |
|---|---|---|
| 1 | `train_val_split` | §1 |
| 2 | `he_normal` | §3 |
| 3 | `SGD.step`, `Momentum.step`, `Adam.step` | §5 |
| 4 | `EarlyStopping.update`, `EarlyStopping.restore` | §6 |
| 5 | `fit`: minibatch training, metrics, early stopping | §1, §6 |
| 6 | `diagnose` | §7 |

## Experiments

1. **Initialisation.** Run the demo. Then build `Sequential` with 10 hidden layers
   of 100, replace each `Dense`'s `W` with `naive_uniform`, and try to train it on
   digits. What does the loss do?
2. **Learning-rate sweep.** For SGD, try `lr` in (0.001, 0.01, 0.1, 1, 5) for 10
   epochs each. Plot or print final validation accuracy. Repeat for Adam with
   (0.0001, 0.001, 0.01, 0.1). Which optimiser is less sensitive to the choice?
3. **More data versus a smaller model.** In the overfitting demo, change 60 to 600
   training images. Then, separately, keep 60 and change `hidden=256` to `hidden=16`.
   Which closes the gap more?
4. **Weight decay.** Add `g += 2 * 1e-3 * p` for each weight matrix before the
   optimiser step (THEORY §8) and rerun the 60-image demo.

## The limit

You can now train and diagnose. What you cannot easily do is invent: every brick
needs a hand-derived `backward`, and every optimiser needs the gradients those
produce. Lesson 8 builds **automatic differentiation**, where the numbers record
their own arithmetic and every backward pass is written for you.

<details>
<summary>Hints (open only after trying)</summary>

- **train_val_split:** `order = rng.permutation(len(X))`; the first `n_val` indices go to validation.
- **Momentum / Adam:** update the state arrays in place (`v *= beta; v += g`), so
  the lists `self.velocities`, `self.m` and `self.v` keep pointing at them.
- **Adam:** increment `self.t` once per `step`, before the loop over parameters.
- **EarlyStopping.update:** compare, save copies, then return `epoch - self.best_epoch >= self.patience`.
- **fit:** evaluate on the whole training and validation sets after the batches, not per batch.
</details>
