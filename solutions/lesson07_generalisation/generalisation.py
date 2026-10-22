"""Lesson 7 - Generalisation: validation, overfitting, initialisation, optimisers.

Read THEORY.md first. Then fill in each TODO, in order, and run:

    pytest lesson07_generalisation -x

Lesson 6's finished layers are imported from nnlib.layers.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from nnlib.data import load_digits
from nnlib.layers import (
    Dense,
    Layer,
    ParamPair,
    ReLU,
    Sequential,
    accuracy,
    cross_entropy,
    cross_entropy_backward,
    minibatches,
    softmax,
)
from nnlib.types import FloatArray, IntArray


# --- Task 1: hold data back (THEORY §1) --------------------------------------
@dataclass(frozen=True)
class Split:
    """GIVEN."""

    X_train: FloatArray  # (n_train, d)
    y_train: IntArray  # (n_train,)
    X_val: FloatArray  # (n_val, d)
    y_val: IntArray  # (n_val,)


def train_val_split(X: FloatArray, y: IntArray, val_fraction: float, rng: np.random.Generator) -> Split:
    """Shuffle the rows, then put the first round(n · val_fraction) into validation.

    Rows and labels must stay together, and no row may appear in both sets.
    """
    # TODO
    # BEGIN SOLUTION
    order = rng.permutation(len(X))
    n_val = round(len(X) * val_fraction)
    val, train = order[:n_val], order[n_val:]
    return Split(X_train=X[train], y_train=y[train], X_val=X[val], y_val=y[val])
    # END SOLUTION


# --- Task 2: initialisation (THEORY §3) --------------------------------------
type InitFn = Callable[[int, int, np.random.Generator], FloatArray]


def naive_uniform(d_in: int, d_out: int, rng: np.random.Generator) -> FloatArray:
    """GIVEN. What lessons 1-5 did: uniform in (-1, 1), whatever the layer's size."""
    return rng.uniform(-1, 1, (d_in, d_out))


def tiny_normal(d_in: int, d_out: int, rng: np.random.Generator) -> FloatArray:
    """GIVEN. Very small random weights."""
    return rng.normal(0.0, 0.01, (d_in, d_out))


def he_normal(d_in: int, d_out: int, rng: np.random.Generator) -> FloatArray:
    """Normal with mean 0 and standard deviation √(2 / d_in). Shape (d_in, d_out)."""
    # TODO
    # BEGIN SOLUTION
    return rng.normal(0.0, math.sqrt(2.0 / d_in), (d_in, d_out))
    # END SOLUTION


def signal_scales(init: InitFn, depth: int, width: int, rng: np.random.Generator) -> list[float]:
    """GIVEN. Push random inputs through `depth` Dense+ReLU layers initialised with `init`.

    Returns the standard deviation of the activations after each layer.
    A healthy network keeps these roughly constant.
    """
    A: FloatArray = rng.normal(size=(500, width))  # (500, width)
    scales: list[float] = []
    for _ in range(depth):
        A = np.maximum(0.0, A @ init(width, width, rng))
        scales.append(float(A.std()))
    return scales


# --- Task 3: optimisers (THEORY §5) ------------------------------------------
class Optimiser:
    """GIVEN. Holds the model's (parameter, gradient) pairs; step() updates parameters in place."""

    def __init__(self, params: list[ParamPair], lr: float) -> None:
        self.params = params
        self.lr = lr

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimiser):
    def step(self) -> None:
        """p ← p − lr · g, for every pair. (Lesson 6's sgd_step.)"""
        # TODO
        # BEGIN SOLUTION
        for p, g in self.params:
            p -= self.lr * g
        # END SOLUTION


class Momentum(Optimiser):
    def __init__(self, params: list[ParamPair], lr: float, beta: float = 0.9) -> None:
        """GIVEN. One velocity array per parameter, starting at zero."""
        super().__init__(params, lr)
        self.beta = beta
        self.velocities: list[FloatArray] = [np.zeros_like(p) for p, _ in params]

    def step(self) -> None:
        """For each parameter, in place:
            v ← β · v + g
            p ← p − lr · v
        """
        # TODO
        # BEGIN SOLUTION
        for (p, g), v in zip(self.params, self.velocities):
            v *= self.beta
            v += g
            p -= self.lr * v
        # END SOLUTION


class Adam(Optimiser):
    def __init__(
        self,
        params: list[ParamPair],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """GIVEN."""
        super().__init__(params, lr)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m: list[FloatArray] = [np.zeros_like(p) for p, _ in params]  # running mean of g
        self.v: list[FloatArray] = [np.zeros_like(p) for p, _ in params]  # running mean of g²
        self.t = 0  # number of steps taken

    def step(self) -> None:
        """Increase t by 1, then for each parameter, in place:
            m ← β₁ · m + (1 − β₁) · g
            v ← β₂ · v + (1 − β₂) · g²
            m̂ = m / (1 − β₁ᵗ)           v̂ = v / (1 − β₂ᵗ)       (bias correction)
            p ← p − lr · m̂ / (√v̂ + ε)
        """
        # TODO
        # BEGIN SOLUTION
        self.t += 1
        for (p, g), m, v in zip(self.params, self.m, self.v):
            m *= self.beta1
            m += (1 - self.beta1) * g
            v *= self.beta2
            v += (1 - self.beta2) * g**2
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        # END SOLUTION


# --- Task 4: early stopping (THEORY §6) --------------------------------------
class EarlyStopping:
    """Remember the best weights seen so far; say when to give up."""

    def __init__(self, patience: int) -> None:
        """GIVEN."""
        self.patience = patience  # epochs without improvement before stopping
        self.best_loss = math.inf
        self.best_epoch = -1
        self.best_params: list[FloatArray] = []  # copies of the parameters at best_epoch

    def update(self, epoch: int, val_loss: float, model: Layer) -> bool:
        """Record this epoch's validation loss. Return True if training should stop.

        If val_loss < best_loss: update best_loss and best_epoch, and save COPIES of
        every parameter array (p.copy() for p, _ in model.params()).
        Stop when epoch − best_epoch >= patience.
        """
        # TODO
        # BEGIN SOLUTION
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_params = [p.copy() for p, _ in model.params()]
        return epoch - self.best_epoch >= self.patience
        # END SOLUTION

    def restore(self, model: Layer) -> None:
        """Write the saved copies back into the model's parameters, in place."""
        # TODO
        # BEGIN SOLUTION
        for (p, _), saved in zip(model.params(), self.best_params):
            p[...] = saved
        # END SOLUTION


# --- Task 5: the training loop, with monitoring (THEORY §1, §6) --------------
@dataclass
class History:
    """GIVEN. One entry per completed epoch."""

    train_loss: list[float] = field(default_factory=lambda: [])
    val_loss: list[float] = field(default_factory=lambda: [])
    train_acc: list[float] = field(default_factory=lambda: [])
    val_acc: list[float] = field(default_factory=lambda: [])


def evaluate_loss(model: Layer, X: FloatArray, y: IntArray) -> float:
    """GIVEN. Mean cross-entropy over a whole dataset, without training."""
    return cross_entropy(softmax(model.forward(X)), y)


def fit(
    model: Layer,
    optimiser: Optimiser,
    split: Split,
    epochs: int,
    batch_size: int,
    rng: np.random.Generator,
    early_stopping: EarlyStopping | None = None,
) -> History:
    """Train for up to `epochs` epochs, recording all four metrics after each.

    Each epoch:
      1. for each minibatch of the training set: forward, softmax, backward of the
         cross-entropy, optimiser.step()
      2. append evaluate_loss and accuracy, on the training set and the validation set
      3. if early_stopping is given and early_stopping.update(epoch, val loss, model)
         returns True: stop
    After the loop, if early_stopping was given, restore its best weights.
    """
    # TODO
    # BEGIN SOLUTION
    history = History()
    for epoch in range(epochs):
        for X_batch, y_batch in minibatches(split.X_train, split.y_train, batch_size, rng):
            P = softmax(model.forward(X_batch))
            model.backward(cross_entropy_backward(P, y_batch))
            optimiser.step()
        history.train_loss.append(evaluate_loss(model, split.X_train, split.y_train))
        history.val_loss.append(evaluate_loss(model, split.X_val, split.y_val))
        history.train_acc.append(accuracy(model, split.X_train, split.y_train))
        history.val_acc.append(accuracy(model, split.X_val, split.y_val))
        if early_stopping is not None and early_stopping.update(epoch, history.val_loss[-1], model):
            break
    if early_stopping is not None:
        early_stopping.restore(model)
    return history
    # END SOLUTION


# --- Task 6: read the curves (THEORY §7) -------------------------------------
type Diagnosis = Literal["diverging", "underfitting", "overfitting", "healthy"]


def diagnose(history: History) -> Diagnosis:
    """Name what went wrong, checking in this order:

      "diverging"    the final training loss is not finite (nan/inf), or is higher
                     than the first epoch's
      "underfitting" final training accuracy below 0.9: it has not even learned
                     the examples it was shown
      "overfitting"  final training accuracy beats final validation accuracy by
                     more than 0.1
      "healthy"      none of the above
    """
    # TODO
    # BEGIN SOLUTION
    final = history.train_loss[-1]
    if not math.isfinite(final) or final > history.train_loss[0]:
        return "diverging"
    if history.train_acc[-1] < 0.9:
        return "underfitting"
    if history.train_acc[-1] - history.val_acc[-1] > 0.1:
        return "overfitting"
    return "healthy"
    # END SOLUTION


def digit_model(rng: np.random.Generator, hidden: int = 32) -> Sequential:
    """GIVEN. Lesson 6's digit classifier."""
    return Sequential([Dense(64, hidden, rng), ReLU(), Dense(hidden, 10, rng)])


if __name__ == "__main__":
    d = load_digits()
    rng = np.random.default_rng(0)
    print("Signal scale after 1, 5, 10, 20 ReLU layers of width 100:")
    for name, init in [("naive_uniform", naive_uniform), ("tiny_normal", tiny_normal), ("he_normal", he_normal)]:
        s = signal_scales(init, 20, 100, np.random.default_rng(0))
        print(f"  {name:14s}", [f"{s[i]:.3g}" for i in (0, 4, 9, 19)])

    print("\nOverfitting: a big model on 60 training images")
    small = Split(d.X_train[:60], d.y_train[:60], d.X_val, d.y_val)
    model = digit_model(rng, hidden=256)
    h = fit(model, SGD(model.params(), lr=0.1), small, epochs=200, batch_size=16, rng=rng)
    for e in (0, 9, 49, 199):
        print(f"  epoch {e + 1:3d}  train acc {h.train_acc[e]:.2f}  val acc {h.val_acc[e]:.2f}  val loss {h.val_loss[e]:.3f}")
    print("  diagnosis:", diagnose(h))
