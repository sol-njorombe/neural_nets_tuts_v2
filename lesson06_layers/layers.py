"""Lesson 6 - Layers as bricks, softmax, minibatches, and real data.

Read THEORY.md first. Then fill in each TODO, in order, and run:

    pytest lesson06_layers -x

Every layer obeys one contract:

    forward(X)   -> its output; remembers whatever backward needs
    backward(dY) -> takes ∂L/∂(its output), stores ∂L/∂(its own params),
                    returns ∂L/∂(its input): the blame it passes back
"""

from collections.abc import Iterator

import numpy as np

from nnlib.data import load_digits
from nnlib.types import BoolArray, FloatArray, IntArray

type ParamPair = tuple[FloatArray, FloatArray]
"""(parameter, its gradient). Same shape. Both updated in place, never replaced."""


class Layer:
    """GIVEN. The base class. Layers with no parameters (activations) keep params() as is."""

    def forward(self, X: FloatArray) -> FloatArray:
        raise NotImplementedError

    def backward(self, dY: FloatArray) -> FloatArray:
        raise NotImplementedError

    def params(self) -> list[ParamPair]:
        return []


# --- Task 1: Dense (THEORY §2) -----------------------------------------------
class Dense(Layer):
    """Fully connected: every input feeds every neuron. Lesson 5's W1/b1, as a brick."""

    def __init__(self, d_in: int, d_out: int, rng: np.random.Generator) -> None:
        """GIVEN. Lesson 7 explains this initialisation ("He"): it keeps signals a sensible size."""
        self.W: FloatArray = rng.normal(0.0, np.sqrt(2.0 / d_in), (d_in, d_out))  # (d_in, d_out)
        self.b: FloatArray = np.zeros(d_out)  # (d_out,)
        self.dW: FloatArray = np.zeros_like(self.W)  # (d_in, d_out)
        self.db: FloatArray = np.zeros_like(self.b)  # (d_out,)
        self.X: FloatArray = np.zeros((0, d_in))  # (n, d_in): forward's input, saved for backward

    def params(self) -> list[ParamPair]:
        """GIVEN."""
        return [(self.W, self.dW), (self.b, self.db)]

    def forward(self, X: FloatArray) -> FloatArray:
        """Save X on self, return X @ W + b."""
        # TODO
        raise NotImplementedError("TODO: implement forward()")

    def backward(self, dY: FloatArray) -> FloatArray:
        """Fill self.dW and self.db IN PLACE, return dX. (Lesson 3 §8.)

        In place means `self.dW[...] = ...`. A plain `self.dW = ...` makes a new
        array and leaves the one handed out by params() unchanged.
        """
        # TODO
        raise NotImplementedError("TODO: implement backward()")


# --- Task 2: activations (THEORY §3) -----------------------------------------
class ReLU(Layer):
    """max(0, x), elementwise. Slope 1 where the input was positive, 0 elsewhere."""

    def __init__(self) -> None:
        """GIVEN."""
        self.mask: BoolArray = np.zeros((0, 0), dtype=np.bool_)  # where the input was > 0

    def forward(self, X: FloatArray) -> FloatArray:
        # TODO: save the mask, then zero the negatives
        raise NotImplementedError("TODO: implement forward()")

    def backward(self, dY: FloatArray) -> FloatArray:
        # TODO
        raise NotImplementedError("TODO: implement backward()")


class Sigmoid(Layer):
    """σ(x), elementwise. Save the OUTPUT: the slope is A(1 − A)."""

    def __init__(self) -> None:
        """GIVEN."""
        self.A: FloatArray = np.zeros((0, 0))

    def forward(self, X: FloatArray) -> FloatArray:
        # TODO
        raise NotImplementedError("TODO: implement forward()")

    def backward(self, dY: FloatArray) -> FloatArray:
        # TODO
        raise NotImplementedError("TODO: implement backward()")


# --- Task 3: softmax and cross-entropy (THEORY §4) ---------------------------
def softmax(logits: FloatArray) -> FloatArray:
    """(n, k) scores -> (n, k) probabilities; each row is positive and sums to 1.

    Subtract each row's max first (keepdims=True): np.exp(1000) is inf.
    """
    # TODO
    raise NotImplementedError("TODO: implement softmax()")


def cross_entropy(P: FloatArray, labels: IntArray) -> float:
    """Mean over rows of −log(P[row, correct class]), as a float.

        P (n, k) probabilities    labels (n,) integers in 0..k-1
    Add 1e-12 inside the log so a probability of exactly 0 gives a large number, not inf.
    """
    # TODO
    raise NotImplementedError("TODO: implement cross_entropy()")


def cross_entropy_backward(P: FloatArray, labels: IntArray) -> FloatArray:
    """∂(mean cross-entropy)/∂logits, straight through the softmax: (P − one_hot) / n.

    Do not modify P: work on a copy.
    """
    # TODO
    raise NotImplementedError("TODO: implement cross_entropy_backward()")


# --- Task 4: a network is a list (THEORY §5) ---------------------------------
class Sequential(Layer):
    """A stack of layers, any number, in any order. Itself a Layer."""

    def __init__(self, layers: list[Layer]) -> None:
        """GIVEN."""
        self.layers = layers

    def forward(self, X: FloatArray) -> FloatArray:
        """Feed X through every layer in order."""
        # TODO
        raise NotImplementedError("TODO: implement forward()")

    def backward(self, dY: FloatArray) -> FloatArray:
        """Feed the blame through every layer in REVERSE order."""
        # TODO
        raise NotImplementedError("TODO: implement backward()")

    def params(self) -> list[ParamPair]:
        """GIVEN. Every (parameter, gradient) pair from every layer."""
        return [pair for layer in self.layers for pair in layer.params()]


def sgd_step(model: Layer, lr: float) -> None:
    """Move every parameter downhill, IN PLACE: `p -= lr * g`, not `p = p - lr * g`."""
    # TODO
    raise NotImplementedError("TODO: implement sgd_step()")


# --- Task 5: minibatches and the training loop (THEORY §6) -------------------
def minibatches(
    X: FloatArray, y: IntArray, batch_size: int, rng: np.random.Generator
) -> Iterator[tuple[FloatArray, IntArray]]:
    """Yield (X_batch, y_batch) pairs that together cover every row exactly once.

    Shuffle with rng.permutation(len(X)), then slice into chunks of batch_size.
    The last chunk may be smaller. Use `yield`.
    """
    # TODO
    raise NotImplementedError("TODO: implement minibatches()")


def train_epoch(
    model: Layer, X: FloatArray, y: IntArray, lr: float, batch_size: int, rng: np.random.Generator
) -> float:
    """One pass over the data in minibatches. Return the mean of the batch losses.

    For each batch: logits → softmax → record cross-entropy → backward → sgd_step.
    """
    # TODO
    raise NotImplementedError("TODO: implement train_epoch()")


def accuracy(model: Layer, X: FloatArray, y: IntArray) -> float:
    """Fraction of rows whose highest-scoring class is the correct one."""
    # TODO: np.argmax(..., axis=1)
    raise NotImplementedError("TODO: implement accuracy()")


if __name__ == "__main__":
    digits = load_digits()
    rng = np.random.default_rng(0)
    model = Sequential([Dense(64, 32, rng), ReLU(), Dense(32, 10, rng)])
    for epoch in range(1, 31):
        mean_loss = train_epoch(model, digits.X_train, digits.y_train, lr=0.1, batch_size=32, rng=rng)
        if epoch in (1, 5, 10, 20, 30):
            train_acc = accuracy(model, digits.X_train, digits.y_train)
            val_acc = accuracy(model, digits.X_val, digits.y_val)
            print(f"epoch {epoch:2d}  loss {mean_loss:.3f}  train acc {train_acc:.3f}  val acc {val_acc:.3f}")
