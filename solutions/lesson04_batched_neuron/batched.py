"""Lesson 4 - One neuron on a whole batch, with matrices.

Read THEORY.md first. Then fill in each TODO, in order, and run:

    pytest lesson04_batched_neuron -x

Rule: no Python loops over examples or weights. The only loop is over epochs.
"""

from dataclasses import dataclass

import numpy as np

from nnlib.data import examples_to_arrays
from nnlib.scalar import OR3, XOR3
from nnlib.types import FloatArray


@dataclass
class NeuronParams:
    """GIVEN. One neuron, stored as a layer with a single output column."""

    W: FloatArray  # (d_in, 1)
    b: FloatArray  # (1,)

    @classmethod
    def random(cls, d_in: int, rng: np.random.Generator) -> "NeuronParams":
        return cls(W=rng.uniform(-1, 1, (d_in, 1)), b=np.zeros(1))


@dataclass
class Cache:
    """GIVEN. What the forward pass saves for the backward pass."""

    X: FloatArray  # (n, d_in)
    A: FloatArray  # (n, 1)


@dataclass
class Grads:
    """GIVEN. ∂L/∂W and ∂L/∂b. Each has the shape of its parameter."""

    dW: FloatArray  # (d_in, 1)
    db: FloatArray  # (1,)


def sigmoid(Z: FloatArray) -> FloatArray:
    """GIVEN. Element by element, so it works on any shape."""
    return 1.0 / (1.0 + np.exp(-Z))


# --- Task 1: forward (THEORY §2) ---------------------------------------------
def forward(params: NeuronParams, X: FloatArray) -> tuple[FloatArray, Cache]:
    """Run every example through the neuron at once.

        Z = X @ W + b       (n, 1)
        A = σ(Z)            (n, 1)
    Return (A, Cache(X=X, A=A)).
    """
    # TODO
    # BEGIN SOLUTION
    Z = X @ params.W + params.b
    A = sigmoid(Z)
    return A, Cache(X=X, A=A)
    # END SOLUTION


def mse(A: FloatArray, Y: FloatArray) -> float:
    """Mean squared error, as a Python float. A and Y are both (n, 1)."""
    # TODO
    # BEGIN SOLUTION
    return float(np.mean((A - Y) ** 2))
    # END SOLUTION


# --- Task 2: backward (THEORY §3) --------------------------------------------
def backward(params: NeuronParams, Y: FloatArray, cache: Cache) -> Grads:
    """Gradients of the mean squared error, for every example at once.

        dA = 2(A − Y) / n          (n, 1)   ∂L/∂A; the /n is the mean
        dZ = dA · A(1 − A)         (n, 1)   δ for every example
        dW = X.T @ dZ              (d_in, 1)
        db = dZ summed over rows   (1,)

    `params` is not needed for this one-layer case. It is in the signature
    because lesson 5's backward does need the weights.
    """
    # TODO
    # BEGIN SOLUTION
    n = cache.X.shape[0]
    dA = 2 * (cache.A - Y) / n
    dZ = dA * cache.A * (1 - cache.A)
    dW = cache.X.T @ dZ
    db = dZ.sum(axis=0)
    return Grads(dW=dW, db=db)
    # END SOLUTION


# --- Task 3: train (THEORY §4) -----------------------------------------------
def train(params: NeuronParams, X: FloatArray, Y: FloatArray, lr: float, epochs: int) -> list[float]:
    """Full-batch gradient descent. Each epoch uses every example.

    Each epoch: forward, record the loss, backward, then update in place:
        params.W -= lr * grads.dW
        params.b -= lr * grads.db
    Return the loss history (measured before each update).
    """
    # TODO
    # BEGIN SOLUTION
    history: list[float] = []
    for _ in range(epochs):
        A, cache = forward(params, X)
        history.append(mse(A, Y))
        grads = backward(params, Y, cache)
        params.W -= lr * grads.dW
        params.b -= lr * grads.db
    return history
    # END SOLUTION


def accuracy(params: NeuronParams, X: FloatArray, Y: FloatArray) -> float:
    """GIVEN."""
    A, _ = forward(params, X)
    return float(((A > 0.5) == (Y > 0.5)).mean())


if __name__ == "__main__":
    for name, data in [("OR3", OR3), ("XOR3", XOR3)]:
        X, Y = examples_to_arrays(data)
        params = NeuronParams.random(3, np.random.default_rng(0))
        history = train(params, X, Y, lr=1.0, epochs=2000)
        print(f"{name:5s} loss {history[0]:.3f} -> {history[-1]:.4f}  accuracy {accuracy(params, X, Y):.2f}")
    print("Same results as lesson 2, no loops. XOR still needs a hidden layer: lesson 5.")
