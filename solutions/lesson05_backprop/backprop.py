"""Lesson 5 - A hidden layer, and backpropagation in matrices.

Read THEORY.md first. Then fill in each TODO, in order, and run:

    pytest lesson05_backprop -x
"""

from dataclasses import dataclass

import numpy as np

from nnlib.data import examples_to_arrays
from nnlib.scalar import XOR3
from nnlib.types import FloatArray


@dataclass
class TwoLayerParams:
    """GIVEN. Every knob in the network: a hidden layer, then one output neuron."""

    W1: FloatArray  # (d_in, d_hidden)    column j = hidden neuron j's weights
    b1: FloatArray  # (d_hidden,)
    W2: FloatArray  # (d_hidden, 1)       the output neuron's weights
    b2: FloatArray  # (1,)

    @classmethod
    def random(cls, d_in: int, d_hidden: int, rng: np.random.Generator) -> "TwoLayerParams":
        return cls(
            W1=rng.uniform(-1, 1, (d_in, d_hidden)),
            b1=rng.uniform(-1, 1, d_hidden),
            W2=rng.uniform(-1, 1, (d_hidden, 1)),
            b2=rng.uniform(-1, 1, 1),
        )


@dataclass
class Cache:
    """GIVEN. Everything backward() needs from forward()."""

    X: FloatArray  # (n, d_in)
    H: FloatArray  # (n, d_hidden)   hidden layer outputs
    A: FloatArray  # (n, 1)          network outputs


@dataclass
class Grads:
    """GIVEN. One gradient per parameter, each the same shape as its parameter."""

    dW1: FloatArray  # (d_in, d_hidden)
    db1: FloatArray  # (d_hidden,)
    dW2: FloatArray  # (d_hidden, 1)
    db2: FloatArray  # (1,)


def sigmoid(Z: FloatArray) -> FloatArray:
    """GIVEN."""
    return 1.0 / (1.0 + np.exp(-Z))


def mse(A: FloatArray, Y: FloatArray) -> float:
    """GIVEN. (Lesson 4.)"""
    return float(np.mean((A - Y) ** 2))


# --- Task 1: build XOR by hand (THEORY §2) -----------------------------------
# Three hidden neurons: "at least one input on", "at least two", "all three".
# Then an output neuron that combines them as  first − second + third.
# Use weights of about ±20 so every sigmoid is decisively near 0 or 1.

# TODO: replace None with TwoLayerParams(W1=np.array(...), b1=..., W2=..., b2=...)
# BEGIN SOLUTION stub: HAND_XOR: TwoLayerParams | None = None
HAND_XOR: TwoLayerParams | None = TwoLayerParams(
    W1=np.full((3, 3), 20.0),
    b1=np.array([-10.0, -30.0, -50.0]),
    W2=np.array([[20.0], [-20.0], [20.0]]),
    b2=np.array([-10.0]),
)
# END SOLUTION


# --- Task 2: forward (THEORY §3) ---------------------------------------------
def forward(params: TwoLayerParams, X: FloatArray) -> tuple[FloatArray, Cache]:
    """Every example through both layers at once.

        H = σ(X @ W1 + b1)       (n, d_hidden)
        A = σ(H @ W2 + b2)       (n, 1)
    Return (A, Cache(X=X, H=H, A=A)).
    """
    # TODO
    # BEGIN SOLUTION
    H = sigmoid(X @ params.W1 + params.b1)
    A = sigmoid(H @ params.W2 + params.b2)
    return A, Cache(X=X, H=H, A=A)
    # END SOLUTION


def accuracy(params: TwoLayerParams, X: FloatArray, Y: FloatArray) -> float:
    """GIVEN."""
    A, _ = forward(params, X)
    return float(((A > 0.5) == (Y > 0.5)).mean())


# --- Task 3: backward (THEORY §4, §5) ----------------------------------------
def backward(params: TwoLayerParams, Y: FloatArray, cache: Cache) -> Grads:
    """Gradients of the mean squared error for all four parameters.

    Output layer (lesson 4, with H as its input):
        dA  = 2(A − Y) / n
        dZ2 = dA · A(1 − A)
        dW2 = H.T @ dZ2           db2 = dZ2 summed over rows

    Pass the blame back to the hidden layer (lesson 3 §8: dX = dY @ W.T):
        dH  = dZ2 @ W2.T

    Hidden layer (the same three lines again, with X as its input):
        dZ1 = dH · H(1 − H)
        dW1 = X.T @ dZ1           db1 = dZ1 summed over rows
    """
    # TODO
    # BEGIN SOLUTION
    X, H, A = cache.X, cache.H, cache.A
    n = X.shape[0]

    dA = 2 * (A - Y) / n
    dZ2 = dA * A * (1 - A)
    dW2 = H.T @ dZ2
    db2 = dZ2.sum(axis=0)

    dH = dZ2 @ params.W2.T
    dZ1 = dH * H * (1 - H)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)
    return Grads(dW1=dW1, db1=db1, dW2=dW2, db2=db2)
    # END SOLUTION


# --- Task 4: train (THEORY §6) -----------------------------------------------
def train(params: TwoLayerParams, X: FloatArray, Y: FloatArray, lr: float, epochs: int) -> list[float]:
    """Full-batch gradient descent on all four parameters, in place.

    Return the loss history (measured before each update).
    """
    # TODO
    # BEGIN SOLUTION
    history: list[float] = []
    for _ in range(epochs):
        A, cache = forward(params, X)
        history.append(mse(A, Y))
        g = backward(params, Y, cache)
        params.W1 -= lr * g.dW1
        params.b1 -= lr * g.db1
        params.W2 -= lr * g.dW2
        params.b2 -= lr * g.db2
    return history
    # END SOLUTION


if __name__ == "__main__":
    X, Y = examples_to_arrays(XOR3)
    params = TwoLayerParams.random(3, 4, np.random.default_rng(1))
    history = train(params, X, Y, lr=2.0, epochs=5000)
    print(f"loss {history[0]:.3f} -> {history[-1]:.5f}   accuracy {accuracy(params, X, Y):.2f}")
    A, cache = forward(params, X)
    print("inputs   target  output   hidden layer")
    for k in range(len(X)):
        print(X[k].astype(int), int(Y[k, 0]), f"  {A[k, 0]:.3f}  ", cache.H[k].round(2))
