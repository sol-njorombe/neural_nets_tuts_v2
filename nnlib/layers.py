"""Layers, losses and training helpers: lesson 6 (and lesson 7's He initialisation)."""

from collections.abc import Iterator

import numpy as np

from nnlib.types import BoolArray, FloatArray, IntArray

type ParamPair = tuple[FloatArray, FloatArray]
"""(parameter array, its gradient array). Same shape. Updated in place."""


class Layer:
    """Base class. forward() remembers what backward() needs."""

    def forward(self, X: FloatArray) -> FloatArray:
        raise NotImplementedError

    def backward(self, dY: FloatArray) -> FloatArray:
        """Takes ∂L/∂(output); stores ∂L/∂(own params); returns ∂L/∂(input)."""
        raise NotImplementedError

    def params(self) -> list[ParamPair]:
        return []


class Dense(Layer):
    def __init__(self, d_in: int, d_out: int, rng: np.random.Generator) -> None:
        self.W: FloatArray = rng.normal(0.0, np.sqrt(2.0 / d_in), (d_in, d_out))  # (d_in, d_out)
        self.b: FloatArray = np.zeros(d_out)  # (d_out,)
        self.dW: FloatArray = np.zeros_like(self.W)  # (d_in, d_out)
        self.db: FloatArray = np.zeros_like(self.b)  # (d_out,)
        self._X: FloatArray = np.zeros((0, d_in))  # (n, d_in), saved by forward

    def forward(self, X: FloatArray) -> FloatArray:
        self._X = X
        return X @ self.W + self.b

    def backward(self, dY: FloatArray) -> FloatArray:
        self.dW[...] = self._X.T @ dY
        self.db[...] = dY.sum(axis=0)
        return dY @ self.W.T

    def params(self) -> list[ParamPair]:
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU(Layer):
    def __init__(self) -> None:
        self._mask: BoolArray = np.zeros((0, 0), dtype=np.bool_)

    def forward(self, X: FloatArray) -> FloatArray:
        self._mask = X > 0
        return X * self._mask

    def backward(self, dY: FloatArray) -> FloatArray:
        return dY * self._mask


class Sigmoid(Layer):
    def __init__(self) -> None:
        self._A: FloatArray = np.zeros((0, 0))

    def forward(self, X: FloatArray) -> FloatArray:
        self._A = 1.0 / (1.0 + np.exp(-X))
        return self._A

    def backward(self, dY: FloatArray) -> FloatArray:
        return dY * self._A * (1 - self._A)


class Sequential(Layer):
    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers

    def forward(self, X: FloatArray) -> FloatArray:
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dY: FloatArray) -> FloatArray:
        for layer in reversed(self.layers):
            dY = layer.backward(dY)
        return dY

    def params(self) -> list[ParamPair]:
        return [pair for layer in self.layers for pair in layer.params()]


def softmax(logits: FloatArray) -> FloatArray:
    """(n, k) scores -> (n, k) probabilities, each row summing to 1."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(P: FloatArray, labels: IntArray) -> float:
    """Mean of −log(probability given to the correct class)."""
    n = len(labels)
    return float(-np.mean(np.log(P[np.arange(n), labels] + 1e-12)))


def cross_entropy_backward(P: FloatArray, labels: IntArray) -> FloatArray:
    """∂(mean cross-entropy)/∂logits = (P − one_hot(labels)) / n."""
    n = len(labels)
    dZ = P.copy()
    dZ[np.arange(n), labels] -= 1.0
    return dZ / n


def sgd_step(model: Layer, lr: float) -> None:
    for p, g in model.params():
        p -= lr * g


def minibatches(
    X: FloatArray, y: IntArray, batch_size: int, rng: np.random.Generator
) -> Iterator[tuple[FloatArray, IntArray]]:
    order = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        chosen = order[start : start + batch_size]
        yield X[chosen], y[chosen]


def train_epoch(
    model: Layer, X: FloatArray, y: IntArray, lr: float, batch_size: int, rng: np.random.Generator
) -> float:
    losses: list[float] = []
    for X_batch, y_batch in minibatches(X, y, batch_size, rng):
        P = softmax(model.forward(X_batch))
        losses.append(cross_entropy(P, y_batch))
        model.backward(cross_entropy_backward(P, y_batch))
        sgd_step(model, lr)
    return float(np.mean(losses))


def accuracy(model: Layer, X: FloatArray, y: IntArray) -> float:
    predicted: IntArray = np.argmax(model.forward(X), axis=1)  # (n,)
    return float((predicted == y).mean())
