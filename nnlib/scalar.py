"""Scalar building blocks: lessons 1 and 2."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product

import numpy as np

type Inputs = tuple[float, ...]


@dataclass(frozen=True)
class Example:
    """One row of a dataset: the inputs x and the target y (0 or 1)."""

    x: Inputs
    y: int


def truth_table(rule: Callable[[Inputs], bool], d_in: int = 3) -> list[Example]:
    """Every combination of 0/1 inputs, each labelled 1 if `rule` is true for it."""
    rows = [tuple(float(bit) for bit in bits) for bits in product((0, 1), repeat=d_in)]
    return [Example(x=row, y=int(rule(row))) for row in rows]


FIRST_INPUT = truth_table(lambda x: x[0] == 1)
OR3 = truth_table(lambda x: any(x))
AND3 = truth_table(lambda x: all(x))
XOR3 = truth_table(lambda x: sum(x) % 2 == 1)


def sigmoid(z: float) -> float:
    """σ(z), safe for any z. (Lesson 1.)"""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class Neuron:
    """One neuron. (Lesson 1.)"""

    w: list[float]  # length d_in
    b: float = 0.0

    @classmethod
    def random(cls, d_in: int, rng: np.random.Generator) -> "Neuron":
        return cls(w=[float(rng.uniform(-1, 1)) for _ in range(d_in)], b=0.0)

    def weighted_sum(self, x: Inputs) -> float:
        if len(x) != len(self.w):
            raise ValueError(f"expected {len(self.w)} inputs, got {len(x)}")
        z = self.b
        for w_i, x_i in zip(self.w, x):
            z += w_i * x_i
        return z

    def forward(self, x: Inputs) -> float:
        return sigmoid(self.weighted_sum(x))

    def predict(self, x: Inputs) -> int:
        return 1 if self.forward(x) > 0.5 else 0


def accuracy(neuron: Neuron, data: list[Example]) -> float:
    """Fraction of examples predicted correctly. (Lesson 1.)"""
    return sum(1 for e in data if neuron.predict(e.x) == e.y) / len(data)


def loss(neuron: Neuron, data: list[Example]) -> float:
    """Mean squared error over the dataset. (Lesson 2.)"""
    return sum((neuron.forward(e.x) - e.y) ** 2 for e in data) / len(data)


@dataclass
class Gradients:
    """∂L/∂w for each weight, and ∂L/∂b. Shaped like a Neuron. (Lesson 2.)"""

    dw: list[float]  # length d_in, like Neuron.w
    db: float


def gradients(neuron: Neuron, data: list[Example]) -> Gradients:
    """Exact gradients of the mean squared error. (Lesson 2.)"""
    n = len(data)
    dw = [0.0] * len(neuron.w)
    db = 0.0
    for e in data:
        a = neuron.forward(e.x)
        delta = 2 * (a - e.y) * a * (1 - a)
        for i, x_i in enumerate(e.x):
            dw[i] += delta * x_i / n
        db += delta / n
    return Gradients(dw=dw, db=db)
