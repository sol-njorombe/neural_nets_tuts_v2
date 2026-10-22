"""Lesson 1 - One neuron, as a typed class.

Read THEORY.md first. Then fill in each TODO, in order, and run:

    pytest lesson01_neuron -x
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product

import numpy as np

type Inputs = tuple[float, ...]
"""One example's inputs, e.g. (1.0, 0.0, 1.0). Length d_in."""


# --- GIVEN: the data ---------------------------------------------------------
@dataclass(frozen=True)
class Example:
    """One row of a dataset: the inputs x, and the target y we want the neuron to output."""

    x: Inputs
    y: int  # 0 or 1


def truth_table(rule: Callable[[Inputs], bool], d_in: int = 3) -> list[Example]:
    """GIVEN. Every combination of 0/1 inputs, each labelled 1 if `rule` is true for it."""
    rows = [tuple(float(bit) for bit in bits) for bits in product((0, 1), repeat=d_in)]
    return [Example(x=row, y=int(rule(row))) for row in rows]


FIRST_INPUT = truth_table(lambda x: x[0] == 1)  # the answer is just the first input
OR3 = truth_table(lambda x: any(x))  # 1 if at least one input is on
AND3 = truth_table(lambda x: all(x))  # 1 only if all three are on
XOR3 = truth_table(lambda x: sum(x) % 2 == 1)  # 1 if an odd number are on: the course's villain


# --- Task 1: sigmoid (THEORY §3) ---------------------------------------------
def sigmoid(z: float) -> float:
    """σ(z) = 1 / (1 + e^-z), safe for any z.

    math.exp(1000) raises OverflowError, so the textbook formula crashes on
    sigmoid(-1000). THEORY §3 shows an equivalent form for negative z.
    """
    # TODO
    # BEGIN SOLUTION
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)
    # END SOLUTION


# --- Tasks 2-4: the neuron (THEORY §1, §2, §4) -------------------------------
@dataclass
class Neuron:
    """One neuron: a weight per input, and a bias.

    Not frozen: from lesson 2 on, training changes w and b in place.
    """

    w: list[float]  # one weight per input, length d_in
    b: float = 0.0

    @classmethod
    def random(cls, d_in: int, rng: np.random.Generator) -> "Neuron":
        """Task 2. A neuron with d_in weights drawn from rng.uniform(-1, 1), and bias 0.

        Convert each draw with float(...): rng returns a NumPy scalar.
        """
        # TODO
        # BEGIN SOLUTION
        return cls(w=[float(rng.uniform(-1, 1)) for _ in range(d_in)], b=0.0)
        # END SOLUTION

    def weighted_sum(self, x: Inputs) -> float:
        """Task 3. z = w₀x₀ + w₁x₁ + ... + b

        Raise ValueError if len(x) != len(self.w). (zip() would silently stop
        at the shorter one, hiding the bug. THEORY §5.)
        """
        # TODO
        # BEGIN SOLUTION
        if len(x) != len(self.w):
            raise ValueError(f"expected {len(self.w)} inputs, got {len(x)}")
        z = self.b
        for w_i, x_i in zip(self.w, x):
            z += w_i * x_i
        return z
        # END SOLUTION

    def forward(self, x: Inputs) -> float:
        """Task 3. a = σ(z): the neuron's output, between 0 and 1."""
        # TODO
        # BEGIN SOLUTION
        return sigmoid(self.weighted_sum(x))
        # END SOLUTION

    def predict(self, x: Inputs) -> int:
        """Task 3. The yes/no answer: 1 if a > 0.5, else 0."""
        # TODO
        # BEGIN SOLUTION
        return 1 if self.forward(x) > 0.5 else 0
        # END SOLUTION


def accuracy(neuron: Neuron, data: list[Example]) -> float:
    """Task 4. The fraction of examples the neuron predicts correctly, from 0.0 to 1.0."""
    # TODO
    # BEGIN SOLUTION
    correct = sum(1 for example in data if neuron.predict(example.x) == example.y)
    return correct / len(data)
    # END SOLUTION


# --- Task 5: be the learning algorithm (THEORY §4) ---------------------------
# Choose weights and a bias BY HAND so the neuron computes each function.
# Think of the bias as the hurdle the weighted sum has to clear.

# TODO: replace None with Neuron(w=[?, ?, ?], b=?)
# BEGIN SOLUTION stub: HAND_OR: Neuron | None = None
HAND_OR: Neuron | None = Neuron(w=[10.0, 10.0, 10.0], b=-5.0)
# END SOLUTION

# TODO: replace None with Neuron(w=[?, ?, ?], b=?)
# BEGIN SOLUTION stub: HAND_AND: Neuron | None = None
HAND_AND: Neuron | None = Neuron(w=[10.0, 10.0, 10.0], b=-25.0)
# END SOLUTION


if __name__ == "__main__":
    # Run this file directly to watch a random neuron be confidently wrong.
    neuron = Neuron.random(3, np.random.default_rng(0))
    print(neuron)
    for example in XOR3:
        print(example.x, "want", example.y, "got", round(neuron.forward(example.x), 3))
    print("accuracy on XOR3:", accuracy(neuron, XOR3))
