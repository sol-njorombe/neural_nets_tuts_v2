"""Lesson 2 - Loss and gradient descent.

Read THEORY.md first. Then fill in each TODO, in order, and run:

    pytest lesson02_gradient_descent -x
"""

import copy
from dataclasses import dataclass
import statistics

import numpy as np

from nnlib.scalar import AND3, FIRST_INPUT, OR3, XOR3, Example, Neuron, accuracy


# --- Task 1: measuring wrongness (THEORY §1) ---------------------------------
def squared_error(a: float, y: float) -> float:
    """How wrong one output is: (a − y)²."""
    return (a - y) ** 2


def loss(neuron: Neuron, data: list[Example]) -> float:
    """L: the mean squared error of the neuron over every example. 0 is perfect."""
    return statistics.mean(
        squared_error(neuron.forward(e.x), e.y) for e in data
    )


# --- GIVEN: learning by trial and error (THEORY §2) ---------------------------
@dataclass
class WiggleResult:
    """GIVEN. What train_by_wiggling did, including how much work it cost."""

    history: list[float]  # loss after each round
    loss_evaluations: int  # how many full passes over the data it needed


def train_by_wiggling(neuron: Neuron, data: list[Example], rounds: int, nudge: float = 0.1) -> WiggleResult:
    """GIVEN. Nudge each knob up, then down; keep whichever lowers the loss.

    Needs no calculus, and works. Read it, then compare its cost with train().
    """
    evaluations = 0

    def measure() -> float:
        nonlocal evaluations
        evaluations += 1
        return loss(neuron, data)

    best = measure()
    history: list[float] = []
    for _ in range(rounds):
        for i in range(len(neuron.w) + 1):  # every weight, then the bias
            original = neuron.w[i] if i < len(neuron.w) else neuron.b
            for candidate in (original + nudge, original - nudge):
                _set_knob(neuron, i, candidate)
                trial = measure()
                if trial < best:
                    best = trial
                    break
                _set_knob(neuron, i, original)
        history.append(best)
    return WiggleResult(history=history, loss_evaluations=evaluations)


def _set_knob(neuron: Neuron, i: int, value: float) -> None:
    if i < len(neuron.w):
        neuron.w[i] = value
    else:
        neuron.b = value


# --- Task 2: the gradient (THEORY §3, §4) ------------------------------------
@dataclass
class Gradients:
    """GIVEN. ∂L/∂ every knob. Shaped exactly like a Neuron."""

    dw: list[float]  # ∂L/∂wᵢ for each weight, length d_in (like Neuron.w)
    db: float  # ∂L/∂b


def gradients(neuron: Neuron, data: list[Example]) -> Gradients:
    """The exact slope of the loss with respect to every weight and the bias.

    For each example, with a = neuron.forward(e.x):
        δ = 2(a − y) · a(1 − a)        (∂ squared error / ∂z for this example)
        ∂/∂wᵢ += δ · xᵢ
        ∂/∂b  += δ
    The loss is a MEAN over the examples, so divide each sum by n.
    Do not change the neuron.
    """
    def delta_of_error(a: float, y: float) -> float:
        return 2 * (a - y) * a * (1 - a)

    deltas = np.array([delta_of_error(neuron.forward(e.x), e.y) for e in data])
    partials_w = []

    for delta, ex in zip(deltas, data):
        partials_w.append([delta * xi for xi in ex.x])

    return Gradients(
        dw= np.array(partials_w).mean(axis=0).tolist(),
        db=statistics.mean(deltas)
    )


def numerical_gradients(neuron: Neuron, data: list[Example], h: float = 1e-5) -> Gradients:
    """GIVEN. The lesson 0 referee: nudge each knob both ways on a copy and measure."""
    dw: list[float] = []
    for i in range(len(neuron.w) + 1):
        up, down = copy.deepcopy(neuron), copy.deepcopy(neuron)
        _set_knob(up, i, (neuron.w[i] if i < len(neuron.w) else neuron.b) + h)
        _set_knob(down, i, (neuron.w[i] if i < len(neuron.w) else neuron.b) - h)
        dw.append((loss(up, data) - loss(down, data)) / (2 * h))
    return Gradients(dw=dw[:-1], db=dw[-1])


# --- Task 3: one step downhill (THEORY §3, §5) -------------------------------
def step(neuron: Neuron, grads: Gradients, lr: float) -> None:
    """Move every knob against its slope, in place:  knob ← knob − lr · slope."""
    for i in range(len(neuron.w)):
        neuron.w[i] -= lr * grads.dw[i]

    neuron.b -= lr * grads.db


# --- Task 4: the training loop (THEORY §5) -----------------------------------
def train(neuron: Neuron, data: list[Example], lr: float = 1.0, epochs: int = 1000) -> list[float]:
    """Repeat for `epochs` rounds: record the loss, compute gradients, take a step.

    Return the loss history, one entry per epoch, each measured BEFORE that
    epoch's step (so history[0] is the untrained neuron's loss).
    """
    losses = []
    for i in range(epochs):
        losses.append(loss(neuron, data))
        step(neuron, gradients(neuron, data), lr=lr)

    return losses


if __name__ == "__main__":
    for name, data in [("OR3", OR3), ("AND3", AND3), ("FIRST_INPUT", FIRST_INPUT), ("XOR3", XOR3)]:
        neuron = Neuron.random(3, np.random.default_rng(0))
        history = train(neuron, data, lr=1.0, epochs=2000)
        print(f"{name:12s} loss {history[0]:.3f} -> {history[-1]:.4f}   accuracy {accuracy(neuron, data):.2f}")
    print("XOR3 is stuck near 0.25 however long you train. THEORY §6 explains why.")
