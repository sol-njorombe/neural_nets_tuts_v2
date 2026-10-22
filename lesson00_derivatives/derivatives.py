"""Lesson 0 - Derivatives, partial derivatives and the chain rule.

Read THEORY.md first. Then fill in each TODO, in order, and run:

    pytest lesson00_derivatives -x
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

type ScalarFn = Callable[[float], float]
"""A function of one number. Example: lambda x: x * x"""

type MultiFn = Callable[[Sequence[float]], float]
"""A function of several numbers, passed as one sequence. Example: lambda v: v[0] * v[1]"""


def sigmoid(z: float) -> float:
    """GIVEN. σ(z) = 1 / (1 + e^-z). Squashes any number into the range (0, 1)."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)  # for negative z, this form avoids overflowing math.exp
    return e / (1.0 + e)


# --- Task 1: the numerical derivative (THEORY §2) ----------------------------
def numerical_derivative(f: ScalarFn, x: float, h: float = 1e-5) -> float:
    """Estimate df/dx at x by nudging x both ways: the central difference.

        df/dx ≈ (f(x + h) - f(x - h)) / (2h)

    >>> numerical_derivative(lambda x: x * x, 3.0)   # ≈ 6.0
    """
    # TODO
    raise NotImplementedError("TODO: implement numerical_derivative()")


# --- Task 2: analytic derivatives (THEORY §3) --------------------------------
def d_square(x: float) -> float:
    """The exact derivative of x², written by hand."""
    # TODO
    raise NotImplementedError("TODO: implement d_square()")


def d_sigmoid(z: float) -> float:
    """The exact derivative of σ(z).

    THEORY §5 derives the neat form: if a = σ(z), then dσ/dz = a(1 - a).
    """
    # TODO
    raise NotImplementedError("TODO: implement d_sigmoid()")


# --- Task 3: the chain rule (THEORY §5) --------------------------------------
def chain_rule(d_outer: ScalarFn, inner: ScalarFn, d_inner: ScalarFn, x: float) -> float:
    """d/dx of outer(inner(x)).

        d_outer  - the derivative of the outside function
        inner    - the inside function
        d_inner  - the derivative of the inside function

    Slope of the outer function, measured AT inner(x), times the slope of the inner one.
    """
    # TODO
    raise NotImplementedError("TODO: implement chain_rule()")


# --- Task 4: partial derivatives and the gradient (THEORY §4) ----------------
def partial_derivative(f: MultiFn, args: Sequence[float], i: int, h: float = 1e-5) -> float:
    """∂f/∂(args[i]): nudge only input i, hold every other input still.

    Do not modify the caller's sequence: work on copies.
    """
    # TODO
    raise NotImplementedError("TODO: implement partial_derivative()")


def numerical_gradient(f: MultiFn, args: Sequence[float], h: float = 1e-5) -> list[float]:
    """∇f: every partial derivative of f at args, in order. Length = len(args)."""
    # TODO
    raise NotImplementedError("TODO: implement numerical_gradient()")


# --- Task 5: several paths (THEORY §6) ---------------------------------------
def sum_over_paths(paths: Sequence[Sequence[float]]) -> float:
    """The total derivative when an input reaches the output by several routes.

    Each path is the list of local slopes met along one route, input to output.
    Multiply along each path, then add the paths together.

    >>> sum_over_paths([[2.0, 3.0], [4.0]])   # 2*3 + 4 = 10
    """
    # TODO
    raise NotImplementedError("TODO: implement sum_over_paths()")


# --- Task 6: backprop on a one-weight network (THEORY §7) --------------------
@dataclass(frozen=True)
class Trace:
    """GIVEN. Every value the forward pass computed, kept for the backward pass."""

    z: float  # weighted input: z = w·x
    a: float  # activation: a = σ(z)
    L: float  # loss: L = (a - y)²


def tiny_forward(w: float, x: float, y: float) -> Trace:
    """Run the smallest possible network forward and record each step.

        z = w·x      a = σ(z)      L = (a - y)²
    """
    # TODO
    raise NotImplementedError("TODO: implement tiny_forward()")


def tiny_backward(trace: Trace, x: float, y: float) -> float:
    """dL/dw, using only the values saved in the trace. No calls to sigmoid().

    Multiply the three local slopes (THEORY §7):
        dL/da = 2(a - y)      da/dz = a(1 - a)      dz/dw = x
    """
    # TODO
    raise NotImplementedError("TODO: implement tiny_backward()")
