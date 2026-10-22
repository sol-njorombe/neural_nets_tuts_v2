"""Lesson 8 - Autograd: numbers that do their own calculus.

Read THEORY.md first. Then fill in each TODO, in order, and run:

    pytest lesson08_autograd -x

A Value wraps one number. Every arithmetic operation on Values creates a new
Value that remembers (a) which Values it was made from and (b) a small function
that passes blame back to them. Calling .backward() on the final result replays
those small functions in reverse order. That is all autograd is.
"""

import math
from collections.abc import Callable


class Value:
    def __init__(self, data: float, children: tuple["Value", ...] = (), op: str = "") -> None:
        """GIVEN."""
        self.data = float(data)
        self.grad = 0.0  # ∂(final result)/∂(this Value); filled in by backward()
        self.children = children  # the Values this one was computed from
        self.op = op  # which operation made it, for printing and debugging
        self._backward: Callable[[], None] = lambda: None  # pushes self.grad to children

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # --- Worked example: read every line before doing the tasks -------------
    def __add__(self, other: "Value | float") -> "Value":
        """GIVEN. out = self + other."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # Local slopes: ∂out/∂self = 1, ∂out/∂other = 1.
            # Chain rule: my blame += (out's blame) × (my local slope).
            # "+=" not "=": a Value used in several places collects blame from
            # every use. Paths add (lesson 0 §6).
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    # --- Task 1: multiplication (THEORY §3) ---------------------------------
    def __mul__(self, other: "Value | float") -> "Value":
        """out = self × other.  Local slopes: ∂out/∂self = other.data,  ∂out/∂other = self.data."""
        other = other if isinstance(other, Value) else Value(other)
        # TODO: follow __add__'s pattern exactly
        # BEGIN SOLUTION
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
        # END SOLUTION

    # --- Task 2: one-input operations (THEORY §3) ---------------------------
    def __pow__(self, exponent: float) -> "Value":
        """out = self ** exponent, for a plain-number exponent.  Slope: exponent · self.data^(exponent − 1)."""
        # TODO: one child: Value(result, (self,), "**")
        # BEGIN SOLUTION
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out
        # END SOLUTION

    def exp(self) -> "Value":
        """out = e^self.  Slope: e^self, which is out.data itself."""
        # TODO
        # BEGIN SOLUTION
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out
        # END SOLUTION

    def tanh(self) -> "Value":
        """out = tanh(self), which squashes into (−1, 1).  Slope: 1 − out.data²."""
        # TODO: math.tanh for the value
        # BEGIN SOLUTION
        out = Value(math.tanh(self.data), (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward
        return out
        # END SOLUTION

    def relu(self) -> "Value":
        """out = max(0, self).  Slope: 1 if self.data > 0, else 0."""
        # TODO
        # BEGIN SOLUTION
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward() -> None:
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out
        # END SOLUTION

    # --- Task 3: backward (THEORY §4) ---------------------------------------
    def topological_order(self) -> list["Value"]:
        """Every Value that fed into this one (including itself), ordered so each
        Value comes AFTER all the Values it was computed from. Each Value appears once.

        Depth-first: visit a Value's children first, then append the Value.
        Keep a set of visited Values so shared ones are listed once.
        """
        # TODO
        # BEGIN SOLUTION
        order: list[Value] = []
        visited: set[Value] = set()

        def visit(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    visit(child)
                order.append(v)

        visit(self)
        return order
        # END SOLUTION

    def backward(self) -> None:
        """Fill in .grad for every Value that fed into this one.

        1. Set self.grad = 1.0  (∂self/∂self).
        2. Walk topological_order() in REVERSE, calling each Value's _backward().
        """
        # TODO
        # BEGIN SOLUTION
        self.grad = 1.0
        for v in reversed(self.topological_order()):
            v._backward()
        # END SOLUTION

    # --- GIVEN: built from the operations above -----------------------------
    def sigmoid(self) -> "Value":
        """σ(x) = 1 / (1 + e^−x), assembled from your operations. No new calculus."""
        return 1 / (1 + (-self).exp())

    def __neg__(self) -> "Value":
        return self * -1.0

    def __sub__(self, other: "Value | float") -> "Value":
        return self + (-other if isinstance(other, Value) else -other)

    def __truediv__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1

    def __radd__(self, other: float) -> "Value":  # other + self, when other is a plain number
        return self + other

    def __rmul__(self, other: float) -> "Value":
        return self * other

    def __rsub__(self, other: float) -> "Value":
        return Value(other) - self

    def __rtruediv__(self, other: float) -> "Value":
        return Value(other) / self
