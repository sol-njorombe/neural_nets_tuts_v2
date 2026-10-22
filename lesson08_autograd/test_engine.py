import math
from collections.abc import Callable

import numpy as np
import pytest

from engine import Value


def numerical_derivative(f: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    """Lesson 0, refereeing one last time."""
    return (f(x + h) - f(x - h)) / (2 * h)


# --- Worked example ---------------------------------------------------------
def test_add_given() -> None:
    a, b = Value(2.0), Value(3.0)
    c = a + b
    assert c.data == 5.0 and c.children == (a, b)


# --- Task 1 -----------------------------------------------------------------
def test_mul_forward_and_children() -> None:
    a, b = Value(2.0), Value(-3.0)
    c = a * b
    assert isinstance(c, Value)
    assert c.data == -6.0
    assert c.children == (a, b)


def test_mul_local_backward() -> None:
    a, b = Value(2.0), Value(-3.0)
    c = a * b
    c.grad = 10.0
    c._backward()  # pyright: ignore[reportPrivateUsage]
    assert (a.grad, b.grad) == (-30.0, 20.0), "each side's slope is the OTHER side's value, times out.grad"


def test_mul_with_plain_numbers() -> None:
    a = Value(4.0)
    assert (a * 2).data == 8.0 and (2 * a).data == 8.0


# --- Task 2 -----------------------------------------------------------------
@pytest.mark.parametrize(
    "op,plain",
    [
        (lambda v: v**3, lambda x: x**3),
        (lambda v: v**-1, lambda x: 1 / x),
        (lambda v: v.exp(), math.exp),
        (lambda v: v.tanh(), math.tanh),
        (lambda v: v.relu(), lambda x: max(0.0, x)),
    ],
    ids=["pow", "reciprocal", "exp", "tanh", "relu"],
)
@pytest.mark.parametrize("x", [-1.3, 0.7, 2.0])
def test_one_input_operations(op: Callable[[Value], Value], plain: Callable[[float], float], x: float) -> None:
    v = Value(x)
    out = op(v)
    assert out.data == pytest.approx(plain(x))
    assert out.children == (v,), "one-input operations have exactly one child"
    out.grad = 1.0
    out._backward()  # pyright: ignore[reportPrivateUsage]
    assert v.grad == pytest.approx(numerical_derivative(plain, x), abs=1e-5)


def test_sigmoid_comes_for_free() -> None:
    v = Value(0.5)
    s = v.sigmoid()
    s.backward()
    a = 1 / (1 + math.exp(-0.5))
    assert s.data == pytest.approx(a)
    assert v.grad == pytest.approx(a * (1 - a)), "lesson 0's σ' = a(1 − a), without writing it"


# --- Task 3 -----------------------------------------------------------------
def test_topological_order_lists_children_before_parents() -> None:
    a, b = Value(1.0), Value(2.0)
    c = a * b
    d = c + a
    order = d.topological_order()
    assert len(order) == 4, "each Value once, even though a is used twice"
    assert order[-1] is d
    assert order.index(c) > order.index(a) and order.index(c) > order.index(b)


def test_backward_simple_expression() -> None:
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    d = a * b + c
    d.backward()
    assert d.grad == 1.0
    assert (a.grad, b.grad, c.grad) == (-3.0, 2.0, 1.0)


def test_a_value_used_twice_collects_blame_from_both_uses() -> None:
    a = Value(3.0)
    (a + a).backward()
    assert a.grad == 2.0


def test_backward_needs_the_topological_order() -> None:
    # b is used twice by c; it must hear from both uses before passing blame to a.
    a = Value(3.0)
    b = a * 2
    c = b * b  # 4a²  ->  dc/da = 8a = 24
    c.backward()
    assert a.grad == pytest.approx(24.0)


def test_theory_worked_example() -> None:
    # THEORY §4: L = (x·w + b)² with x=2, w=3, b=-4  ->  L = 4, dL/dw = 8, dL/db = 4
    x, w, b = Value(2.0), Value(3.0), Value(-4.0)
    L = (x * w + b) ** 2
    L.backward()
    assert L.data == pytest.approx(4.0)
    assert (w.grad, b.grad, x.grad) == pytest.approx((8.0, 4.0, 12.0))


@pytest.mark.parametrize("x", [-1.5, 0.3, 2.0])
def test_backward_matches_the_wiggle_method_on_a_messy_expression(x: float) -> None:
    def plain(t: float) -> float:
        return math.tanh(t * t - 2 * t) / (t * t + 1) + 3 * math.exp(-t)

    v = Value(x)
    out = (v * v - 2 * v).tanh() / (v * v + 1) + 3 * (-v).exp()
    out.backward()
    assert out.data == pytest.approx(plain(x))
    assert v.grad == pytest.approx(numerical_derivative(plain, x), abs=1e-5)


# --- The payoff -------------------------------------------------------------
def test_autograd_reproduces_your_lesson_5_gradients() -> None:
    """Lesson 5's network on all of XOR3, built from Values. Compare with the hand-derived formulas."""
    rng = np.random.default_rng(0)
    W1, b1 = rng.uniform(-1, 1, (3, 2)), rng.uniform(-1, 1, 2)
    W2, b2 = rng.uniform(-1, 1, (2, 1)), rng.uniform(-1, 1, 1)
    X = np.array([[i >> 2 & 1, i >> 1 & 1, i & 1] for i in range(8)], dtype=float)
    Y = (X.sum(axis=1, keepdims=True) % 2).astype(float)

    # Hand-derived (lesson 5):
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    H = sig(X @ W1 + b1)
    A = sig(H @ W2 + b2)
    dZ2 = 2 * (A - Y) / 8 * A * (1 - A)
    dZ1 = (dZ2 @ W2.T) * H * (1 - H)
    expected_dW1, expected_dW2 = X.T @ dZ1, H.T @ dZ2

    # Autograd: just write the forward pass.
    vW1 = [[Value(W1[i, j]) for j in range(2)] for i in range(3)]
    vb1 = [Value(b) for b in b1]
    vW2 = [Value(W2[j, 0]) for j in range(2)]
    vb2 = Value(b2[0])
    total = Value(0.0)
    for x, y in zip(X, Y[:, 0]):
        h = [(sum((vW1[i][j] * float(x[i]) for i in range(3)), vb1[j])).sigmoid() for j in range(2)]
        a = (sum((vW2[j] * h[j] for j in range(2)), vb2)).sigmoid()
        total = total + (a - float(y)) ** 2
    (total / 8).backward()

    assert np.array([[v.grad for v in row] for row in vW1]) == pytest.approx(expected_dW1, abs=1e-10)
    assert np.array([v.grad for v in vW2]) == pytest.approx(expected_dW2[:, 0], abs=1e-10)


def test_a_two_hidden_layer_network_learns_xor_with_no_handwritten_gradients() -> None:
    from tiny_nn import MLP, XOR3, train

    model = MLP(3, [4, 4, 1], np.random.default_rng(1))
    history = train(model, XOR3, lr=0.1, epochs=1000)
    assert history[-1] < 0.01
    for x, y in XOR3.items():
        assert round(model(x).data) == y
