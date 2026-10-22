from collections.abc import Callable

import numpy as np
import pytest

from nnlib.scalar import OR3, XOR3
from nnlib.types import FloatArray
from shapes import (
    as_column,
    as_row,
    broadcast_shape,
    column_means,
    examples_to_arrays,
    matmul_backward,
    matmul_by_loops,
    normalise_rows,
    one_hot,
    sum_of_outer_products,
    sum_of_outer_products_fast,
    weighted_sums,
)

rng = np.random.default_rng(0)


# --- Task 1 -----------------------------------------------------------------
def test_as_column_and_as_row() -> None:
    v = np.array([1.0, 2.0, 3.0])
    assert as_column(v).shape == (3, 1)
    assert as_row(v).shape == (1, 3)
    assert as_column(v)[2, 0] == 3.0


# --- Task 2 -----------------------------------------------------------------
def test_examples_to_arrays_shapes() -> None:
    X, Y = examples_to_arrays(XOR3)
    assert X.shape == (8, 3)
    assert Y.shape == (8, 1), "Y must be a column: (8, 1), not (8,)"
    assert X.dtype == np.float64 and Y.dtype == np.float64


def test_examples_to_arrays_keeps_rows_and_targets_together() -> None:
    X, Y = examples_to_arrays(OR3)
    for k, e in enumerate(OR3):
        assert tuple(X[k]) == e.x
        assert Y[k, 0] == e.y


# --- Task 3 -----------------------------------------------------------------
def test_matmul_by_loops_matches_the_at_operator() -> None:
    A, B = rng.normal(size=(4, 3)), rng.normal(size=(3, 5))
    C = matmul_by_loops(A, B)
    assert C.shape == (4, 5)
    assert C == pytest.approx(A @ B)


def test_matmul_by_loops_on_the_theory_example() -> None:
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    B = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]])
    assert matmul_by_loops(A, B) == pytest.approx(np.array([[1, 2, 0], [3, 4, 2], [5, 6, 4]]))


def test_matmul_by_loops_rejects_mismatched_inner_sizes() -> None:
    with pytest.raises(ValueError):
        matmul_by_loops(np.ones((2, 3)), np.ones((2, 3)))


def test_weighted_sums_shape_and_values() -> None:
    X, W, b = rng.normal(size=(6, 3)), rng.normal(size=(3, 4)), rng.normal(size=4)
    Z = weighted_sums(X, W, b)
    assert Z.shape == (6, 4)
    by_hand = sum(X[2, i] * W[i, 1] for i in range(3)) + b[1]
    assert Z[2, 1] == pytest.approx(by_hand), "Z[r, j] = example r through neuron j"


# --- Task 4 -----------------------------------------------------------------
@pytest.mark.parametrize(
    "s1,s2",
    [
        ((8, 4), (4,)),
        ((8, 1), (8,)),  # the classic bug: a column minus a flat array
        ((8, 4), (8, 1)),
        ((1, 3), (5, 1)),
        ((2, 1, 3), (4, 1)),
        ((), (3, 2)),
        ((7,), (7,)),
    ],
)
def test_broadcast_shape_matches_numpy(s1: tuple[int, ...], s2: tuple[int, ...]) -> None:
    assert broadcast_shape(s1, s2) == np.broadcast_shapes(s1, s2)


@pytest.mark.parametrize("s1,s2", [((8, 4), (8,)), ((3,), (4,)), ((2, 3), (3, 2))])
def test_broadcast_shape_rejects_incompatible_shapes(s1: tuple[int, ...], s2: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        broadcast_shape(s1, s2)


def test_the_column_versus_flat_trap() -> None:
    """Why Y is always a column: (8, 1) − (8,) is silently (8, 8)."""
    assert broadcast_shape((8, 1), (8,)) == (8, 8)


# --- Task 5 -----------------------------------------------------------------
def test_column_means() -> None:
    A = np.array([[1.0, 10.0], [3.0, 20.0]])
    assert column_means(A).shape == (2,)
    assert column_means(A) == pytest.approx([2.0, 15.0])


def test_normalise_rows() -> None:
    A = np.array([[1.0, 3.0], [2.0, 2.0], [0.0, 5.0]])
    out = normalise_rows(A)
    assert out.shape == (3, 2)
    assert out.sum(axis=1) == pytest.approx([1.0, 1.0, 1.0])
    assert out[0] == pytest.approx([0.25, 0.75])


def test_normalise_rows_on_a_square_matrix_divides_rows_not_columns() -> None:
    # On a square matrix, forgetting keepdims gives the right shape and the wrong answer.
    A = np.array([[1.0, 1.0], [1.0, 3.0]])
    assert normalise_rows(A) == pytest.approx(np.array([[0.5, 0.5], [0.25, 0.75]]))


def test_one_hot() -> None:
    out = one_hot(np.array([2, 0, 1, 2]), k=3)
    assert out.shape == (4, 3)
    assert out == pytest.approx(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))


# --- Task 6 -----------------------------------------------------------------
def test_sum_of_outer_products() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    D = np.array([[1.0], [10.0]])
    # outer([1,2],[1]) + outer([3,4],[10]) = [[1],[2]] + [[30],[40]]
    assert sum_of_outer_products(X, D) == pytest.approx(np.array([[31.0], [42.0]]))


def test_the_fast_version_agrees() -> None:
    X, D = rng.normal(size=(9, 4)), rng.normal(size=(9, 3))
    fast = sum_of_outer_products_fast(X, D)
    assert fast.shape == (4, 3)
    assert fast == pytest.approx(sum_of_outer_products(X, D))


# --- Task 7: the payoff ------------------------------------------------------
def numerical_grad(f: Callable[[], float], A: FloatArray, h: float = 1e-6) -> FloatArray:
    grad = np.zeros_like(A)
    for index in np.ndindex(A.shape):
        original = A[index]
        A[index] = original + h
        up = f()
        A[index] = original - h
        down = f()
        A[index] = original
        grad[index] = (up - down) / (2 * h)
    return grad


def test_matmul_backward_shapes() -> None:
    X, W, dY = rng.normal(size=(5, 3)), rng.normal(size=(3, 2)), rng.normal(size=(5, 2))
    dX, dW = matmul_backward(X, W, dY)
    assert dX.shape == X.shape, "a gradient has the shape of the thing it belongs to"
    assert dW.shape == W.shape


def test_matmul_backward_matches_the_wiggle_method() -> None:
    """Use L = Σ (Y * R) for a fixed random R, so that ∂L/∂Y = R."""
    X, W = rng.normal(size=(5, 3)), rng.normal(size=(3, 2))
    R = rng.normal(size=(5, 2))
    L = lambda: float(np.sum((X @ W) * R))  # noqa: E731
    dX, dW = matmul_backward(X, W, R)
    assert dX == pytest.approx(numerical_grad(L, X), abs=1e-6)
    assert dW == pytest.approx(numerical_grad(L, W), abs=1e-6)
