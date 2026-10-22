"""Lesson 3 - NumPy and shapes bootcamp. No neural networks in this lesson.

Read THEORY.md first. Then fill in each TODO, in order, and run:

    pytest lesson03_shapes -x

Rule for this lesson: no Python `for` loops, except where a docstring asks for one.
"""

from collections.abc import Sequence

import numpy as np

from nnlib.scalar import Example
from nnlib.types import FloatArray, IntArray


# --- Task 1: columns and rows (THEORY §2, §6) --------------------------------
def as_column(v: FloatArray) -> FloatArray:
    """(n,) -> (n, 1). One value per row, in a single column."""
    # TODO: reshape with -1 for "however many rows"
    # BEGIN SOLUTION
    return v.reshape(-1, 1)
    # END SOLUTION


def as_row(v: FloatArray) -> FloatArray:
    """(n,) -> (1, n)."""
    # TODO
    # BEGIN SOLUTION
    return v.reshape(1, -1)
    # END SOLUTION


# --- Task 2: data as arrays (THEORY §1) --------------------------------------
def examples_to_arrays(examples: Sequence[Example]) -> tuple[FloatArray, FloatArray]:
    """Lesson 1's examples as a matrix and a column.

    X  (n, d_in)   row k = examples[k].x
    Y  (n, 1)      row k = [examples[k].y]   <- a column, not a flat (n,) array
    Both float64.
    """
    # TODO
    # BEGIN SOLUTION
    X = np.array([e.x for e in examples], dtype=np.float64)
    Y = np.array([e.y for e in examples], dtype=np.float64).reshape(-1, 1)
    return X, Y
    # END SOLUTION


# --- Task 3: matrix multiplication (THEORY §3) -------------------------------
def matmul_by_loops(A: FloatArray, B: FloatArray) -> FloatArray:
    """A @ B, the slow way, so you know exactly what @ computes. Loops allowed here.

        A (n, k)  B (k, m)  ->  C (n, m)
        C[r, c] = Σ_j  A[r, j] · B[j, c]      (row r of A, dotted with column c of B)

    Raise ValueError if the inner sizes differ.
    """
    # TODO: three nested loops, or two loops and a sum
    # BEGIN SOLUTION
    n, k = A.shape
    k2, m = B.shape
    if k != k2:
        raise ValueError(f"inner sizes differ: {A.shape} @ {B.shape}")
    C = np.zeros((n, m))
    for r in range(n):
        for c in range(m):
            C[r, c] = sum(A[r, j] * B[j, c] for j in range(k))
    return C
    # END SOLUTION


def weighted_sums(X: FloatArray, W: FloatArray, b: FloatArray) -> FloatArray:
    """Every neuron's weighted sum for every example, in one expression.

        X (n, d_in)   W (d_in, d_out)   b (d_out,)   ->   Z (n, d_out)
    """
    # TODO: one line
    # BEGIN SOLUTION
    return X @ W + b
    # END SOLUTION


# --- Task 4: broadcasting (THEORY §4) ----------------------------------------
def broadcast_shape(s1: tuple[int, ...], s2: tuple[int, ...]) -> tuple[int, ...]:
    """The shape NumPy gives `a + b` for arrays of shapes s1 and s2. Loops allowed.

    The rule, applied from the RIGHTMOST dimension leftwards:
      1. if one shape is shorter, pad it on the LEFT with 1s;
      2. in each position the sizes must be equal, or one of them must be 1;
      3. the result takes the larger size.
    Raise ValueError if the shapes are incompatible.
    """
    # TODO
    # BEGIN SOLUTION
    length = max(len(s1), len(s2))
    p1 = (1,) * (length - len(s1)) + s1
    p2 = (1,) * (length - len(s2)) + s2
    result: list[int] = []
    for a, b in zip(p1, p2):
        if a != b and a != 1 and b != 1:
            raise ValueError(f"cannot broadcast {s1} with {s2}")
        result.append(max(a, b))
    return tuple(result)
    # END SOLUTION


# --- Task 5: axis and keepdims (THEORY §5) -----------------------------------
def column_means(A: FloatArray) -> FloatArray:
    """(n, d) -> (d,): the mean of each column, i.e. averaged DOWN the rows."""
    # TODO
    # BEGIN SOLUTION
    return A.mean(axis=0)
    # END SOLUTION


def normalise_rows(A: FloatArray) -> FloatArray:
    """(n, d) -> (n, d): divide each row by its own sum, so every row sums to 1.

    The row sums must keep shape (n, 1) to broadcast back across each row.
    """
    # TODO
    # BEGIN SOLUTION
    return A / A.sum(axis=1, keepdims=True)
    # END SOLUTION


def one_hot(labels: IntArray, k: int) -> FloatArray:
    """(n,) integers in 0..k-1  ->  (n, k): row r is all 0s except a 1 at labels[r].

    No loops: index with  [np.arange(n), labels].
    """
    # TODO
    # BEGIN SOLUTION
    out = np.zeros((len(labels), k))
    out[np.arange(len(labels)), labels] = 1.0
    return out
    # END SOLUTION


# --- Task 6: matrix multiply as a sum over rows (THEORY §7) -------------------
def sum_of_outer_products(X: FloatArray, D: FloatArray) -> FloatArray:
    """Σ over rows r of outer(X[r], D[r]). Loop over rows allowed here.

        X (n, d_in)   D (n, d_out)   ->   (d_in, d_out)

    np.outer(u, v) is the (len(u), len(v)) table of every u[i] · v[j].
    """
    # TODO
    # BEGIN SOLUTION
    total = np.zeros((X.shape[1], D.shape[1]))
    for r in range(X.shape[0]):
        total += np.outer(X[r], D[r])
    return total
    # END SOLUTION


def sum_of_outer_products_fast(X: FloatArray, D: FloatArray) -> FloatArray:
    """The same result as sum_of_outer_products, with one matrix multiply.

    Find it by shapes: you have (n, d_in) and (n, d_out) and need (d_in, d_out).
    """
    # TODO
    # BEGIN SOLUTION
    return X.T @ D
    # END SOLUTION


# --- Task 7: the backward rule for Y = X @ W (THEORY §8) --------------------
def matmul_backward(X: FloatArray, W: FloatArray, dY: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Given dY = ∂L/∂Y for Y = X @ W, return (dX, dW).

        X (n, d_in)   W (d_in, d_out)   dY (n, d_out)
        dX must be shaped like X: (n, d_in)
        dW must be shaped like W: (d_in, d_out)

    Each is one matrix multiply with one transpose. Let the shapes tell you which.
    """
    # TODO
    # BEGIN SOLUTION
    dX = dY @ W.T
    dW = X.T @ dY
    return dX, dW
    # END SOLUTION
