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
    raise NotImplementedError("TODO: implement as_column()")


def as_row(v: FloatArray) -> FloatArray:
    """(n,) -> (1, n)."""
    # TODO
    raise NotImplementedError("TODO: implement as_row()")


# --- Task 2: data as arrays (THEORY §1) --------------------------------------
def examples_to_arrays(examples: Sequence[Example]) -> tuple[FloatArray, FloatArray]:
    """Lesson 1's examples as a matrix and a column.

    X  (n, d_in)   row k = examples[k].x
    Y  (n, 1)      row k = [examples[k].y]   <- a column, not a flat (n,) array
    Both float64.
    """
    # TODO
    raise NotImplementedError("TODO: implement examples_to_arrays()")


# --- Task 3: matrix multiplication (THEORY §3) -------------------------------
def matmul_by_loops(A: FloatArray, B: FloatArray) -> FloatArray:
    """A @ B, the slow way, so you know exactly what @ computes. Loops allowed here.

        A (n, k)  B (k, m)  ->  C (n, m)
        C[r, c] = Σ_j  A[r, j] · B[j, c]      (row r of A, dotted with column c of B)

    Raise ValueError if the inner sizes differ.
    """
    # TODO: three nested loops, or two loops and a sum
    raise NotImplementedError("TODO: implement matmul_by_loops()")


def weighted_sums(X: FloatArray, W: FloatArray, b: FloatArray) -> FloatArray:
    """Every neuron's weighted sum for every example, in one expression.

        X (n, d_in)   W (d_in, d_out)   b (d_out,)   ->   Z (n, d_out)
    """
    # TODO: one line
    raise NotImplementedError("TODO: implement weighted_sums()")


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
    raise NotImplementedError("TODO: implement broadcast_shape()")


# --- Task 5: axis and keepdims (THEORY §5) -----------------------------------
def column_means(A: FloatArray) -> FloatArray:
    """(n, d) -> (d,): the mean of each column, i.e. averaged DOWN the rows."""
    # TODO
    raise NotImplementedError("TODO: implement column_means()")


def normalise_rows(A: FloatArray) -> FloatArray:
    """(n, d) -> (n, d): divide each row by its own sum, so every row sums to 1.

    The row sums must keep shape (n, 1) to broadcast back across each row.
    """
    # TODO
    raise NotImplementedError("TODO: implement normalise_rows()")


def one_hot(labels: IntArray, k: int) -> FloatArray:
    """(n,) integers in 0..k-1  ->  (n, k): row r is all 0s except a 1 at labels[r].

    No loops: index with  [np.arange(n), labels].
    """
    # TODO
    raise NotImplementedError("TODO: implement one_hot()")


# --- Task 6: matrix multiply as a sum over rows (THEORY §7) -------------------
def sum_of_outer_products(X: FloatArray, D: FloatArray) -> FloatArray:
    """Σ over rows r of outer(X[r], D[r]). Loop over rows allowed here.

        X (n, d_in)   D (n, d_out)   ->   (d_in, d_out)

    np.outer(u, v) is the (len(u), len(v)) table of every u[i] · v[j].
    """
    # TODO
    raise NotImplementedError("TODO: implement sum_of_outer_products()")


def sum_of_outer_products_fast(X: FloatArray, D: FloatArray) -> FloatArray:
    """The same result as sum_of_outer_products, with one matrix multiply.

    Find it by shapes: you have (n, d_in) and (n, d_out) and need (d_in, d_out).
    """
    # TODO
    raise NotImplementedError("TODO: implement sum_of_outer_products_fast()")


# --- Task 7: the backward rule for Y = X @ W (THEORY §8) --------------------
def matmul_backward(X: FloatArray, W: FloatArray, dY: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Given dY = ∂L/∂Y for Y = X @ W, return (dX, dW).

        X (n, d_in)   W (d_in, d_out)   dY (n, d_out)
        dX must be shaped like X: (n, d_in)
        dW must be shaped like W: (d_in, d_out)

    Each is one matrix multiply with one transpose. Let the shapes tell you which.
    """
    # TODO
    raise NotImplementedError("TODO: implement matmul_backward()")
