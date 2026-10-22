"""Datasets as arrays: lesson 3 onward."""

import pathlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nnlib.scalar import Example
from nnlib.types import FloatArray, IntArray

_DIGITS = pathlib.Path(__file__).parent / "digits.npz"


def examples_to_arrays(examples: Sequence[Example]) -> tuple[FloatArray, FloatArray]:
    """A list of Examples as X (n, d_in) and a column of targets Y (n, 1). (Lesson 3.)"""
    X = np.array([e.x for e in examples], dtype=np.float64)  # (n, d_in)
    Y = np.array([[e.y] for e in examples], dtype=np.float64)  # (n, 1)
    return X, Y


@dataclass(frozen=True)
class Digits:
    """Handwritten digits, split into training and validation sets.

    1,797 images of the digits 0-9, each 8×8 pixels flattened to 64 numbers
    scaled to 0..1. From the UCI "Optical Recognition of Handwritten Digits"
    dataset (CC BY 4.0), the copy scikit-learn ships as load_digits.
    """

    X_train: FloatArray  # (1437, 64)
    y_train: IntArray  # (1437,)  the digit each image shows
    X_val: FloatArray  # (360, 64)
    y_val: IntArray  # (360,)


def load_digits(n_val: int = 360, seed: int = 0) -> Digits:
    """Load the digits with a fixed shuffle. Never train on the validation rows."""
    with np.load(_DIGITS) as f:
        X = f["images"].astype(np.float64) / 16.0
        y = f["labels"].astype(np.int64)
    order = np.random.default_rng(seed).permutation(len(X))
    X, y = X[order], y[order]
    return Digits(X_train=X[n_val:], y_train=y[n_val:], X_val=X[:n_val], y_val=y[:n_val])


def show_digit(image: FloatArray) -> str:
    """One 64-number row as ASCII art."""
    shades = " .:-=+*#%@"
    lines = ["".join(shades[int(v * 9)] * 2 for v in row) for row in image.reshape(8, 8)]
    return "\n".join(lines)
