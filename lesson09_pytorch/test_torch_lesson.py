import math

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="install PyTorch for lesson 9: see the README")
from torch import nn  # noqa: E402

from nnlib.data import load_digits  # noqa: E402
from nnlib.layers import Dense, ReLU, Sequential  # noqa: E402
from torch_lesson import (  # noqa: E402
    DigitClassifier,
    TwoLayerTensors,
    copy_from_numpy,
    count_parameters,
    evaluate,
    forward,
    make_loader,
    to_tensors,
    torch_derivative,
    train_epoch,
    train_step,
)

DIGITS = load_digits()
XOR_X = np.array([[i >> 2 & 1, i >> 1 & 1, i & 1] for i in range(8)], dtype=np.float64)
XOR_Y = (XOR_X.sum(axis=1, keepdims=True) % 2).astype(np.float64)


# --- Task 1 -----------------------------------------------------------------
def test_to_tensors_dtypes_and_shapes() -> None:
    X, y = to_tensors(DIGITS.X_val, DIGITS.y_val)
    assert isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor)
    assert X.dtype == torch.float32, "networks use float32"
    assert y.dtype == torch.int64, "cross-entropy wants int64 class labels"
    assert tuple(X.shape) == (360, 64) and tuple(y.shape) == (360,)


# --- Task 2 -----------------------------------------------------------------
def test_torch_derivative_of_x_squared() -> None:
    result = torch_derivative(lambda x: x * x, 3.0)
    assert isinstance(result, float)
    assert result == pytest.approx(6.0)


def test_torch_derivative_does_the_lesson_0_chain_for_you() -> None:
    # L(w) = (σ(w·x) − y)², x = 2, y = 1, at w = 0.5  ->  −0.2115084 (lesson 0 THEORY §7)
    result = torch_derivative(lambda w: (torch.sigmoid(w * 2.0) - 1.0) ** 2, 0.5)
    assert result == pytest.approx(-0.2115084, abs=1e-6)


# --- Task 3 -----------------------------------------------------------------
def lesson5_grads(W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> list[np.ndarray]:
    """Your lesson 5 backward, in NumPy, as the referee."""
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    H = sig(XOR_X @ W1 + b1)
    A = sig(H @ W2 + b2)
    dZ2 = 2 * (A - XOR_Y) / 8 * A * (1 - A)
    dZ1 = (dZ2 @ W2.T) * H * (1 - H)
    return [XOR_X.T @ dZ1, dZ1.sum(0), H.T @ dZ2, dZ2.sum(0)]


def test_forward_shape() -> None:
    params = TwoLayerTensors.random(3, 4, np.random.default_rng(0))
    out = forward(params, torch.tensor(XOR_X))
    assert tuple(out.shape) == (8, 1)


def test_train_step_returns_the_loss_before_the_step() -> None:
    params = TwoLayerTensors.random(3, 4, np.random.default_rng(0))
    with torch.no_grad():
        expected = float(((forward(params, torch.tensor(XOR_X)) - torch.tensor(XOR_Y)) ** 2).mean())
    result = train_step(params, torch.tensor(XOR_X), torch.tensor(XOR_Y), lr=0.5)
    assert isinstance(result, float)
    assert result == pytest.approx(expected)


def test_pytorch_gradients_equal_your_hand_derived_ones() -> None:
    """The step PyTorch takes must be exactly the one your lesson 5 formulas give."""
    params = TwoLayerTensors.random(3, 4, np.random.default_rng(1))
    before = [p.detach().numpy().copy() for p in params.all()]
    grads = lesson5_grads(*before)
    train_step(params, torch.tensor(XOR_X), torch.tensor(XOR_Y), lr=0.5)
    for p, b, g in zip(params.all(), before, grads):
        assert p.detach().numpy() == pytest.approx(b - 0.5 * g, abs=1e-12)


def test_train_step_clears_the_gradients() -> None:
    params = TwoLayerTensors.random(3, 4, np.random.default_rng(1))
    train_step(params, torch.tensor(XOR_X), torch.tensor(XOR_Y), lr=0.5)
    for p in params.all():
        assert p.grad is not None and torch.all(p.grad == 0), "call p.grad.zero_() after the update"


def test_pytorch_learns_xor() -> None:
    params = TwoLayerTensors.random(3, 4, np.random.default_rng(1))
    X, Y = torch.tensor(XOR_X), torch.tensor(XOR_Y)
    for _ in range(5000):
        loss = train_step(params, X, Y, lr=2.0)
    assert loss < 0.02


# --- Task 4 -----------------------------------------------------------------
def test_digit_classifier_shapes() -> None:
    model = DigitClassifier()
    out = model(torch.zeros(5, 64))
    assert tuple(out.shape) == (5, 10)
    assert isinstance(model.hidden, nn.Linear) and isinstance(model.output, nn.Linear)


def test_count_parameters() -> None:
    assert count_parameters(DigitClassifier()) == 64 * 32 + 32 + 32 * 10 + 10
    assert count_parameters(nn.Linear(3, 4)) == 16


def test_pytorch_linear_stores_weights_transposed() -> None:
    """Load your lesson 6 Dense weights into PyTorch: identical outputs, once you transpose."""
    rng = np.random.default_rng(0)
    hidden, output = Dense(64, 32, rng), Dense(32, 10, rng)
    hidden.b[...] = rng.normal(size=32)
    output.b[...] = rng.normal(size=10)
    yours = Sequential([hidden, ReLU(), output]).forward(DIGITS.X_val)

    model = DigitClassifier().double()
    copy_from_numpy(model, hidden, output)
    with torch.no_grad():
        theirs = model(torch.tensor(DIGITS.X_val)).numpy()
    assert tuple(model.hidden.weight.shape) == (32, 64), "nn.Linear's weight is (d_out, d_in)"
    assert theirs == pytest.approx(yours, abs=1e-10)


# --- Task 5 -----------------------------------------------------------------
def test_train_epoch_returns_mean_loss_and_learns() -> None:
    torch.manual_seed(0)
    X, y = to_tensors(DIGITS.X_train, DIGITS.y_train)
    model = DigitClassifier()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = make_loader(X, y, batch_size=32, seed=0)
    first = train_epoch(model, loader, optimiser, nn.CrossEntropyLoss())
    second = train_epoch(model, loader, optimiser, nn.CrossEntropyLoss())
    assert isinstance(first, float) and math.isfinite(first)
    assert second < first


def test_evaluate_does_not_track_gradients() -> None:
    model = DigitClassifier()
    X, y = to_tensors(DIGITS.X_val, DIGITS.y_val)
    acc = evaluate(model, X, y)
    assert isinstance(acc, float) and 0.0 <= acc <= 1.0
    assert all(p.grad is None for p in model.parameters()), "wrap evaluation in torch.no_grad()"


# --- The payoff -------------------------------------------------------------
def test_pytorch_reads_handwriting() -> None:
    torch.manual_seed(0)
    X_train, y_train = to_tensors(DIGITS.X_train, DIGITS.y_train)
    X_val, y_val = to_tensors(DIGITS.X_val, DIGITS.y_val)
    model = DigitClassifier()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.005)
    loader = make_loader(X_train, y_train, batch_size=32, seed=0)
    for _ in range(10):
        train_epoch(model, loader, optimiser, nn.CrossEntropyLoss())
    assert evaluate(model, X_val, y_val) > 0.95
