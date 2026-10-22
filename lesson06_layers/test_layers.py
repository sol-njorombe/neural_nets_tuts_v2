from collections.abc import Callable

import numpy as np
import pytest

from layers import (
    Dense,
    Layer,
    ReLU,
    Sequential,
    Sigmoid,
    accuracy,
    cross_entropy,
    cross_entropy_backward,
    minibatches,
    sgd_step,
    softmax,
    train_epoch,
)
from nnlib.data import load_digits
from nnlib.types import FloatArray


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


def check_input_gradient(layer: Layer, X: FloatArray) -> None:
    """Stand-in loss L = Σ(output · R), so ∂L/∂output = R."""
    R = np.random.default_rng(99).normal(size=layer.forward(X).shape)
    layer.forward(X)
    exact = layer.backward(R)
    assert exact.shape == X.shape, "backward returns blame shaped like the layer's input"
    assert exact == pytest.approx(numerical_grad(lambda: float(np.sum(layer.forward(X) * R)), X), abs=1e-6)


# --- Task 1 -----------------------------------------------------------------
def test_dense_forward() -> None:
    layer = Dense(3, 2, np.random.default_rng(0))
    layer.W[...] = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    layer.b[...] = [0.5, -0.5]
    assert layer.forward(np.array([[1.0, 2.0, 3.0]])) == pytest.approx(np.array([[4.5, 4.5]]))


def test_dense_input_gradient() -> None:
    rng = np.random.default_rng(0)
    check_input_gradient(Dense(4, 3, rng), rng.normal(size=(5, 4)))


def test_dense_parameter_gradients() -> None:
    rng = np.random.default_rng(1)
    layer, X, R = Dense(4, 3, rng), rng.normal(size=(5, 4)), rng.normal(size=(5, 3))
    L = lambda: float(np.sum(layer.forward(X) * R))  # noqa: E731
    layer.forward(X)
    layer.backward(R)
    assert layer.dW.shape == layer.W.shape and layer.db.shape == layer.b.shape
    assert layer.dW == pytest.approx(numerical_grad(L, layer.W), abs=1e-6)
    assert layer.db == pytest.approx(numerical_grad(L, layer.b), abs=1e-6)


def test_dense_fills_its_gradients_in_place() -> None:
    rng = np.random.default_rng(0)
    layer = Dense(4, 3, rng)
    (_, dW_handed_out), (_, db_handed_out) = layer.params()
    layer.forward(rng.normal(size=(5, 4)))
    layer.backward(rng.normal(size=(5, 3)))
    assert dW_handed_out is layer.dW and db_handed_out is layer.db, "use self.dW[...] = ..., not self.dW = ..."
    assert np.any(dW_handed_out != 0)


# --- Task 2 -----------------------------------------------------------------
def test_relu() -> None:
    relu = ReLU()
    assert relu.forward(np.array([[-2.0, 0.0, 3.0]])) == pytest.approx(np.array([[0.0, 0.0, 3.0]]))
    assert relu.backward(np.array([[5.0, 5.0, 5.0]])) == pytest.approx(np.array([[0.0, 0.0, 5.0]]))


def test_relu_input_gradient() -> None:
    check_input_gradient(ReLU(), np.random.default_rng(0).normal(size=(4, 3)))


def test_sigmoid_layer() -> None:
    assert Sigmoid().forward(np.zeros((1, 1))) == pytest.approx(np.array([[0.5]]))
    check_input_gradient(Sigmoid(), np.random.default_rng(0).normal(size=(4, 3)))


# --- Task 3 -----------------------------------------------------------------
def test_softmax_rows_are_probabilities() -> None:
    P = softmax(np.random.default_rng(0).normal(size=(6, 10)))
    assert P.shape == (6, 10)
    assert P.sum(axis=1) == pytest.approx(np.ones(6))
    assert np.all(P > 0)


def test_softmax_known_values() -> None:
    P = softmax(np.array([[0.0, 0.0], [0.0, np.log(3.0)]]))
    assert P == pytest.approx(np.array([[0.5, 0.5], [0.25, 0.75]]))


def test_softmax_survives_huge_scores() -> None:
    P = softmax(np.array([[1000.0, 1000.0], [-1000.0, 0.0]]))
    assert not np.any(np.isnan(P)), "subtract each row's max before np.exp"
    assert P == pytest.approx(np.array([[0.5, 0.5], [0.0, 1.0]]))


def test_cross_entropy_known_values() -> None:
    value = cross_entropy(np.array([[0.5, 0.5], [0.25, 0.75]]), np.array([0, 1]))
    assert isinstance(value, float)
    assert value == pytest.approx(-(np.log(0.5) + np.log(0.75)) / 2)


def test_cross_entropy_punishes_confident_mistakes() -> None:
    confident = np.array([[0.99, 0.01]])
    assert cross_entropy(confident, np.array([0])) < 0.02
    assert cross_entropy(confident, np.array([1])) > 4.0


def test_cross_entropy_backward_matches_the_wiggle_method() -> None:
    rng = np.random.default_rng(0)
    logits, labels = rng.normal(size=(5, 4)), np.array([0, 3, 1, 1, 2])
    exact = cross_entropy_backward(softmax(logits), labels)
    assert exact == pytest.approx(numerical_grad(lambda: cross_entropy(softmax(logits), labels), logits), abs=1e-6)


def test_cross_entropy_backward_leaves_P_alone() -> None:
    P = softmax(np.zeros((2, 3)))
    before = P.copy()
    cross_entropy_backward(P, np.array([0, 1]))
    assert np.array_equal(P, before), "copy P before subtracting"


# --- Task 4 -----------------------------------------------------------------
def deep_model(seed: int = 0) -> Sequential:
    rng = np.random.default_rng(seed)
    return Sequential([Dense(6, 5, rng), ReLU(), Dense(5, 4, rng), Sigmoid(), Dense(4, 3, rng)])


def test_sequential_forward_chains_the_layers() -> None:
    model = deep_model()
    X = np.random.default_rng(1).normal(size=(7, 6))
    expected = X
    for layer in model.layers:
        expected = layer.forward(expected)
    assert model.forward(X) == pytest.approx(expected)


def test_every_gradient_in_a_deep_stack_is_correct() -> None:
    """Three Dense layers, two kinds of activation, and no new derivation from you."""
    model = deep_model()
    rng = np.random.default_rng(1)
    X, labels = rng.normal(size=(7, 6)), rng.integers(0, 3, 7)
    L = lambda: cross_entropy(softmax(model.forward(X)), labels)  # noqa: E731
    model.backward(cross_entropy_backward(softmax(model.forward(X)), labels))
    exact = [g.copy() for _, g in model.params()]
    for (p, _), g in zip(model.params(), exact):
        assert g == pytest.approx(numerical_grad(L, p), abs=1e-6)


def test_sgd_step_updates_the_real_weights() -> None:
    model = deep_model()
    first = model.layers[0]
    assert isinstance(first, Dense)
    before = first.W.copy()
    first.dW[...] = 1.0
    sgd_step(model, lr=0.1)
    assert first.W == pytest.approx(before - 0.1), "update in place: p -= lr * g"


# --- Task 5 -----------------------------------------------------------------
def test_minibatches_cover_every_row_exactly_once() -> None:
    X, y = np.arange(10.0).reshape(10, 1), np.arange(10)
    batches = list(minibatches(X, y, batch_size=4, rng=np.random.default_rng(0)))
    assert [len(xb) for xb, _ in batches] == [4, 4, 2]
    assert sorted(np.concatenate([yb for _, yb in batches]).tolist()) == list(range(10))


def test_minibatches_shuffle_but_keep_rows_and_labels_together() -> None:
    X, y = np.arange(50.0).reshape(50, 1), np.arange(50)
    batches = list(minibatches(X, y, batch_size=10, rng=np.random.default_rng(0)))
    assert np.concatenate([yb for _, yb in batches]).tolist() != list(range(50)), "shuffle first"
    for xb, yb in batches:
        assert np.array_equal(xb[:, 0], yb)


def test_accuracy() -> None:
    model = Sequential([Dense(2, 2, np.random.default_rng(0))])
    first = model.layers[0]
    assert isinstance(first, Dense)
    first.W[...] = np.eye(2)
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert accuracy(model, X, np.array([0, 1, 1, 1])) == pytest.approx(0.75)


def test_train_epoch_returns_a_falling_mean_loss() -> None:
    d = load_digits()
    rng = np.random.default_rng(0)
    model = Sequential([Dense(64, 32, rng), ReLU(), Dense(32, 10, rng)])
    first = train_epoch(model, d.X_train, d.y_train, lr=0.1, batch_size=32, rng=rng)
    second = train_epoch(model, d.X_train, d.y_train, lr=0.1, batch_size=32, rng=rng)
    assert isinstance(first, float)
    assert second < first


# --- The payoff -------------------------------------------------------------
def test_your_network_reads_handwriting() -> None:
    d = load_digits()
    rng = np.random.default_rng(0)
    model = Sequential([Dense(64, 32, rng), ReLU(), Dense(32, 10, rng)])
    assert accuracy(model, d.X_val, d.y_val) < 0.3, "untrained, it should be guessing"
    for _ in range(30):
        train_epoch(model, d.X_train, d.y_train, lr=0.1, batch_size=32, rng=rng)
    assert accuracy(model, d.X_val, d.y_val) > 0.93
