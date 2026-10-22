import numpy as np
import pytest

import backprop as lesson
from backprop import Cache, Grads, TwoLayerParams, accuracy, backward, forward, mse, sigmoid, train
from nnlib.data import examples_to_arrays
from nnlib.scalar import OR3, XOR3
from nnlib.types import FloatArray

X, Y = examples_to_arrays(XOR3)
NAMES = ("W1", "b1", "W2", "b2")


def numerical_grads(params: TwoLayerParams, X: FloatArray, Y: FloatArray, h: float = 1e-6) -> dict[str, FloatArray]:
    out: dict[str, FloatArray] = {}
    for name in NAMES:
        array: FloatArray = getattr(params, name)
        grad = np.zeros_like(array)
        for index in np.ndindex(array.shape):
            original = array[index]
            array[index] = original + h
            up = mse(forward(params, X)[0], Y)
            array[index] = original - h
            down = mse(forward(params, X)[0], Y)
            array[index] = original
            grad[index] = (up - down) / (2 * h)
        out[name] = grad
    return out


# --- Task 1 -----------------------------------------------------------------
def test_hand_built_network_computes_xor() -> None:
    assert lesson.HAND_XOR is not None, "set HAND_XOR in backprop.py"
    hand = lesson.HAND_XOR
    assert hand.W1.shape == (3, 3) and hand.b1.shape == (3,)
    assert hand.W2.shape == (3, 1) and hand.b2.shape == (1,)
    H = sigmoid(X @ hand.W1 + hand.b1)
    A = sigmoid(H @ hand.W2 + hand.b2)
    assert np.all(np.abs(A - Y) < 0.1), "every output should be decisively right"


# --- Task 2 -----------------------------------------------------------------
def test_forward_shapes() -> None:
    params = TwoLayerParams.random(3, 4, np.random.default_rng(0))
    A, cache = forward(params, X)
    assert isinstance(cache, Cache)
    assert cache.H.shape == (8, 4), "one hidden output per example per hidden neuron"
    assert A.shape == (8, 1) and cache.A.shape == (8, 1)


def test_forward_known_values() -> None:
    params = TwoLayerParams(
        W1=np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]]),
        b1=np.array([0.0, -1.0]),
        W2=np.array([[1.0], [-1.0]]),
        b2=np.array([0.5]),
    )
    A, cache = forward(params, np.array([[1.0, 1.0, 0.0]]))
    s1 = 1 / (1 + np.exp(-1.0))
    assert cache.H == pytest.approx(np.array([[s1, s1]]))
    assert A == pytest.approx(np.array([[1 / (1 + np.exp(-0.5))]]))


def test_output_layer_reads_the_hidden_layer_not_the_inputs() -> None:
    params = TwoLayerParams(W1=np.zeros((3, 1)), b1=np.zeros(1), W2=np.array([[4.0]]), b2=np.zeros(1))
    A, _ = forward(params, X)
    assert A == pytest.approx(np.full((8, 1), 1 / (1 + np.exp(-2.0)))), "the output should be σ(4 · 0.5)"


# --- Task 3 -----------------------------------------------------------------
def test_backward_shapes_match_the_parameters() -> None:
    params = TwoLayerParams.random(3, 4, np.random.default_rng(0))
    g = backward(params, Y, forward(params, X)[1])
    assert isinstance(g, Grads)
    for name in NAMES:
        assert getattr(g, "d" + name).shape == getattr(params, name).shape, f"d{name} must be shaped like {name}"


@pytest.mark.parametrize("name", ["W2", "b2", "W1", "b1"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_backward_matches_the_wiggle_method(name: str, seed: int) -> None:
    params = TwoLayerParams.random(3, 4, np.random.default_rng(seed))
    exact = backward(params, Y, forward(params, X)[1])
    wiggled = numerical_grads(params, X, Y)
    hint = "" if name in ("W2", "b2") else " (output layer right, hidden wrong? check dH = dZ2 @ W2.T and H(1 − H))"
    assert getattr(exact, "d" + name) == pytest.approx(wiggled[name], abs=1e-8), f"d{name} is wrong{hint}"


def test_backward_with_one_hidden_neuron_and_one_example_by_hand() -> None:
    # THEORY §5's worked example.
    params = TwoLayerParams(W1=np.array([[1.0]]), b1=np.zeros(1), W2=np.array([[2.0]]), b2=np.zeros(1))
    Xs, Ys = np.array([[1.0]]), np.array([[1.0]])
    g = backward(params, Ys, forward(params, Xs)[1])
    assert g.dW2 == pytest.approx(np.array([[-0.0420]]), abs=1e-4)
    assert g.dW1 == pytest.approx(np.array([[-0.0226]]), abs=1e-4)


# --- Task 4 -----------------------------------------------------------------
def test_train_records_history_and_improves() -> None:
    Xo, Yo = examples_to_arrays(OR3)
    params = TwoLayerParams.random(3, 4, np.random.default_rng(0))
    start = mse(forward(params, Xo)[0], Yo)
    history = train(params, Xo, Yo, lr=2.0, epochs=50)
    assert len(history) == 50
    assert history[0] == pytest.approx(start)
    assert history[-1] < history[0]


# --- The payoff -------------------------------------------------------------
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_hidden_layer_learns_xor(seed: int) -> None:
    """The problem a single neuron provably cannot solve (lesson 2 §6)."""
    params = TwoLayerParams.random(3, 4, np.random.default_rng(seed))
    history = train(params, X, Y, lr=2.0, epochs=5000)
    assert accuracy(params, X, Y) == 1.0
    assert history[-1] < 0.02
