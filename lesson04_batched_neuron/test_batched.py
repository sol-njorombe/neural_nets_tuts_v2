import time

import numpy as np
import pytest

from batched import Cache, Grads, NeuronParams, accuracy, backward, forward, mse, sigmoid, train
from nnlib.data import examples_to_arrays
from nnlib.scalar import AND3, OR3, XOR3, Neuron, gradients
from nnlib.types import FloatArray

X_OR, Y_OR = examples_to_arrays(OR3)
X_XOR, Y_XOR = examples_to_arrays(XOR3)


def params_from_seed(seed: int, d_in: int = 3) -> NeuronParams:
    rng = np.random.default_rng(seed)
    return NeuronParams(W=rng.uniform(-1, 1, (d_in, 1)), b=rng.uniform(-1, 1, 1))


def numerical_grads(params: NeuronParams, X: FloatArray, Y: FloatArray, h: float = 1e-6) -> Grads:
    out: dict[str, FloatArray] = {}
    for name in ("W", "b"):
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
    return Grads(dW=out["W"], db=out["b"])


# --- Task 1 -----------------------------------------------------------------
def test_forward_shapes() -> None:
    A, cache = forward(params_from_seed(0), X_OR)
    assert A.shape == (8, 1), "one output per example, as a column"
    assert isinstance(cache, Cache)
    assert cache.A.shape == (8, 1) and cache.X.shape == (8, 3)


def test_forward_matches_the_lesson_1_neuron_row_by_row() -> None:
    params = params_from_seed(0)
    neuron = Neuron(w=list(params.W[:, 0]), b=float(params.b[0]))
    A, _ = forward(params, X_OR)
    for k, e in enumerate(OR3):
        assert A[k, 0] == pytest.approx(neuron.forward(e.x))


def test_mse_returns_a_float_mean() -> None:
    value = mse(np.array([[0.5], [1.0]]), np.array([[1.0], [1.0]]))
    assert isinstance(value, float)
    assert value == pytest.approx(0.125)


# --- Task 2 -----------------------------------------------------------------
def test_backward_shapes_match_the_parameters() -> None:
    params = params_from_seed(0)
    _, cache = forward(params, X_OR)
    grads = backward(params, Y_OR, cache)
    assert grads.dW.shape == params.W.shape == (3, 1)
    assert grads.db.shape == params.b.shape == (1,), "db = dZ.sum(axis=0)"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_backward_matches_the_wiggle_method(seed: int) -> None:
    params = params_from_seed(seed)
    _, cache = forward(params, X_XOR)
    exact = backward(params, Y_XOR, cache)
    wiggled = numerical_grads(params, X_XOR, Y_XOR)
    assert exact.dW == pytest.approx(wiggled.dW, abs=1e-8)
    assert exact.db == pytest.approx(wiggled.db, abs=1e-8)


def test_backward_equals_your_lesson_2_loop_version() -> None:
    params = params_from_seed(0)
    _, cache = forward(params, X_OR)
    grads = backward(params, Y_OR, cache)
    looped = gradients(Neuron(w=list(params.W[:, 0]), b=float(params.b[0])), OR3)
    assert grads.dW[:, 0] == pytest.approx(looped.dw)
    assert grads.db[0] == pytest.approx(looped.db)


def test_the_theory_worked_example() -> None:
    # THEORY §3: zero neuron on OR3 -> dW = [-0.125]*3, db = [-0.1875]
    params = NeuronParams(W=np.zeros((3, 1)), b=np.zeros(1))
    _, cache = forward(params, X_OR)
    grads = backward(params, Y_OR, cache)
    assert grads.dW[:, 0] == pytest.approx([-0.125, -0.125, -0.125])
    assert grads.db == pytest.approx([-0.1875])


# --- Task 3 -----------------------------------------------------------------
def test_train_history_and_learning() -> None:
    params = params_from_seed(0)
    start = mse(forward(params, X_OR)[0], Y_OR)
    history = train(params, X_OR, Y_OR, lr=1.0, epochs=2000)
    assert len(history) == 2000
    assert history[0] == pytest.approx(start)
    assert accuracy(params, X_OR, Y_OR) == 1.0


def test_learns_and() -> None:
    X, Y = examples_to_arrays(AND3)
    params = params_from_seed(0)
    train(params, X, Y, lr=1.0, epochs=3000)
    assert accuracy(params, X, Y) == 1.0


def test_still_cannot_learn_xor() -> None:
    """Same wall as lesson 2: matrices made it faster, not more capable."""
    params = params_from_seed(0)
    history = train(params, X_XOR, Y_XOR, lr=1.0, epochs=3000)
    assert history[-1] > 0.2


# --- The payoff -------------------------------------------------------------
def test_batched_training_is_fast_on_a_real_sized_problem() -> None:
    """10,000 examples, 100 inputs. In lesson 2 one epoch is a million Python steps."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10_000, 100))
    true_w = rng.normal(size=(100, 1))
    Y = (X @ true_w > 0).astype(np.float64)  # a rule a single neuron can learn
    params = NeuronParams(W=np.zeros((100, 1)), b=np.zeros(1))
    start = time.perf_counter()
    train(params, X, Y, lr=2.0, epochs=200)
    elapsed = time.perf_counter() - start
    assert accuracy(params, X, Y) > 0.95
    assert elapsed < 5.0, "too slow: is there a Python loop over examples?"


def test_sigmoid_given() -> None:
    assert sigmoid(np.zeros((2, 1))) == pytest.approx(np.full((2, 1), 0.5))
