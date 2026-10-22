import copy

import numpy as np
import pytest

from nnlib.scalar import truth_table

from descent import (
    AND3,
    FIRST_INPUT,
    OR3,
    XOR3,
    Gradients,
    Neuron,
    accuracy,
    gradients,
    loss,
    numerical_gradients,
    squared_error,
    step,
    train,
    train_by_wiggling,
)


def random_neuron(seed: int) -> Neuron:
    rng = np.random.default_rng(seed)
    return Neuron(w=[float(rng.uniform(-1, 1)) for _ in range(3)], b=float(rng.uniform(-1, 1)))


# --- Task 1 -----------------------------------------------------------------
def test_squared_error() -> None:
    assert squared_error(1.0, 1.0) == 0.0
    assert squared_error(0.5, 1.0) == pytest.approx(0.25)
    assert squared_error(0.2, 0.5) == pytest.approx(squared_error(0.8, 0.5)), "both directions count"


def test_loss_of_a_clueless_neuron_is_a_quarter() -> None:
    # Every output is 0.5, so every squared error is 0.25 (THEORY §1).
    assert loss(Neuron(w=[0.0, 0.0, 0.0]), OR3) == pytest.approx(0.25)


def test_loss_is_a_mean_not_a_sum() -> None:
    always_one = Neuron(w=[0.0, 0.0, 0.0], b=50.0)
    assert loss(always_one, OR3) == pytest.approx(1 / 8), "wrong on 1 row of 8: divide by n"


# --- Task 2 -----------------------------------------------------------------
def test_gradients_on_the_theory_worked_example() -> None:
    # THEORY §4: a zero neuron on OR3.
    g = gradients(Neuron(w=[0.0, 0.0, 0.0]), OR3)
    assert isinstance(g, Gradients)
    assert g.dw == pytest.approx([-0.125, -0.125, -0.125])
    assert g.db == pytest.approx(-0.1875)


def test_gradients_have_one_slope_per_weight() -> None:
    assert len(gradients(random_neuron(0), OR3).dw) == 3


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("name", ["OR3", "AND3", "XOR3"])
def test_gradients_match_the_wiggle_method(seed: int, name: str) -> None:
    data = {"OR3": OR3, "AND3": AND3, "XOR3": XOR3}[name]
    n = random_neuron(seed)
    exact, wiggled = gradients(n, data), numerical_gradients(n, data)
    assert exact.dw == pytest.approx(wiggled.dw, abs=1e-7)
    assert exact.db == pytest.approx(wiggled.db, abs=1e-7), "the bias slope is δ alone: ∂z/∂b = 1"


def test_gradients_do_not_change_the_neuron() -> None:
    n = random_neuron(0)
    before = copy.deepcopy(n)
    gradients(n, OR3)
    assert n == before


def test_an_input_that_is_always_zero_gets_no_blame() -> None:
    data = [e for e in OR3 if e.x[0] == 0.0]  # first input is 0 on every row
    assert gradients(random_neuron(0), data).dw[0] == 0.0


# --- Task 3 -----------------------------------------------------------------
def test_step_moves_against_the_slope() -> None:
    n = Neuron(w=[1.0, 1.0, 1.0], b=1.0)
    step(n, Gradients(dw=[0.5, -0.5, 0.0], db=2.0), lr=0.1)
    assert n.w == pytest.approx([0.95, 1.05, 1.0])
    assert n.b == pytest.approx(0.8)


def test_one_small_step_lowers_the_loss() -> None:
    n = random_neuron(0)
    before = loss(n, OR3)
    step(n, gradients(n, OR3), lr=0.1)
    assert loss(n, OR3) < before


# --- Task 4 -----------------------------------------------------------------
def test_train_history_starts_untrained_and_falls() -> None:
    n = random_neuron(0)
    start = loss(n, OR3)
    history = train(n, OR3, lr=1.0, epochs=25)
    assert len(history) == 25
    assert history[0] == pytest.approx(start), "record the loss BEFORE each step"
    assert history[-1] < history[0]


@pytest.mark.parametrize("name", ["OR3", "AND3", "FIRST_INPUT"])
def test_one_neuron_learns_the_easy_functions(name: str) -> None:
    data = {"OR3": OR3, "AND3": AND3, "FIRST_INPUT": FIRST_INPUT}[name]
    n = random_neuron(0)
    train(n, data, lr=1.0, epochs=3000)
    assert accuracy(n, data) == 1.0


def test_gradient_descent_does_far_less_work_than_wiggling() -> None:
    """Both learn "copy the first of 10 inputs". Count the passes over the data each needs."""
    data = truth_table(lambda x: x[0] == 1, d_in=10)
    wiggled = train_by_wiggling(Neuron(w=[0.0] * 10), data, rounds=60)
    history = train(Neuron(w=[0.0] * 10), data, lr=2.0, epochs=100)
    assert wiggled.history[-1] < 0.01 and history[-1] < 0.01, "both should have learned it"
    gradient_passes = 2 * len(history)  # one pass to record the loss, one for the gradients
    assert wiggled.loss_evaluations > 3 * gradient_passes


def test_wiggling_costs_grow_with_the_number_of_knobs() -> None:
    small = train_by_wiggling(Neuron(w=[0.0] * 3), truth_table(lambda x: x[0] == 1, 3), rounds=10)
    large = train_by_wiggling(Neuron(w=[0.0] * 10), truth_table(lambda x: x[0] == 1, 10), rounds=10)
    assert large.loss_evaluations > 2.5 * small.loss_evaluations


# --- The wall -----------------------------------------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_one_neuron_cannot_learn_xor(seed: int) -> None:
    """Passes when your code is RIGHT. The failure is the neuron's, not yours (THEORY §6)."""
    n = random_neuron(seed)
    history = train(n, XOR3, lr=1.0, epochs=3000)
    assert accuracy(n, XOR3) < 1.0
    assert history[-1] > 0.2
