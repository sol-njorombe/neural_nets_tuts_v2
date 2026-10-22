import dataclasses

import numpy as np
import pytest

import neuron as lesson
from neuron import AND3, FIRST_INPUT, OR3, XOR3, Example, Neuron, accuracy, sigmoid


# --- GIVEN data -------------------------------------------------------------
def test_datasets_have_eight_rows_of_three_inputs() -> None:
    for data in (FIRST_INPUT, OR3, AND3, XOR3):
        assert len(data) == 8
        assert all(isinstance(e, Example) and len(e.x) == 3 for e in data)


# --- Task 1 -----------------------------------------------------------------
def test_sigmoid_known_values() -> None:
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(2.0) == pytest.approx(0.8807970779778823)
    assert sigmoid(-2.0) == pytest.approx(1 - 0.8807970779778823)


def test_sigmoid_survives_huge_inputs() -> None:
    assert sigmoid(1000.0) == pytest.approx(1.0)
    assert sigmoid(-1000.0) == pytest.approx(0.0), "for negative z, use e^z / (1 + e^z)"


# --- Task 2 -----------------------------------------------------------------
def test_random_neuron_has_one_weight_per_input_and_zero_bias() -> None:
    n = Neuron.random(3, np.random.default_rng(0))
    assert len(n.w) == 3
    assert n.b == 0.0


def test_random_weights_are_plain_floats_between_minus_one_and_one() -> None:
    n = Neuron.random(50, np.random.default_rng(1))
    assert all(type(w) is float for w in n.w), "wrap each draw in float(...)"
    assert all(-1 <= w <= 1 for w in n.w)
    assert len(set(n.w)) == 50
    assert min(n.w) < 0 < max(n.w)


def test_random_neuron_is_reproducible_with_the_same_seed() -> None:
    a = Neuron.random(3, np.random.default_rng(42))
    b = Neuron.random(3, np.random.default_rng(42))
    c = Neuron.random(3, np.random.default_rng(43))
    assert a == b, "draw every weight from the rng you were given"
    assert a != c


# --- Task 3 -----------------------------------------------------------------
def test_weighted_sum() -> None:
    assert Neuron(w=[1.0, 2.0, 3.0]).weighted_sum((1.0, 0.0, 1.0)) == pytest.approx(4.0)
    assert Neuron(w=[0.5, -0.5], b=1.0).weighted_sum((2.0, 2.0)) == pytest.approx(1.0)


def test_weighted_sum_includes_the_bias() -> None:
    assert Neuron(w=[0.0, 0.0, 0.0], b=-2.5).weighted_sum((1.0, 1.0, 1.0)) == pytest.approx(-2.5)


def test_weighted_sum_rejects_the_wrong_number_of_inputs() -> None:
    with pytest.raises(ValueError):
        Neuron(w=[1.0, 1.0, 1.0]).weighted_sum((1.0, 1.0))
    with pytest.raises(ValueError):
        Neuron(w=[1.0, 1.0]).weighted_sum((1.0, 1.0, 1.0))


def test_forward_on_the_theory_example() -> None:
    # THEORY §2: w = [1, -2, 0.5], b = 0.5, x = (1, 0, 1)  ->  z = 2, a = 0.8808
    n = Neuron(w=[1.0, -2.0, 0.5], b=0.5)
    assert n.weighted_sum((1.0, 0.0, 1.0)) == pytest.approx(2.0)
    assert n.forward((1.0, 0.0, 1.0)) == pytest.approx(0.8807970779778823)


def test_forward_with_zero_weights_is_one_half() -> None:
    assert Neuron(w=[0.0, 0.0, 0.0]).forward((1.0, 0.0, 1.0)) == pytest.approx(0.5)


def test_predict_thresholds_at_one_half() -> None:
    assert Neuron(w=[5.0, 0.0, 0.0]).predict((1.0, 0.0, 0.0)) == 1
    assert Neuron(w=[-5.0, 0.0, 0.0]).predict((1.0, 0.0, 0.0)) == 0


def test_forward_does_not_change_the_neuron() -> None:
    n = Neuron(w=[1.0, 2.0, 3.0], b=0.5)
    before = dataclasses.replace(n, w=list(n.w))
    n.forward((1.0, 1.0, 1.0))
    assert n == before


# --- Task 4 -----------------------------------------------------------------
def test_accuracy_counts_correct_rows() -> None:
    always_yes = Neuron(w=[0.0, 0.0, 0.0], b=10.0)
    assert accuracy(always_yes, OR3) == pytest.approx(7 / 8)
    assert accuracy(always_yes, AND3) == pytest.approx(1 / 8)
    assert accuracy(always_yes, XOR3) == pytest.approx(0.5)


def test_a_neuron_that_copies_its_first_input() -> None:
    assert accuracy(Neuron(w=[10.0, 0.0, 0.0], b=-5.0), FIRST_INPUT) == 1.0


# --- Task 5: the payoff -----------------------------------------------------
def test_your_hand_picked_neuron_computes_or() -> None:
    assert lesson.HAND_OR is not None, "set HAND_OR in neuron.py"
    assert accuracy(lesson.HAND_OR, OR3) == 1.0


def test_your_hand_picked_neuron_computes_and() -> None:
    assert lesson.HAND_AND is not None, "set HAND_AND in neuron.py"
    assert accuracy(lesson.HAND_AND, AND3) == 1.0


def test_your_neurons_are_confident() -> None:
    """Right answers should be decisive, not 0.51. Large weights make sigmoid commit."""
    assert lesson.HAND_OR is not None and lesson.HAND_AND is not None
    for hand, data in ((lesson.HAND_OR, OR3), (lesson.HAND_AND, AND3)):
        for example in data:
            assert abs(hand.forward(example.x) - example.y) < 0.1, f"unsure on {example.x}"
