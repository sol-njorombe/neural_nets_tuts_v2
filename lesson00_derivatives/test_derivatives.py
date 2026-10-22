import math

import pytest

import derivatives as lesson
from derivatives import (
    Trace,
    chain_rule,
    d_sigmoid,
    d_square,
    numerical_derivative,
    numerical_gradient,
    partial_derivative,
    sigmoid,
    sum_over_paths,
    tiny_backward,
    tiny_forward,
)


# --- Task 1 -----------------------------------------------------------------
def test_numerical_derivative_of_a_line_is_its_slope() -> None:
    assert numerical_derivative(lambda x: 3 * x + 1, 5.0) == pytest.approx(3.0)


def test_numerical_derivative_of_x_squared() -> None:
    assert numerical_derivative(lambda x: x * x, 3.0) == pytest.approx(6.0)


def test_numerical_derivative_of_sine_is_cosine() -> None:
    assert numerical_derivative(math.sin, 0.7) == pytest.approx(math.cos(0.7))


def test_numerical_derivative_uses_the_central_difference() -> None:
    # With a big step at x = 1, the one-sided estimate of x² gives 2.5; the central one gives 2.
    result = numerical_derivative(lambda x: x * x, 1.0, h=0.5)
    assert result == pytest.approx(2.0), "look both ways: f(x + h) - f(x - h), divided by 2h"


# --- Task 2 -----------------------------------------------------------------
@pytest.mark.parametrize("x", [-2.0, 0.0, 0.5, 3.0])
def test_d_square_matches_the_wiggle_method(x: float) -> None:
    assert d_square(x) == pytest.approx(numerical_derivative(lambda v: v * v, x), abs=1e-6)


@pytest.mark.parametrize("z", [-3.0, -0.5, 0.0, 1.0, 4.0])
def test_d_sigmoid_matches_the_wiggle_method(z: float) -> None:
    assert d_sigmoid(z) == pytest.approx(numerical_derivative(sigmoid, z), abs=1e-6)


def test_d_sigmoid_is_steepest_in_the_middle_and_flat_at_the_edges() -> None:
    assert d_sigmoid(0.0) == pytest.approx(0.25)
    assert d_sigmoid(6.0) < 0.01


# --- Task 3 -----------------------------------------------------------------
def test_chain_rule_square_of_a_line() -> None:
    # y = (3x + 1)², at x = 2: outer slope 2·7 = 14, inner slope 3 -> 42 (THEORY §5)
    result = chain_rule(d_square, lambda x: 3 * x + 1, lambda x: 3.0, 2.0)
    assert result == pytest.approx(42.0), "evaluate d_outer at inner(x), not at x"


def test_chain_rule_sigmoid_of_a_square_matches_the_wiggle_method() -> None:
    composed = lambda x: sigmoid(x * x)  # noqa: E731
    expected = numerical_derivative(composed, 0.8)
    assert chain_rule(d_sigmoid, lambda x: x * x, d_square, 0.8) == pytest.approx(expected, abs=1e-6)


# --- Task 4 -----------------------------------------------------------------
def test_partial_derivatives_of_x_squared_y_plus_3y() -> None:
    # The worked example in THEORY §4: at (2, 5), ∂f/∂x = 2xy = 20 and ∂f/∂y = x² + 3 = 7.
    f = lambda v: v[0] ** 2 * v[1] + 3 * v[1]  # noqa: E731
    assert partial_derivative(f, [2.0, 5.0], 0) == pytest.approx(20.0)
    assert partial_derivative(f, [2.0, 5.0], 1) == pytest.approx(7.0)


def test_partial_derivative_nudges_only_one_input() -> None:
    f = lambda v: v[0] * v[1] + v[2]  # noqa: E731
    args = [2.0, 5.0, 7.0]
    assert partial_derivative(f, args, 2) == pytest.approx(1.0)


def test_partial_derivative_leaves_the_inputs_unchanged() -> None:
    args = [2.0, 5.0, 7.0]
    partial_derivative(lambda v: v[0] * v[1], args, 0)
    assert args == [2.0, 5.0, 7.0], "copy args before nudging: list(args)"


def test_partial_derivative_accepts_a_tuple() -> None:
    assert partial_derivative(lambda v: v[0] * v[1], (3.0, 4.0), 1) == pytest.approx(3.0)


def test_numerical_gradient_is_every_partial_in_order() -> None:
    grad = numerical_gradient(lambda v: v[0] ** 2 + 3 * v[1], [4.0, 1.0])
    assert isinstance(grad, list)
    assert len(grad) == 2, "one partial derivative per input"
    assert grad == pytest.approx([8.0, 3.0])


# --- Task 5 -----------------------------------------------------------------
def test_sum_over_paths_multiplies_along_and_adds_across() -> None:
    assert sum_over_paths([[2.0, 3.0], [4.0]]) == pytest.approx(10.0)


def test_a_single_path_is_just_the_chain_rule() -> None:
    assert sum_over_paths([[14.0, 3.0]]) == pytest.approx(42.0)


def test_sum_over_paths_on_the_theory_example() -> None:
    # z = x·y, x = t², y = 3t  (THEORY §6). At t = 2: x = 4, y = 6.
    t = 2.0
    x, y = t * t, 3 * t
    through_x = [y, 2 * t]  # ∂z/∂x = y,  dx/dt = 2t
    through_y = [x, 3.0]  # ∂z/∂y = x,  dy/dt = 3
    result = sum_over_paths([through_x, through_y])
    # Substituting directly: z = 3t³, so dz/dt = 9t² = 36.
    assert result == pytest.approx(36.0)
    assert result == pytest.approx(numerical_derivative(lambda s: (s * s) * (3 * s), t), abs=1e-5)


def test_sum_over_paths_with_no_paths_is_zero() -> None:
    assert sum_over_paths([]) == 0.0


# --- Task 6 -----------------------------------------------------------------
def test_tiny_forward_records_every_step() -> None:
    trace = tiny_forward(w=0.5, x=2.0, y=1.0)
    assert isinstance(trace, Trace)
    # The worked example in THEORY §7
    assert trace.z == pytest.approx(1.0)
    assert trace.a == pytest.approx(0.7310586)
    assert trace.L == pytest.approx(0.0723295)


def test_tiny_backward_on_the_theory_example() -> None:
    trace = tiny_forward(w=0.5, x=2.0, y=1.0)
    assert tiny_backward(trace, x=2.0, y=1.0) == pytest.approx(-0.2115084)


@pytest.mark.parametrize("w,x,y", [(0.5, 1.0, 1.0), (-1.2, 2.0, 0.0), (2.0, -0.7, 1.0), (0.0, 1.0, 0.0)])
def test_tiny_backward_matches_the_wiggle_method(w: float, x: float, y: float) -> None:
    exact = tiny_backward(tiny_forward(w, x, y), x, y)
    wiggled = numerical_derivative(lambda v: tiny_forward(v, x, y).L, w)
    assert exact == pytest.approx(wiggled, abs=1e-6)


def test_tiny_backward_uses_the_trace_not_a_fresh_forward_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    trace = tiny_forward(0.5, 2.0, 1.0)

    def forbidden(z: float) -> float:
        raise AssertionError("tiny_backward should read trace.a instead of calling sigmoid again")

    monkeypatch.setattr(lesson, "sigmoid", forbidden)
    tiny_backward(trace, 2.0, 1.0)


# --- The payoff -------------------------------------------------------------
def test_one_step_against_the_slope_lowers_the_loss() -> None:
    """Gradient descent in one line. Lesson 2 is this, repeated."""
    w, x, y = 0.5, 2.0, 1.0
    before = tiny_forward(w, x, y)
    new_w = w - 0.5 * tiny_backward(before, x, y)
    assert new_w > w, "the slope was negative, so stepping against it raises w"
    assert tiny_forward(new_w, x, y).L < before.L
