import math
from collections.abc import Callable

import numpy as np
import pytest

from generalisation import (
    SGD,
    Adam,
    EarlyStopping,
    History,
    Momentum,
    Optimiser,
    Split,
    diagnose,
    digit_model,
    evaluate_loss,
    fit,
    he_normal,
    naive_uniform,
    signal_scales,
    tiny_normal,
    train_val_split,
)
from nnlib.data import load_digits
from nnlib.layers import ParamPair
from nnlib.types import FloatArray

DIGITS = load_digits()
FULL = Split(DIGITS.X_train, DIGITS.y_train, DIGITS.X_val, DIGITS.y_val)
SMALL = Split(DIGITS.X_train[:60], DIGITS.y_train[:60], DIGITS.X_val, DIGITS.y_val)


# --- Task 1 -----------------------------------------------------------------
def test_split_sizes() -> None:
    X, y = np.arange(200.0).reshape(100, 2), np.arange(100)
    s = train_val_split(X, y, 0.2, np.random.default_rng(0))
    assert s.X_val.shape == (20, 2) and s.y_val.shape == (20,)
    assert s.X_train.shape == (80, 2) and s.y_train.shape == (80,)


def test_split_shuffles_keeps_pairs_and_never_overlaps() -> None:
    X, y = np.arange(100.0).reshape(100, 1), np.arange(100)
    s = train_val_split(X, y, 0.3, np.random.default_rng(0))
    assert np.array_equal(s.X_train[:, 0], s.y_train) and np.array_equal(s.X_val[:, 0], s.y_val)
    assert set(s.y_train.tolist()).isdisjoint(s.y_val.tolist()), "a row in both sets leaks answers"
    assert sorted(s.y_train.tolist() + s.y_val.tolist()) == list(range(100))
    assert s.y_val.tolist() != list(range(30)), "shuffle before splitting"


# --- Task 2 -----------------------------------------------------------------
def test_he_normal_shape_and_spread() -> None:
    W = he_normal(200, 300, np.random.default_rng(0))
    assert W.shape == (200, 300)
    assert W.std() == pytest.approx(math.sqrt(2 / 200), rel=0.02)
    assert abs(W.mean()) < 0.005


def test_he_initialisation_keeps_the_signal_steady_through_20_layers() -> None:
    scales = signal_scales(he_normal, 20, 100, np.random.default_rng(0))
    assert all(0.3 < s < 3.0 for s in scales), scales


def test_the_other_initialisations_explode_or_vanish() -> None:
    assert signal_scales(naive_uniform, 20, 100, np.random.default_rng(0))[-1] > 1e6
    assert signal_scales(tiny_normal, 20, 100, np.random.default_rng(0))[-1] < 1e-6


# --- Task 3 -----------------------------------------------------------------
def one_param(value: float, grad: float) -> tuple[FloatArray, FloatArray]:
    return np.array([value]), np.array([grad])


def test_sgd_step() -> None:
    p, g = one_param(1.0, 2.0)
    SGD([(p, g)], lr=0.1).step()
    assert p == pytest.approx([0.8])


def test_momentum_accumulates_velocity() -> None:
    p, g = one_param(0.0, 1.0)
    opt = Momentum([(p, g)], lr=0.1, beta=0.9)
    opt.step()  # v = 1      p = -0.1
    opt.step()  # v = 1.9    p = -0.29
    assert p == pytest.approx([-0.29])
    assert opt.velocities[0] == pytest.approx([1.9])


def test_adam_first_step_is_lr_whatever_the_gradient_scale() -> None:
    """After bias correction, step one moves every parameter by about lr, in the sign of -g."""
    p = np.array([0.0, 0.0, 0.0])
    g = np.array([1000.0, 0.001, -5.0])
    Adam([(p, g)], lr=0.01).step()
    assert p == pytest.approx([-0.01, -0.01, 0.01], rel=1e-3)


def test_adam_second_step_known_values() -> None:
    p, g = one_param(0.0, 1.0)
    opt = Adam([(p, g)], lr=0.1)
    opt.step()
    g[...] = 3.0
    opt.step()
    m_hat = (0.9 * 0.1 + 0.1 * 3.0) / (1 - 0.9**2)
    v_hat = (0.999 * 0.001 + 0.001 * 9.0) / (1 - 0.999**2)
    assert opt.t == 2
    assert p == pytest.approx([-0.1 - 0.1 * m_hat / (math.sqrt(v_hat) + 1e-8)])


def test_optimisers_update_in_place() -> None:
    for make in (lambda pairs: SGD(pairs, 0.1), lambda pairs: Momentum(pairs, 0.1), lambda pairs: Adam(pairs, 0.1)):
        p, g = one_param(1.0, 1.0)
        original = p
        make([(p, g)]).step()
        assert original is p and p[0] != 1.0


def test_momentum_and_adam_beat_sgd_in_a_long_narrow_valley() -> None:
    """f(p) = ½(100·p₀² + p₁²). SGD's step size is capped by the steep direction (THEORY §5)."""

    def run(make: Callable[[list[ParamPair]], Optimiser]) -> float:
        p, g = np.array([1.0, 1.0]), np.zeros(2)
        opt = make([(p, g)])
        for _ in range(100):
            g[...] = [100 * p[0], p[1]]
            opt.step()
        return 0.5 * (100 * p[0] ** 2 + p[1] ** 2)

    sgd = run(lambda pairs: SGD(pairs, lr=0.015))
    momentum = run(lambda pairs: Momentum(pairs, lr=0.015))
    adam = run(lambda pairs: Adam(pairs, lr=0.1))
    assert momentum < sgd / 10
    assert adam < sgd / 10


# --- Task 4 -----------------------------------------------------------------
def test_early_stopping_tracks_the_best_epoch_and_stops_after_patience() -> None:
    model = digit_model(np.random.default_rng(0))
    es = EarlyStopping(patience=2)
    assert es.update(0, 1.0, model) is False
    assert es.update(1, 0.8, model) is False
    assert es.update(2, 0.9, model) is False
    assert es.update(3, 0.85, model) is True, "two epochs without beating 0.8"
    assert es.best_epoch == 1 and es.best_loss == 0.8


def test_early_stopping_saves_copies_and_restores_them() -> None:
    model = digit_model(np.random.default_rng(0))
    es = EarlyStopping(patience=5)
    es.update(0, 1.0, model)
    first_W = model.params()[0][0]
    saved = first_W.copy()
    first_W += 5.0  # training moves on...
    assert not np.array_equal(es.best_params[0], first_W), "save p.copy(), not p itself"
    es.restore(model)
    assert np.array_equal(model.params()[0][0], saved)
    assert model.params()[0][0] is first_W, "restore in place: p[...] = saved"


# --- Task 5 -----------------------------------------------------------------
def test_fit_records_every_metric_every_epoch() -> None:
    rng = np.random.default_rng(0)
    model = digit_model(rng)
    h = fit(model, SGD(model.params(), 0.1), FULL, epochs=3, batch_size=32, rng=rng)
    assert len(h.train_loss) == len(h.val_loss) == len(h.train_acc) == len(h.val_acc) == 3
    assert h.val_loss[-1] == pytest.approx(evaluate_loss(model, FULL.X_val, FULL.y_val))
    assert h.train_loss[-1] < h.train_loss[0]


def test_fit_with_adam_reads_digits() -> None:
    rng = np.random.default_rng(0)
    model = digit_model(rng)
    h = fit(model, Adam(model.params(), lr=0.005), FULL, epochs=10, batch_size=32, rng=rng)
    assert h.val_acc[-1] > 0.95


def test_fit_with_early_stopping_stops_and_keeps_the_best_weights() -> None:
    rng = np.random.default_rng(0)
    model = digit_model(rng, hidden=256)
    es = EarlyStopping(patience=10)
    h = fit(model, SGD(model.params(), 0.1), SMALL, epochs=200, batch_size=16, rng=rng, early_stopping=es)
    assert len(h.val_loss) < 200, "it should have stopped early"
    assert evaluate_loss(model, SMALL.X_val, SMALL.y_val) == pytest.approx(min(h.val_loss)), "restore the best weights"


# --- Task 6 -----------------------------------------------------------------
def make_history(train_loss: list[float], train_acc: float, val_acc: float) -> History:
    return History(train_loss=train_loss, val_loss=[1.0] * len(train_loss), train_acc=[train_acc], val_acc=[val_acc])


def test_diagnose_rules() -> None:
    assert diagnose(make_history([2.3, math.nan], 0.1, 0.1)) == "diverging"
    assert diagnose(make_history([2.3, 2.5], 0.1, 0.1)) == "diverging"
    assert diagnose(make_history([2.3, 2.0], 0.5, 0.5)) == "underfitting"
    assert diagnose(make_history([2.3, 0.01], 1.0, 0.8)) == "overfitting"
    assert diagnose(make_history([2.3, 0.1], 0.98, 0.96)) == "healthy"


# --- The payoff: diagnose real training runs ---------------------------------
@pytest.mark.parametrize(
    "lr,split,hidden,expected",
    [
        (5.0, "full", 32, "diverging"),
        (1e-4, "full", 32, "underfitting"),
        (0.1, "small", 256, "overfitting"),
        (0.1, "full", 32, "healthy"),
    ],
)
def test_diagnose_real_training_runs(lr: float, split: str, hidden: int, expected: str) -> None:
    data = FULL if split == "full" else SMALL
    rng = np.random.default_rng(0)
    model = digit_model(rng, hidden=hidden)
    epochs = 10 if split == "full" else 100
    h = fit(model, SGD(model.params(), lr), data, epochs=epochs, batch_size=16 if split == "small" else 32, rng=rng)
    assert diagnose(h) == expected
