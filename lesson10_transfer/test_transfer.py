import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="install PyTorch for lesson 10: see lesson 9's README")
from torch import nn  # noqa: E402

from transfer import (  # noqa: E402
    N_FEATURES,
    TransferModel,
    extract_features,
    few_shot_split,
    fine_tune_optimiser,
    fit,
    freeze,
    make_backbone,
    parity_experiment,
    trainable_parameters,
)


# --- Task 1 -----------------------------------------------------------------
def test_few_shot_split_takes_exactly_per_class_of_each_class() -> None:
    X = np.arange(60.0 * 64).reshape(60, 64)
    labels = np.repeat(np.arange(3), 20)
    s = few_shot_split(X, labels, per_class=4, rng=np.random.default_rng(0))
    assert tuple(s.X_train.shape) == (12, 64) and tuple(s.X_test.shape) == (48, 64)
    assert sorted(s.y_train.tolist()) == [0] * 4 + [1] * 4 + [2] * 4
    assert s.X_train.dtype == torch.float32 and s.y_train.dtype == torch.int64


def test_few_shot_split_never_reuses_an_image_and_keeps_labels_aligned() -> None:
    X = np.repeat(np.arange(40.0).reshape(40, 1), 64, axis=1)  # row k is all k
    labels = np.arange(40) % 2
    s = few_shot_split(X, labels, per_class=5, rng=np.random.default_rng(1))
    train_rows = set(s.X_train[:, 0].int().tolist())
    test_rows = set(s.X_test[:, 0].int().tolist())
    assert train_rows.isdisjoint(test_rows) and len(train_rows | test_rows) == 40
    assert all(int(x) % 2 == int(y) for x, y in zip(s.X_train[:, 0], s.y_train))


def test_few_shot_split_is_random() -> None:
    labels = np.repeat(np.arange(2), 50)
    X = np.arange(100.0).repeat(64).reshape(100, 64)
    a = few_shot_split(X, labels, 5, np.random.default_rng(0)).X_train[:, 0].tolist()
    assert a != [0, 1, 2, 3, 4, 50, 51, 52, 53, 54], "use rng.choice, not the first few"


# --- Task 2 -----------------------------------------------------------------
def test_freeze_stops_every_parameter_learning() -> None:
    backbone = make_backbone()
    freeze(backbone)
    assert all(not p.requires_grad for p in backbone.parameters())
    assert trainable_parameters(backbone) == []


def test_trainable_parameters_after_freezing_only_the_backbone() -> None:
    model = nn.Sequential(make_backbone(), nn.Linear(N_FEATURES, 2))
    freeze(model[0])
    trainable = trainable_parameters(model)
    assert len(trainable) == 2, "only the head's weight and bias"
    assert sum(p.numel() for p in trainable) == N_FEATURES * 2 + 2


# --- Task 3 -----------------------------------------------------------------
def test_transfer_model_shapes() -> None:
    model = TransferModel(make_backbone(), N_FEATURES, 2)
    assert isinstance(model.head, nn.Linear)
    assert tuple(model.head.weight.shape) == (2, N_FEATURES)
    assert tuple(model(torch.zeros(7, 64)).shape) == (7, 2)


def test_training_with_a_frozen_backbone_leaves_it_unchanged() -> None:
    torch.manual_seed(0)
    model = TransferModel(make_backbone(), N_FEATURES, 2)
    freeze(model.backbone)
    before = [p.detach().clone() for p in model.backbone.parameters()]
    head_before = model.head.weight.detach().clone()
    X, y = torch.rand(20, 64), torch.randint(0, 2, (20,))
    fit(model, torch.optim.Adam(trainable_parameters(model), lr=0.01), X, y, epochs=20)
    assert all(torch.equal(b, p) for b, p in zip(before, model.backbone.parameters()))
    assert not torch.equal(head_before, model.head.weight), "the head should have learned"


def test_extract_features() -> None:
    backbone = make_backbone()
    X = torch.rand(9, 64)
    features = extract_features(backbone, X)
    assert tuple(features.shape) == (9, N_FEATURES)
    assert not features.requires_grad, "wrap in torch.no_grad()"
    assert torch.allclose(features, backbone(X).detach())


# --- Task 4 -----------------------------------------------------------------
def test_fine_tune_optimiser_unfreezes_and_uses_two_learning_rates() -> None:
    model = TransferModel(make_backbone(), N_FEATURES, 2)
    freeze(model.backbone)
    optimiser = fine_tune_optimiser(model, head_lr=0.01, backbone_lr=0.001)
    assert all(p.requires_grad for p in model.backbone.parameters())
    groups = optimiser.param_groups
    assert len(groups) == 2
    assert {g["lr"] for g in groups} == {0.01, 0.001}
    head_ids = {id(p) for p in model.head.parameters()}
    head_group = next(g for g in groups if g["lr"] == 0.01)
    assert {id(p) for p in head_group["params"]} == head_ids, "the head gets the larger learning rate"


# --- The payoff -------------------------------------------------------------
def test_transfer_beats_training_from_scratch_with_few_labels() -> None:
    """25 labelled images per class. Same architecture; only the starting point differs."""
    runs = [parity_experiment(per_class=25, seed=s) for s in range(3)]
    mean = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    assert mean["frozen backbone"] > mean["from scratch"] + 0.02
    assert mean["frozen backbone"] > 0.85
    assert mean["then fine-tuned"] >= mean["frozen backbone"] - 0.01
