# PyTorch's own type hints leave a few parameters untyped; that is not your code's problem.
# pyright: reportUnknownMemberType=false
"""Lesson 10 - Transfer learning: start from what another model already learned.

Read THEORY.md first. Then fill in each TODO, in order, and run:

    pytest lesson10_transfer -x

The setup (THEORY §3):
  source task  - which digit is this?  1,437 labelled images  (the "pretraining")
  target task  - is this digit even or odd?  only a handful of labelled images,
                 and they are DIFFERENT images from the source task's
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from nnlib.data import load_digits
from nnlib.types import FloatArray, IntArray

N_FEATURES = 32  # the backbone's output size


# --- GIVEN: the pretrained backbone ------------------------------------------
def make_backbone() -> nn.Sequential:
    """GIVEN. Image (64 pixels) -> 32 features. No classification layer."""
    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, N_FEATURES), nn.ReLU())


def pretrain_backbone(X: FloatArray, digits: IntArray, seed: int, epochs: int = 20) -> nn.Sequential:
    """GIVEN. Train backbone + a temporary 10-digit head (lesson 9's loop), then throw the head away.

    What survives is the backbone: a function that turns pixels into 32 numbers
    that turned out to be useful for telling digits apart.
    """
    torch.manual_seed(seed)
    backbone = make_backbone()
    model = nn.Sequential(backbone, nn.Linear(N_FEATURES, 10))
    data = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(digits, dtype=torch.int64))
    loader = DataLoader(data, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimiser = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            optimiser.zero_grad()
            loss_fn(model(X_batch), y_batch).backward()
            optimiser.step()
    return backbone


# --- Task 1: a few labelled examples (THEORY §3) -----------------------------
@dataclass(frozen=True)
class FewShot:
    """GIVEN."""

    X_train: Tensor  # (per_class · n_classes, 64) float32
    y_train: Tensor  # (per_class · n_classes,)    int64
    X_test: Tensor  # (everything else, 64)
    y_test: Tensor  # (everything else,)


def few_shot_split(X: FloatArray, labels: IntArray, per_class: int, rng: np.random.Generator) -> FewShot:
    """Pick `per_class` random examples of each class for training; everything else is test.

    For each class c in np.unique(labels): rng.choice(indices of c, per_class, replace=False).
    Return float32 / int64 tensors.
    """
    # TODO
    raise NotImplementedError("TODO: implement few_shot_split()")


# --- Task 2: freezing (THEORY §4) --------------------------------------------
def freeze(module: nn.Module) -> None:
    """Stop every parameter in `module` from learning: p.requires_grad = False."""
    # TODO
    raise NotImplementedError("TODO: implement freeze()")


def trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    """Only the parameters that still have requires_grad == True. Give these to the optimiser."""
    # TODO
    raise NotImplementedError("TODO: implement trainable_parameters()")


# --- Task 3: a new head on an old backbone (THEORY §4) -----------------------
class TransferModel(nn.Module):
    """backbone (pretrained) → head (new, nn.Linear(n_features, n_classes))."""

    backbone: nn.Module
    head: nn.Linear

    def __init__(self, backbone: nn.Module, n_features: int, n_classes: int) -> None:
        super().__init__()
        # TODO: store self.backbone, create self.head
        raise NotImplementedError("TODO: implement __init__()")

    def forward(self, X: Tensor) -> Tensor:
        # TODO
        raise NotImplementedError("TODO: implement forward()")


def extract_features(backbone: nn.Module, X: Tensor) -> Tensor:
    """Run the backbone once over X, without tracking gradients, in eval mode.

    With a frozen backbone, these features never change, so you can compute them
    once and train the head on them directly: much faster on a large backbone.
    """
    # TODO
    raise NotImplementedError("TODO: implement extract_features()")


# --- Task 4: fine-tuning (THEORY §5) -----------------------------------------
def fine_tune_optimiser(model: TransferModel, head_lr: float, backbone_lr: float) -> torch.optim.Adam:
    """Unfreeze the backbone, then return Adam with two parameter groups:

        [{"params": model.head.parameters(), "lr": head_lr},
         {"params": model.backbone.parameters(), "lr": backbone_lr}]

    The backbone gets the smaller rate so it is adjusted, not overwritten.
    """
    # TODO
    raise NotImplementedError("TODO: implement fine_tune_optimiser()")


# --- GIVEN: training and scoring ---------------------------------------------
def fit(model: nn.Module, optimiser: torch.optim.Optimizer, X: Tensor, y: Tensor, epochs: int) -> list[float]:
    """GIVEN. Full-batch training (the few-shot set is tiny). Returns the loss history."""
    loss_fn = nn.CrossEntropyLoss()
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        optimiser.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimiser.step()
        history.append(loss.item())
    return history


def accuracy(model: nn.Module, X: Tensor, y: Tensor) -> float:
    """GIVEN."""
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(dim=1) == y).float().mean().item()


def parity_experiment(per_class: int, seed: int) -> dict[str, float]:
    """GIVEN. Train three ways on the same few labelled images; report test accuracy for each."""
    d = load_digits()
    backbone = pretrain_backbone(d.X_train, d.y_train, seed=seed)  # source: the training images
    data = few_shot_split(d.X_val, d.y_val % 2, per_class, np.random.default_rng(seed))  # target: other images

    results: dict[str, float] = {}

    torch.manual_seed(seed)
    scratch = TransferModel(make_backbone(), N_FEATURES, 2)  # same shape, random backbone
    fit(scratch, torch.optim.Adam(scratch.parameters(), lr=0.01), data.X_train, data.y_train, 300)
    results["from scratch"] = accuracy(scratch, data.X_test, data.y_test)

    torch.manual_seed(seed)
    frozen = TransferModel(backbone, N_FEATURES, 2)
    freeze(frozen.backbone)
    fit(frozen, torch.optim.Adam(trainable_parameters(frozen), lr=0.01), data.X_train, data.y_train, 300)
    results["frozen backbone"] = accuracy(frozen, data.X_test, data.y_test)

    fit(frozen, fine_tune_optimiser(frozen, head_lr=0.005, backbone_lr=0.0005), data.X_train, data.y_train, 100)
    results["then fine-tuned"] = accuracy(frozen, data.X_test, data.y_test)
    return results


if __name__ == "__main__":
    for per_class in (5, 25):
        print(f"{per_class} labelled images per class (even / odd):")
        for name, acc in parity_experiment(per_class, seed=0).items():
            print(f"  {name:16s} test accuracy {acc:.3f}")
